import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, BertConfig
import datasets
import typing, collections
from datasets import load_dataset, load_from_disk
from torch.distributed import init_process_group, destroy_process_group, barrier, all_gather
from typing import Optional
from types import NoneType
from config import Config
from trainer import Trainer
import time
import random
import numpy as np

def load_dataset_split(path: str,
                    name: typing.Optional[str] = None,
                    data_dir: typing.Optional[str] = None,
                    data_files: typing.Union[str, collections.abc.Sequence[str], collections.abc.Mapping[str, typing.Union[str, collections.abc.Sequence[str]]], NoneType] = None,
                    split: typing.Union[str, datasets.splits.Split, list[str], list[datasets.splits.Split], NoneType] = None,
                    num_proc: typing.Optional[int] = None) -> Optional[Dataset]:
    try:
        return load_dataset(path,name,data_dir,data_files,split,num_proc=num_proc)
    except (ValueError, FileNotFoundError):
        return None

def ddp_setup():
    init_process_group(backend="nccl")

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    ddp_setup()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = Config(dataloaderprocs=5,batch_size=16,epochs=6,learning_rate=5e-5)
    #Get Model Online
    model_name = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bertconfig = BertConfig.from_pretrained(model_name)
    model = BertForMaskedLM(config=bertconfig)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #Get Dataset online
    #train_dataset = load_dataset_split("allenai/c4", "en", split='train[:1%]')
    #test_dataset = load_dataset_split("allenai/c4", "en", split="validation")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
        )
    #tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=56, remove_columns=['text','timestamp','url'])
    #tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=56)

    tokenized_train_dataset = load_from_disk("/mnt/storage/sarthak.joshi/tokenized_padded_c4_train_3_percent")
    #tokenized_train_dataset = tokenized_train_dataset.select(range(int(0.03*len(tokenized_train_dataset))))
    tokenized_test_dataset = load_from_disk("/mnt/storage/sarthak.joshi/tokenized_padded_c4_test")
    #tokenized_test_dataset = tokenized_test_dataset.select(range(int(0.005*len(tokenized_test_dataset))))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = Trainer(config,model,optimizer,tokenizer=tokenizer,train_dataset=tokenized_train_dataset,test_dataset=tokenized_test_dataset, data_collator=data_collator, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    barrier(device_ids=[trainer.local_rank])

    start_time = time.time()

    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    barrier(device_ids=[trainer.local_rank])

    end_time = time.time()
    local_training_time = end_time - start_time

    local_time_tensor = torch.tensor(local_training_time, dtype=torch.float32, device=trainer.local_rank)
    all_times = [torch.zeros_like(local_time_tensor) for _ in range(config.total_gpus)]
    all_gather(all_times, local_time_tensor)

    all_times = [t.item() for t in all_times]

    if trainer.global_rank == 0:
        max_time = max(all_times)
        min_time = min(all_times)
        avg_time = sum(all_times) / len(all_times)
        print(f"Training times across {config.total_gpus} processes (GPU time):")
        for i, t in enumerate(all_times):
            print(f"  Rank {i}: {t:.2f} seconds")
        print(f"Max training time: {max_time:.2f} seconds")
        print(f"Min training time: {min_time:.2f} seconds")
        print(f"Average training time: {avg_time:.2f} seconds")
    
    
    destroy_process_group()

if __name__ == "__main__":
    main()
