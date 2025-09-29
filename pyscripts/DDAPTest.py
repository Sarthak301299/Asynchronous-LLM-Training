import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import datasets
import typing, collections
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group, barrier, all_gather
from typing import Optional
from types import NoneType
from config import Config
from trainer import Trainer
import time

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
    ddp_setup()
    config = Config(dataloaderprocs=5,batch_size=8,epochs=3,learning_rate=5e-5)
    #Get Model Online
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #Get Dataset online
    train_dataset = load_dataset_split("nyu-mll/glue", "mrpc", split="train", num_proc=4)
    test_dataset = load_dataset_split("nyu-mll/glue", "mrpc", split="test", num_proc=4)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=128,  # Adjust based on GPU memory
            return_tensors="pt"
        )
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["sentence1","sentence2"])
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["sentence1","sentence2"])
    tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer = Trainer(config,model,optimizer,train_dataset=tokenized_train_dataset,test_dataset=tokenized_test_dataset, useDDAP=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    barrier(device_ids=[trainer.local_rank])

    start_time = time.time()

    trainer.train(eval_every_epoch=False)
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