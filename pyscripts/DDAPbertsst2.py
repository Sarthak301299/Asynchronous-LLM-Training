import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorForLanguageModeling, BertConfig
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
from datetime import timedelta

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
    delta = timedelta(minutes=600)
    init_process_group(backend="nccl",timeout=delta)

# Custom Dataset wrapper for tokenized data
class SST2TokenizedDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset
        # Verify dataset format (for debugging)
        sample = self.dataset[0]
        if isinstance(sample['input_ids'], list):
            print("Warning: input_ids is a list, converting to tensor")
        if isinstance(sample['attention_mask'], list):
            print("Warning: attention_mask is a list, converting to tensor")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Ensure input_ids and attention_mask are tensors, even if stored as lists
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long) if isinstance(item['input_ids'], list) else item['input_ids']
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long) if isinstance(item['attention_mask'], list) else item['attention_mask']
        labels = torch.tensor(item['labels'], dtype=torch.long) if isinstance(item['labels'], (int, list)) else item['labels']
        idx = torch.tensor(item['idx'], dtype=torch.int32) if isinstance(item['idx'], (int, list)) else item['idx']
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'idx': idx
        }

# Custom collate function to ensure tensor batching
def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    idx = torch.stack([item['idx'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'idx': idx
    }

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
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #Get Dataset online
    #train_dataset = load_dataset("nyu-mll/glue","sst2", split='train')
    #test_dataset = load_dataset("nyu-mll/glue","sst2", split='validation')

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding='max_length',
            max_length=128
        )
    #tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=['sentence'])
    #tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=['sentence'])
    #tokenized_train_dataset = tokenized_train_dataset.rename_column("label","labels")
    #tokenized_test_dataset = tokenized_test_dataset.rename_column("label","labels")

    #tokenized_train_dataset.save_to_disk("/mnt/storage/sarthak.joshi/tokenized_sst2_train")
    #tokenized_test_dataset.save_to_disk("/mnt/storage/sarthak.joshi/tokenized_sst2_validation")
    tokenized_train_dataset = load_from_disk("/mnt/storage/sarthak.joshi/tokenized_sst2_train")
    tokenized_test_dataset = load_from_disk("/mnt/storage/sarthak.joshi/tokenized_sst2_validation")
    tokenized_train_dataset = SST2TokenizedDataset(tokenized_train_dataset)
    tokenized_test_dataset = SST2TokenizedDataset(tokenized_test_dataset)

    trainer = Trainer(config,model,optimizer,tokenizer=tokenizer,train_dataset=tokenized_train_dataset,test_dataset=tokenized_test_dataset, seed=seed, useDDAP=True)
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