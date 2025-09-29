import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
import os,sys,time
import datasets
from datasets import load_dataset
import typing, collections
from typing import Any, Optional, Union
from types import NoneType
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP

class Config:
    def __init__(self, train_test_split: float=0.8, learning_rate: float=0.01, epochs: int=1000, batch_size: int=32, gpu_per_model: int=1, dataloaderprocs: int=4, grad_norm_clip: float=1.0, weight_decay: float=0.01):
        self.train_test_split = train_test_split
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu_per_model = gpu_per_model
        self.total_gpus = get_world_size()
        self.data_partitions = int(self.total_gpus / self.gpu_per_model)
        self.dataloaderprocs = dataloaderprocs
        self.grad_norm_clip = grad_norm_clip
        self.weight_decay = weight_decay
        self.total_steps = 0

class Trainer:
    def __init__(self, config, model, optimizer, dataset: Dataset=None, train_dataset: Dataset=None, test_dataset: Dataset=None):
        self.config = config
        self.optimizer = optimizer
        self.dataset = dataset
        self.local_rank = int(os.environ.get("LOCAL_RANK")) if os.environ.get("LOCAL_RANK") != None else 0
        self.global_rank = int(os.environ.get("RANK")) if os.environ.get("RANK") != None else 0
        if train_dataset == None and test_dataset == None:
            train_len = int(len(dataset) * self.config.train_test_split)
            train_dataset, test_dataset = random_split(dataset,[train_len,len(dataset)-train_len])
        self.train_loader = self._prepare_dataloader(train_dataset,DistributedSampler(train_dataset) if self.config.data_partitions > 1 else None)
        self.test_loader = self._prepare_dataloader(test_dataset,DistributedSampler(test_dataset) if self.config.data_partitions > 1 else None) if test_dataset else None
        self.epochs_run = 0
        self.config.total_steps = self.config.epochs * len(self.train_loader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.config.total_steps)
        self.scaler = GradScaler('cuda')
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def _prepare_dataloader(self, dataset: Dataset, sampler=None):
        return DataLoader(dataset,batch_size=self.config.batch_size,
                          pin_memory=True,
                          shuffle=(sampler==None),
                          num_workers=self.config.dataloaderprocs,
                          sampler=sampler)

    def train(self):
        for epoch in range(self.epochs_run, self.config.epochs):
            epoch += 1
            self.model.train()
            train_loss = 0
            for iter, batch in enumerate(tqdm(self.train_loader)):
                input_ids = batch["input_ids"].to(self.local_rank)
                attention_mask = batch["attention_mask"].to(self.local_rank)
                labels = batch["label"].to(self.local_rank)

                self.optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                scale = self.scaler.get_scale()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip * scale)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.scheduler.step()
                train_loss += loss.item()

            train_loss_tensor = torch.tensor(train_loss, device=self.local_rank)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / (len(self.train_loader)*self.config.total_gpus)
            if self.global_rank == 0:
                print(f"Epoch {epoch} | Train Loss {avg_train_loss:.4f}")  

            self.model.eval()
            if self.test_loader:
                eval_loss = 0
                with torch.no_grad():
                    for iter, batch in enumerate(tqdm(self.test_loader)):
                        input_ids = batch["input_ids"].to(self.local_rank)
                        attention_mask = batch["attention_mask"].to(self.local_rank)
                        labels = batch["label"].to(self.local_rank)
                        with torch.no_grad():
                            with autocast('cuda'):
                                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                                loss = outputs.loss
                            eval_loss += loss.item()
                    eval_loss_tensor = torch.tensor(eval_loss, device=self.local_rank)
                    dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                    avg_eval_loss = eval_loss_tensor.item() / (len(self.test_loader)*self.config.total_gpus)
                    if self.global_rank == 0:
                        print(f"Epoch {epoch} | Average Eval Loss {avg_eval_loss:.4f}")

        all_predictions = []
        all_true_labels = []
        final_eval_loss = 0
        self.model.eval()
        if self.test_loader:
            with torch.no_grad():
                for iter, batch in enumerate(tqdm(self.test_loader)):
                    input_ids = batch["input_ids"].to(self.local_rank)
                    attention_mask = batch["attention_mask"].to(self.local_rank)
                    labels = batch["label"].to(self.local_rank)
                    with autocast('cuda'):
                        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                        loss = outputs.loss
                        final_eval_loss += loss.item()
                    all_predictions.extend(torch.argmax(outputs.logits,dim=-1).cpu().numpy())
                    all_true_labels.extend(labels.cpu().numpy())
        
        final_eval_loss_tensor = torch.tensor(final_eval_loss,device=self.local_rank)
        dist.all_reduce(final_eval_loss_tensor, op=dist.ReduceOp.SUM)
        final_avg_eval_loss = final_eval_loss_tensor.item() / (len(self.test_loader)*self.config.total_gpus)

        all_predictions_tensor = torch.tensor(all_predictions, device=self.local_rank)
        all_true_labels_tensor = torch.tensor(all_true_labels, device=self.local_rank)
        gathered_predictions = [torch.zeros_like(all_predictions_tensor) for _ in range(self.config.total_gpus)]
        gathered_true_labels = [torch.zeros_like(all_true_labels_tensor) for _ in range(self.config.total_gpus)]
        dist.all_gather(gathered_predictions,all_predictions_tensor)
        dist.all_gather(gathered_true_labels,all_true_labels_tensor)

        if self.global_rank == 0:
            all_predictions = torch.cat(gathered_predictions).cpu().numpy()
            all_true_labels = torch.cat(gathered_true_labels).cpu().numpy()
            total_acc = accuracy_score(all_true_labels,all_predictions)
            total_f1 = f1_score(all_true_labels,all_predictions)
            print(f"Final Loss: {final_avg_eval_loss:.4f}, Overall Accuracy: {total_acc:.4f}, Overall F1: {total_f1:.4f}")

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
    config = Config(dataloaderprocs=5,batch_size=8,epochs=5,learning_rate=5e-5)
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

    trainer = Trainer(config,model,optimizer,train_dataset=tokenized_train_dataset,test_dataset=tokenized_test_dataset)
    trainer.train()

    destroy_process_group()

if __name__ == "__main__":
    main()