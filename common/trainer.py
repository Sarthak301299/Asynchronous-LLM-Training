import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os, random, math
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from distasync import DistributedDataAsynchronousParallel as DDAP
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, config, model, optimizer, tokenizer=None, dataset: Dataset=None, train_dataset: Dataset=None, test_dataset: Dataset=None, data_collator=None, seed=0, useDDAP=False):
        self.config = config
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seed = seed
        self.local_rank = int(os.environ.get("LOCAL_RANK")) if os.environ.get("LOCAL_RANK") != None else 0
        self.global_rank = int(os.environ.get("RANK")) if os.environ.get("RANK") != None else 0
        if train_dataset == None and test_dataset == None:
            train_len = int(len(dataset) * self.config.train_test_split)
            train_dataset, test_dataset = random_split(dataset,[train_len,len(dataset)-train_len])
        self.train_loader = self._prepare_dataloader(train_dataset,DistributedSampler(train_dataset,seed=seed) if self.config.data_partitions > 1 else None, collate_fn=data_collator)
        self.test_loader = self._prepare_dataloader(test_dataset,DistributedSampler(test_dataset,seed=seed) if self.config.data_partitions > 1 else None, collate_fn=data_collator) if test_dataset else None
        self.epochs_run = 0
        self.train_steps_per_epoch = len(self.train_loader)
        self.test_steps_per_epoch = len(self.test_loader) if test_dataset else 0
        self.config.total_steps = self.config.epochs * len(self.train_loader)
        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.config.total_steps)
        self.scaler = GradScaler('cuda')
        self.model = model.to(self.local_rank)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.useDDAP = useDDAP
        if useDDAP:
            self.model, self.train_loader = DDAP(self.model, self.optimizer, self.train_loader,
                                                 device_ids=[self.local_rank],
                                                 steps_per_epoch = self.train_steps_per_epoch,
                                                 epochs = self.config.epochs)
        else:
            self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def worker_init_fn(self, worker_id):
        np.random.seed(self.seed + worker_id)
    
    def _prepare_dataloader(self, dataset: Dataset, sampler=None, collate_fn=None):
        return DataLoader(dataset,batch_size=self.config.batch_size,
                          pin_memory=True,
                          shuffle=(sampler==None),
                          num_workers=self.config.dataloaderprocs,
                          sampler=sampler, collate_fn=collate_fn,
                          worker_init_fn=self.worker_init_fn)
    
    def _compute_mlm_metrics(self, predictions, labels, ignore_index=-100):
        ignore_index_tensor = torch.tensor(ignore_index).to(self.local_rank)
        mask = labels != ignore_index_tensor
        correct = (predictions == labels) & mask
        num_correct = correct.sum().float()
        num_masked = mask.sum().float()
        return num_correct,num_masked
    
    def evaluate(self):
        final_eval_loss = 0
        total_correct = 0
        total_samples = 0
        self.model.eval()
        if self.test_loader:
            with torch.no_grad():
                for batch in tqdm(self.test_loader):
                    input_ids = batch["input_ids"].to(self.local_rank)
                    attention_mask = batch["attention_mask"].to(self.local_rank)
                    labels = batch["labels"].to(self.local_rank)
                    with autocast('cuda'):
                        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                        loss = outputs.loss
                    final_eval_loss += loss.item()
                    prediction_logits = outputs.logits
                    predictions = torch.argmax(prediction_logits, dim=-1)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
        
        local_loss = final_eval_loss/len(self.test_loader)
        final_eval_loss_tensor = torch.tensor(final_eval_loss,device=self.local_rank)
        dist.all_reduce(final_eval_loss_tensor, op=dist.ReduceOp.SUM)
        final_avg_eval_loss = final_eval_loss_tensor.item() / (len(self.test_loader)*self.config.total_gpus)
        
        local_accuracy = total_correct/total_samples
        total_correct_tensor = torch.tensor(total_correct, device=self.local_rank)
        total_samples_tensor = torch.tensor(total_samples, device=self.local_rank)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        global_accuracy = total_correct_tensor.item()/total_samples_tensor.item()

        if self.global_rank == 0:
            print(f"Final Loss: {final_avg_eval_loss:.4f}, Local Loss: {local_loss:.4f}")
            print(f"Accuracy: {global_accuracy:.4f}, Local Accuracy {local_accuracy:.4f}")

    def test(self, output_file="sst2_predictions.tsv"):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                input_ids = batch['input_ids'].to(self.local_rank)
                attention_mask = batch['attention_mask'].to(self.local_rank)
                idx = batch['idx'].cpu().numpy()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                for i, pred in zip(idx, batch_preds):
                    predictions.append((i, pred))
        
        with open(output_file, 'w') as f:
            f.write("index\tprediction\n")
            for idx, pred in predictions:
                f.write(f"{idx}\t{pred}\n")
        print(f"Predictions saved to {output_file}")

    def train(self, eval_every_epoch: bool=True):
        for epoch in range(self.epochs_run, self.config.epochs):
            epoch += 1
            self.model.train()
            train_loss = 0
            iteration = 0
            for batch in tqdm(self.train_loader):
                input_ids = batch["input_ids"].to(self.local_rank)
                attention_mask = batch["attention_mask"].to(self.local_rank)
                labels = batch["labels"].to(self.local_rank)
                
                self.optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                scale = self.scaler.get_scale()
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.config.grad_norm_clip * scale)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                iteration += 1
                train_loss += loss.item()
            
            train_loss_tensor = torch.tensor(train_loss, device=self.local_rank)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / (len(self.train_loader)*self.config.total_gpus)
            if self.global_rank == 0:
                print(f"Rank {self.global_rank} | Epoch {epoch} | Train Loss {avg_train_loss:.4f} Local Loss {train_loss/len(self.train_loader):.4f}")
            
            self.model.eval()
            if self.test_loader and eval_every_epoch:
                eval_loss = 0
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader):
                        input_ids = batch["input_ids"].to(self.local_rank)
                        attention_mask = batch["attention_mask"].to(self.local_rank)
                        labels = batch["labels"].to(self.local_rank)
                        with autocast('cuda'):
                            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                            loss = outputs.loss
                            eval_loss += loss.item()
                        prediction_logits = outputs.logits
                        predictions = torch.argmax(prediction_logits, dim=-1)
                        total_correct, total_samples = self._compute_mlm_metrics(predictions,labels)
                        #total_correct += (predictions == labels).sum().item()
                        #total_samples += labels.size(0)
            
            local_loss = eval_loss/len(self.test_loader)
            eval_loss_tensor = torch.tensor(eval_loss,device=self.local_rank)
            dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
            avg_eval_loss = eval_loss_tensor.item() / (len(self.test_loader)*self.config.total_gpus)
            
            local_accuracy = total_correct/total_samples
            total_correct_tensor = torch.tensor(total_correct, device=self.local_rank)
            total_samples_tensor = torch.tensor(total_samples, device=self.local_rank)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            global_accuracy = total_correct_tensor.item()/total_samples_tensor.item()

            if self.global_rank == 0:
                print(f"Rank {self.global_rank} | Loss: {avg_eval_loss:.4f}, Local Loss: {local_loss:.4f}")
                print(f"Rank {self.global_rank} | Accuracy: {global_accuracy:.4f}, Local Accuracy {local_accuracy:.4f}")

        final_eval_loss = 0
        total_correct = 0
        total_samples = 0
        self.model.eval()
        if self.test_loader:
            with torch.no_grad():
                for batch in tqdm(self.test_loader):
                    input_ids = batch["input_ids"].to(self.local_rank)
                    attention_mask = batch["attention_mask"].to(self.local_rank)
                    labels = batch["labels"].to(self.local_rank)
                    with autocast('cuda'):
                        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
                        loss = outputs.loss
                        final_eval_loss += loss.item()
                    prediction_logits = outputs.logits
                    predictions = torch.argmax(prediction_logits, dim=-1)
                    total_correct, total_samples = self._compute_mlm_metrics(predictions,labels)
                    #total_correct += (predictions == labels).sum().item()
                    #total_samples += labels.size(0)
        
        local_loss = final_eval_loss/len(self.test_loader)
        final_eval_loss_tensor = torch.tensor(final_eval_loss,device=self.local_rank)
        dist.all_reduce(final_eval_loss_tensor, op=dist.ReduceOp.SUM)
        final_avg_eval_loss = final_eval_loss_tensor.item() / (len(self.test_loader)*self.config.total_gpus)
        
        local_accuracy = total_correct/total_samples
        total_correct_tensor = torch.tensor(total_correct, device=self.local_rank)
        total_samples_tensor = torch.tensor(total_samples, device=self.local_rank)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        global_accuracy = total_correct_tensor.item()/total_samples_tensor.item()

        if self.global_rank == 0:
            print(f"Final Loss: {final_avg_eval_loss:.4f}, Local Loss: {local_loss:.4f}")
            print(f"Accuracy: {global_accuracy:.4f}, Local Accuracy {local_accuracy:.4f}")
