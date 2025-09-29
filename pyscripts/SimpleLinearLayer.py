import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.nn.functional import mse_loss
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import os
import time


class Config:
    def __init__(self, train_test_split: float=0.8, learning_rate: float=0.01, epochs: int=1000, batch_size: int=32, gpu_per_model: int=1, data_partitions: int=1, dataloaderprocs: int=4, grad_norm_clip: float=1.0):
        self.train_test_split = train_test_split
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu_per_model = gpu_per_model
        self.data_partitions = data_partitions
        self.total_gpus = self.gpu_per_model * self.data_partitions
        self.dataloaderprocs = dataloaderprocs
        self.grad_norm_clip = grad_norm_clip

class Trainer:
    def __init__(self, config, model, optimizer, dataset):
        self.config = config
        self.optimizer = optimizer
        self.dataset = dataset
        self.local_rank = int(os.environ.get("LOCAL_RANK")) if os.environ.get("LOCAL_RANK") != None else 0
        self.global_rank = int(os.environ.get("RANK")) if os.environ.get("RANK") != None else 0
        train_len = int(len(dataset) * self.config.train_test_split)
        train_dataset, test_dataset = random_split(dataset,[train_len,len(dataset)-train_len])
        self.train_loader = self._prepare_dataloader(train_dataset,DistributedSampler(train_dataset) if self.config.data_partitions > 1 else None)
        self.test_loader = self._prepare_dataloader(test_dataset,DistributedSampler(test_dataset) if self.config.data_partitions > 1 else None)
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
    
    def _prepare_dataloader(self, dataset: Dataset, sampler=None):
        return DataLoader(dataset,batch_size=self.config.batch_size,
                          pin_memory=True,
                          shuffle=(sampler==None),
                          num_workers=self.config.dataloaderprocs,
                          sampler=sampler)
    
    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train):
            output = torch.squeeze(self.model(source))
            loss = mse_loss(output,targets)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        for iter, (source,targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source,targets,train)
            if iter % 100 == 0 and epoch % 100 == 0:
                print(f"Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss:.5f}")

    def train(self):
        for epoch in range(self.epochs_run, self.config.epochs):
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)
            self._run_epoch(epoch, self.test_loader, train=False)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.linear = nn.Linear(10,1)
    
    def forward(self, input):
        return self.linear(input)
    

def main():
    config = Config(dataloaderprocs=0,epochs=1000)
    model = Model(config)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    dataset = TensorDataset(X,y)
    trainer = Trainer(config,model,optimizer,dataset)
    trainer.train()

if __name__ == "__main__":
    main()