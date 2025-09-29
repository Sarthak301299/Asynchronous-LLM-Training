import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import os

# Custom dataset for toy example
class ToyDataset(Dataset):
    def __init__(self, num_samples=100):
        torch.manual_seed(42)  # Consistent seed for reproducibility
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Simple neural network model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(local_rank, global_rank):
    # Set up DDP
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size(dist.group.WORLD)

    # Create model and move it to DDP
    model = ToyModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Create dataset and DataLoader with DistributedSampler
    dataset = ToyDataset(num_samples=100)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)
    
    # Training loop
    for epoch in range(5):
        ddp_model.train()
        total_loss = 0.0
        for data, labels in dataloader:
            data, labels = data.to(local_rank), labels.to(local_rank)
            optimizer.zero_grad()

            # Forward pass
            outputs = ddp_model(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for this rank
        if global_rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/5], Average Loss: {avg_loss:.4f}")

    # Clean up
    dist.destroy_process_group()

def main():
    local_rank = int(os.environ.get("LOCAL_RANK")) if os.environ.get("LOCAL_RANK") != None else 0
    global_rank = int(os.environ.get("RANK")) if os.environ.get("RANK") != None else 0
    train(local_rank, global_rank)

if __name__ == "__main__":
    main()
