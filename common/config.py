from torch.distributed import get_world_size

class Config:
    def __init__(self, train_test_split: float=0.8, learning_rate: float=0.01, epochs: int=1000, batch_size: int=32, gpu_per_model: int=1, dataloaderprocs: int=4, grad_norm_clip: float=1.0, weight_decay: float=0.00, accumulation_steps: int=1):
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
        self.accumulation_steps = accumulation_steps
        self.total_steps = 0
