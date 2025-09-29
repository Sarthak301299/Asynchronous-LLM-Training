import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Dict, Iterator
import math, os
import numpy as np

class WeightBucketReducer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, bucket_size_bytes, process_group: ProcessGroup=None):
        self.model = model
        self.optimizer = optimizer
        self.bucket_size_bytes = bucket_size_bytes
        self.buckets: List[List[torch.nn.Parameter]] = []
        self.current_bucket_size = 0
        self.current_bucket: List[torch.nn.Parameter] = []
        self.process_group = dist.group.WORLD if process_group == None else process_group
        self._create_buckets()
        self.base_tensor = torch.cat([param.data.view(-1) for param in self.model.parameters() if param.requires_grad])
        self.exp_avg_base = None
        self.exp_avg_sq_base = None
        self.local_rank = int(os.environ.get("LOCAL_RANK")) if os.environ.get("LOCAL_RANK") != None else 0
        self.global_rank = int(os.environ.get("RANK")) if os.environ.get("RANK") != None else 0
        self.world_size = int(os.environ.get("WORLD_SIZE")) if os.environ.get("WORLD_SIZE") != None else 0
        #self.base_grads = torch.zeros_like(self.base_tensor)
        #self.sum_grads = torch.zeros_like(self.base_tensor)
    
    def _get_param_size_bytes(self, param: torch.nn.Parameter) -> int:
        return param.numel() * param.element_size()
    
    def _create_buckets(self):
        self.buckets = []
        self.current_bucket = []
        self.current_bucket_size = 0

        params = sorted(
            [p for p in self.model.parameters() if p.requires_grad],
            key=lambda p: self._get_param_size_bytes(p),
            reverse=True
        )

        for param in params:
            param_size = self._get_param_size_bytes(param)
            
            if self.current_bucket_size + param_size > self.bucket_size_bytes:
                if self.current_bucket:
                    self.buckets.append(self.current_bucket)
                self.current_bucket = [param]
                self.current_bucket_size = param_size
            else:
                self.current_bucket.append(param)
                self.current_bucket_size += param_size
        
        if self.current_bucket:
            self.buckets.append(self.current_bucket)
    
    def update_grads(self):
        pass
        #grads = torch.cat([param.grad.view(-1) for param in self.model.parameters() if param.requires_grad])
        #self.sum_grads += grads
    
    def _allreduce_bucket(self, bucket: List[torch.nn.Parameter]):
        total_size = sum(self._get_param_size_bytes(p) for p in bucket)
        bucket_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)

        offset = 0
        param_views = []
        for param in bucket:
            numel = param.numel()
            bucket_tensor[offset:offset + numel].copy_(param.data.view(-1))
            param_views.append((param, offset, numel))
            offset += numel
        
        dist.all_reduce(bucket_tensor, op=dist.ReduceOp.SUM, group=self.process_group)

        world_size = dist.get_world_size(self.process_group)
        bucket_tensor.div_(world_size)

        for param, offset, numel in param_views:
            param.data.copy_(bucket_tensor[offset:offset + numel].view_as(param.data))

    def _bcast_bucket(self, bucket: List[torch.nn.Parameter], root):
        total_size = sum(self._get_param_size_bytes(p) for p in bucket)
        bucket_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)

        offset = 0
        param_views = []
        for param in bucket:
            numel = param.numel()
            bucket_tensor[offset:offset + numel].copy_(param.data.view(-1))
            param_views.append((param, offset, numel))
            offset += numel

        dist.broadcast(bucket_tensor, src=root, group=self.process_group)

        world_size = dist.get_world_size(self.process_group)

        for param, offset, numel in param_views:
            param.data.copy_(bucket_tensor[offset:offset + numel].view_as(param.data))

    def _exchange_bucket(self, bucket, source, dest, world_size, global_rank):
        total_size = sum(self._get_param_size_bytes(p) for p in bucket)
        bucket_send_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)
        bucket_recv_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)
        offset = 0
        param_views = []
        for param in bucket:
            numel = param.numel()
            bucket_send_tensor[offset:offset + numel].copy_(param.data.view(-1))
            param_views.append((param, offset, numel))
            offset += numel
        if source > global_rank:
            sreq = dist.isend(bucket_send_tensor, dst=dest)
            rreq = dist.irecv(bucket_recv_tensor, src=source)
        else:
            rreq = dist.irecv(bucket_recv_tensor, src=source)
            sreq = dist.isend(bucket_send_tensor, dst=dest)
        rreq.wait()
        sreq.wait()

        for param, offset, numel in param_views:
            param.data.copy_(bucket_recv_tensor[offset:offset + numel].view_as(param.data))

    def _send_bucket(self, bucket, dest):
        total_size = sum(self._get_param_size_bytes(p) for p in bucket)
        bucket_send_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)
        offset = 0
        param_views = []
        for param in bucket:
            numel = param.numel()
            bucket_send_tensor[offset:offset + numel].copy_(param.data.view(-1))
            param_views.append((param, offset, numel))
            offset += numel
        for dst in dest:
            dist.send(bucket_send_tensor, dst=dst)

    def _recv_bucket(self, bucket, source, dest):
        total_size = sum(self._get_param_size_bytes(p) for p in bucket)
        bucket_recv_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)
        offset = 0
        param_views = []
        for param in bucket:
            numel = param.numel()
            param_views.append((param, offset, numel))
            offset += numel
        
        dist.recv(bucket_recv_tensor, src=source)

        for param, offset, numel in param_views:
            param.data.copy_(bucket_recv_tensor[offset:offset + numel].view_as(param.data))
    
    def generate_diffs(self, mysendpairs, myrecvpairs, world_size, global_rank):
        diffs = np.zeros((world_size, world_size))

        for bucket in self.buckets:
            total_size = sum(self._get_param_size_bytes(p) for p in bucket)
            bucket_send_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)
            bucket_recv_tensor = torch.empty(total_size, dtype=bucket[0].dtype, device=bucket[0].device)
            offset = 0
            param_views = []
            for param in bucket:
                numel = param.numel()
                bucket_send_tensor[offset:offset + numel].copy_(param.data.view(-1))
                param_views.append((param, offset, numel))
                offset += numel
            
            sreqs = []
            if global_rank < world_size/2:
                for pair in mysendpairs:
                    sreq = dist.isend(bucket_send_tensor, dst=pair)
                    sreqs.append(sreq)
                
                for pair in myrecvpairs:
                    rreq = dist.irecv(bucket_recv_tensor,pair)
                    rreq.wait()
                    for param, offset, numel in param_views:
                        local_data = param.data
                        remote_data = bucket_recv_tensor[offset:offset + numel].view_as(param.data)
                        diffs[global_rank][pair] += torch.sum((local_data - remote_data) ** 2).item()
                for sreq in sreqs:
                    sreq.wait()
            else:
                for pair in myrecvpairs:
                    rreq = dist.irecv(bucket_recv_tensor,pair)
                    rreq.wait()
                    for param, offset, numel in param_views:
                        local_data = param.data
                        remote_data = bucket_recv_tensor[offset:offset + numel].view_as(param.data)
                        diffs[global_rank][pair] += torch.sum((local_data - remote_data) ** 2).item()
                for pair in mysendpairs:
                    sreq = dist.isend(bucket_send_tensor, dst=pair)
                    sreqs.append(sreq)
                for sreq in sreqs:
                    sreq.wait()
        for pair in myrecvpairs:
            diffs[global_rank][pair] = torch.sqrt(torch.tensor(diffs[global_rank][pair])).item()
            diffs[pair][global_rank] = diffs[global_rank][pair]
        
        return diffs
    
    def exchange_weights(self, source, dest, world_size, global_rank):
        if source == -1 or dest == -1:
            return
        for bucket in self.buckets:
            self._exchange_bucket(bucket, source, dest, world_size, global_rank)

    def bcast_weights(self,root):
        for bucket in self.buckets:
            self._bcast_bucket(bucket, root)
    
    def send_weights(self, dest):
        for bucket in self.buckets:
            self._send_bucket(bucket, dest)
    
    def recv_weights(self, source):
        for bucket in self.buckets:
            self._recv_bucket(bucket, source)
        
    def allreduce_weights(self):
        if not dist.is_initialized():
            raise RuntimeError("Distributed backend is not initialized")
        
        for bucket in self.buckets:
            self._allreduce_bucket(bucket)

    def average_optimizer_state(self):
        with torch.no_grad():
            if self.exp_avg_base == None:
                offset = 0
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        state = self.optimizer.state[param]
                        if 'exp_avg' in state:
                            numel = state['exp_avg'].numel()
                            offset+=numel
                self.exp_avg_base = torch.zeros(offset).to(self.local_rank)
                self.exp_avg_sq_base = torch.zeros(offset).to(self.local_rank)
            offset = 0
            exp_avg_avg = torch.zeros_like(self.exp_avg_base)
            exp_avg_sq_avg = torch.zeros_like(self.exp_avg_sq_base)
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    state = self.optimizer.state[param]
                    if 'momentum_buffer' in state:
                        dist.all_reduce(state['momentum_buffer'])
                        state['momentum_buffer'] /= dist.get_world_size()
                    if 'exp_avg' in state:
                        numel = state['exp_avg'].numel()
                        exp_avg_avg[offset:offset+numel] = state['exp_avg'].view(-1)
                        exp_avg_sq_avg[offset:offset+numel] = state['exp_avg_sq'].view(-1)
                        offset += numel
            dist.all_reduce(exp_avg_avg)
            dist.all_reduce(exp_avg_sq_avg)

            offset = 0
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    state = self.optimizer.state[param]
                    if 'exp_avg' in state:
                        numel = state['exp_avg'].numel()
                        state['exp_avg'].copy_(exp_avg_avg[offset:offset + numel].view_as(state['exp_avg']))
                        state['exp_avg_sq'].copy_(exp_avg_sq_avg[offset:offset + numel].view_as(state['exp_avg_sq']))
                        offset += numel
    
    def merge_optimizer_state(self):
        with torch.no_grad():
            lambda_scale = 1.0
            if self.exp_avg_base == None:
                offset = 0
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        state = self.optimizer.state[param]
                        if 'exp_avg' in state:
                            numel = state['exp_avg'].numel()
                            offset+=numel
                self.exp_avg_base = torch.zeros(offset).to(self.local_rank)
                self.exp_avg_sq_base = torch.zeros(offset).to(self.local_rank)
            offset = 0
            exp_avg_diff = torch.zeros_like(self.exp_avg_base)
            exp_avg_sq_diff = torch.zeros_like(self.exp_avg_sq_base)
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    state = self.optimizer.state[param]
                    if 'momentum_buffer' in state:
                        dist.all_reduce(state['momentum_buffer'])
                        state['momentum_buffer'] /= dist.get_world_size()
                    if 'exp_avg' in state:
                        numel = state['exp_avg'].numel()
                        exp_avg_diff[offset:offset+numel] = state['exp_avg'].view(-1)
                        exp_avg_sq_diff[offset:offset+numel] = state['exp_avg_sq'].view(-1)
                        offset += numel
            sum_exp_avg_diff = exp_avg_diff.clone()
            sum_exp_avg_sq_diff = exp_avg_sq_diff.clone()
            dist.all_reduce(sum_exp_avg_diff)
            dist.all_reduce(sum_exp_avg_sq_diff)
            
            sum_exp_avg_diff = torch.sign(sum_exp_avg_diff)
            sum_exp_avg_diff = sum_exp_avg_diff == torch.sign(exp_avg_diff)
            exp_avg_diff = exp_avg_diff*sum_exp_avg_diff
            sum_exp_avg_diff = sum_exp_avg_diff.int()

            sum_exp_avg_sq_diff = torch.sign(sum_exp_avg_sq_diff)
            sum_exp_avg_sq_diff = sum_exp_avg_sq_diff == torch.sign(exp_avg_sq_diff)
            exp_avg_sq_diff = exp_avg_sq_diff*sum_exp_avg_sq_diff
            sum_exp_avg_sq_diff = sum_exp_avg_sq_diff.int()
            dist.all_reduce(exp_avg_diff, op=dist.ReduceOp.SUM, group=self.process_group)
            dist.all_reduce(sum_exp_avg_diff, op=dist.ReduceOp.SUM, group=self.process_group)
            dist.all_reduce(exp_avg_sq_diff, op=dist.ReduceOp.SUM, group=self.process_group)
            dist.all_reduce(sum_exp_avg_sq_diff, op=dist.ReduceOp.SUM, group=self.process_group)

            exp_avg_diff = torch.where(sum_exp_avg_diff != 0, exp_avg_diff / sum_exp_avg_diff, torch.zeros_like(exp_avg_diff))
            self.exp_avg_base = self.exp_avg_base + (lambda_scale*exp_avg_diff)

            exp_avg_sq_diff = torch.where(sum_exp_avg_sq_diff != 0, exp_avg_sq_diff / sum_exp_avg_sq_diff, torch.zeros_like(exp_avg_sq_diff))
            self.exp_avg_sq_base = self.exp_avg_sq_base + (lambda_scale*exp_avg_sq_diff)

            offset = 0
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    state = self.optimizer.state[param]
                    if 'momentum_buffer' in state:
                        dist.all_reduce(state['momentum_buffer'])
                        state['momentum_buffer'] /= dist.get_world_size()
                    if 'exp_avg' in state:
                        numel = state['exp_avg'].numel()
                        state['exp_avg'].copy_(self.exp_avg_base[offset:offset + numel].view_as(state['exp_avg']))
                        state['exp_avg_sq'].copy_(self.exp_avg_sq_base[offset:offset + numel].view_as(state['exp_avg_sq']))
                        offset += numel
    
    def perform_ties_merge(self, epoch, max_epoch, grads=None):
        density = 0.2
        base = 4
        sample_size = 100_000
        dare_prob = 0.9
        #expo = (0.8*(base ** (epoch/max_epoch) - 1)/(base-1)) + 0.1

        expo = 0
        lambda_scale = 1.0 - expo
        diff = torch.cat([param.data.view(-1) for param in self.model.parameters() if param.requires_grad])
        diff = diff - self.base_tensor

        #mask = torch.bernoulli(torch.full_like(diff,1-dare_prob))
        #diff = diff*mask
        #diff /= (1-dare_prob)

        #print(f"min diff {diff.abs().min()} sec min {0} max diff {diff.abs().max()}")
        #diff = torch.where(diff.abs() > (diff.abs().max()/10), diff, 0)

        #diff_grads = self.sum_grads

        #offset = 0
        #for param in self.model.parameters():
        #    if not param.requires_grad:
        #        continue
        #    numel = param.numel()
        #    param_slice = diff[offset:offset+numel]
        #    abs_param = param_slice.abs()
        #    thresh = torch.quantile(abs_param[torch.randperm(numel)[:sample_size]] if numel > sample_size else abs_param, 1 - density) if param_slice.numel() > 0 else 0.0
        #    param_slice.mul_((abs_param >= thresh).float())
        #    offset += numel

        sum_diff = diff.clone()
        #sum_diff = torch.sign(sum_diff)
        #sum_diff = torch.abs(diff_grads)*diff
        
        dist.all_reduce(sum_diff, op=dist.ReduceOp.SUM, group=self.process_group)
        sum_diff = torch.sign(sum_diff)
        sum_diff = sum_diff == torch.sign(diff)
        diff = diff*sum_diff
        sum_diff = sum_diff.int()
        dist.all_reduce(diff, op=dist.ReduceOp.SUM, group=self.process_group)
        dist.all_reduce(sum_diff, op=dist.ReduceOp.SUM, group=self.process_group)
        diff = torch.where(sum_diff != 0, diff / sum_diff, torch.zeros_like(diff))
        self.base_tensor = self.base_tensor + (lambda_scale*diff)
        offset = 0
        for param in self.model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param.data.copy_(self.base_tensor[offset:offset + numel].view_as(param.data))
                offset += numel
        
        #self.base_grads = self.sum_grads

    
    def get_buckets(self) -> List[List[torch.nn.Parameter]]:
        return self.buckets
