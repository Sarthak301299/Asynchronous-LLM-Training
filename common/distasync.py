import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
from math import floor, ceil, log2
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.distributed as dist
from torch._utils import _get_device_index
from torch.autograd import Function, Variable
from torch.utils.data import random_split, Dataset, DataLoader
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs
from torch.distributed import barrier
from torch.utils._pytree import tree_flatten, tree_unflatten

from reducer import WeightBucketReducer

RPC_AVAILABLE = False
if dist.is_available():
    from torch.distributed.distributed_c10d import (
        _get_default_group,
        _rank_not_in_group,
        ReduceOp,
    )
    from torch.distributed.utils import (
        _alloc_storage,
        _cast_forward_inputs,
        _free_storage,
        _sync_module_states,
        _to_kwargs,
        _verify_param_shape_across_processes,
    )
if dist.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


__all__ = ["DistributedDataAsynchronousParallel"]

logger = logging.getLogger(__name__)

def _dump_DDP_relevant_env_vars():
    relevant_env_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_PORT",
        "MASTER_ADDR",
        "CUDA_VISIBLE_DEVICES",
        "GLOO_SOCKET_IFNAME",
        "GLOO_DEVICE_TRANSPORT",
        "NCCL_SOCKET_IFNAME",
        "TORCH_NCCL_BLOCKING_WAIT",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_IB_DISABLE",
        # More NCCL env vars:
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_BUFFSIZE",
        "NCCL_NTHREADS",
        "NCCL_RINGS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CHECKS_DISABLE",
        "NCCL_CHECK_POINTERS",
        "NCCL_LAUNCH_MODE",
        "NCCL_IB_HCA",
        "NCCL_IB_TIMEOUT",
        "NCCL_IB_RETRY_CNT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SL",
        "NCCL_IB_TC",
        "NCCL_IB_AR_THRESHOLD",
        "NCCL_IB_CUDA_SUPPORT",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NET_GDR_READ",
        "NCCL_SINGLE_RING_THRESHOLD",
        "NCCL_LL_THRESHOLD",
        "NCCL_TREE_THRESHOLD",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_IGNORE_CPU_AFFINITY",
        "NCCL_DEBUG_FILE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    formatted_output = ""
    for var in relevant_env_vars:
        value = os.environ[var] if var in os.environ else "N/A"
        formatted_output += f"env:{var}={value}\n"
    print(formatted_output)

def get_layer_flats(layers, device):
    """Flatten each layer's parameters into a vector."""
    flats = []
    for layer in layers:
        # Detach to avoid grad issues and move to device
        flat = torch.cat([p.view(-1).detach() for p in layer.parameters()]).to(device)
        flats.append(flat)
    return torch.stack(flats)

class _DDAPJoinHook(JoinHook):
    def __init__(self, ddap, divide_by_initial_world_size):
        assert isinstance(ddap, DistributedDataAsynchronousParallel), (
             "DDP join hook requires passing in a DistributedDataAsynchronousParallel "
             "instance as the state"
        )
        assert ddap.logger is not None
        ddap.logger._set_uneven_input_join()
        self.ddap = ddap
        self.ddap._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()
    
    def main_hook(self):
        ddap = self.ddap
    
    def post_hook(self, is_last_joiner: bool):
        self.ddap._sync_final_model(is_last_joiner)

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

class DataLoaderCallbackWrapper(DataLoader):
    def __init__(self, dataloader, on_complete_callback=None, on_step_complete=None):
        self.dataloader = dataloader
        self.on_complete_callback = on_complete_callback
        self.on_step_complete = on_step_complete
    
    def __iter__(self):
        # Get the iterator from the original DataLoader
        iterator = iter(self.dataloader)
        return CallbackIterator(iterator, self.on_complete_callback, self.on_step_complete, len(self.dataloader))

    def __getattr__(self, name):
        # Delegate any other attribute access to the original DataLoader
        return getattr(self.dataloader, name)
    
    def __len__(self):
        return len(self.dataloader)
    
class CallbackIterator:
    def __init__(self, iterator, on_complete_callback, on_step_complete, total_batches):
        self.iterator = iterator
        self.on_complete_callback = on_complete_callback
        self.on_step_complete = on_step_complete
        self.current_batch = 0
        self.total_batches = total_batches

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            if self.on_complete_callback is not None:
                self.on_complete_callback()
                self.current_batch = 0
            raise
        self.current_batch += 1
        if (self.current_batch > 1) and (self.on_step_complete is not None):
            self.on_step_complete(self.current_batch-2)
        return batch

class DistributedDataAsynchronousParallel(Module, Joinable):
    _active_ddap_module: Optional["DistributedDataAsynchronousParallel"] = None

    def __init__(
        self,
        module,
        optimizer,
        dataloader,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        init_sync=True,
        process_group=None,
        bucket_cap_mb=None,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        device_mesh=None,
        skip_all_reduce_unused_params=False,
        steps_per_allreduce = 0,
        steps_per_model_shift = 0,
        steps_per_epoch = 0,
        epochs = 1
    ):
        super().__init__()
        Joinable.__init__(self)
        self._use_python_reducer = False
        self.logger: Optional[dist.Logger] = None
        if bool(delay_all_reduce_named_params is not None) != bool(
            param_to_hook_all_reduce is not None
        ):
            self._log_and_throw(
                ValueError,
                "delay_all_reduce_named_params and param_to_hook_all_reduce "
                "need to be set at the same time.",
            )
        
        if process_group and device_mesh is not None:
            raise RuntimeError(
                "Cannot specify both process_group and device_mesh arguments."
            )
        elif process_group is None and device_mesh is None:
            self.process_group = _get_default_group()
        elif device_mesh is None:
            self.process_group = process_group
        else:
            if device_mesh.ndim != 1:
                raise RuntimeError(
                    f"Only 1D device mesh is supported, but got {device_mesh}."
                )
            self.device_mesh = device_mesh
            self.process_group = device_mesh.get_group(mesh_dim=0)
            from torch.distributed.device_mesh import _mesh_resources

            root_mesh = _mesh_resources.get_root_mesh(device_mesh)
            # if a root mesh is not the same as device_mesh,
            # meaning the device_mesh is sliced out from the root mesh.
            if root_mesh != device_mesh:
                # TODO: This is a temporary work around to enable DDP + TP.
                # We should do the logic in DDP so that the 2D implementation is
                # sound and the state_dict works out of the box.
                # This has to be done before check UninitializedParameter.
                from torch.distributed.tensor.parallel.ddp import (
                    _pre_dp_module_transform,
                )

                _pre_dp_module_transform(module)
            
        self._delay_all_reduce_params = []
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = set(module._ddp_params_and_buffers_to_ignore)
        else:
            self.parameters_to_ignore = set()
        if delay_all_reduce_named_params is not None:
            for name, param in delay_all_reduce_named_params:
                self.parameters_to_ignore.add(name)
                self._delay_all_reduce_params.append(param)
        
        self._module_parameters = [
            p
            for n, p in module.named_parameters()
            if n not in self.parameters_to_ignore
        ]
        if not any(p.requires_grad for p in self._module_parameters):
            if len(self._delay_all_reduce_params):
                logger.info("Delay the AllReduce of all parameters.")
            else:
                self._log_and_throw(
                    RuntimeError,
                    "DistributedDataAsynchronousParallel is not needed when a module "
                    "doesn't have any parameter that requires a gradient.",
                )
        
        if device_ids is not None and len(device_ids) > 1:
            self._log_and_throw(
                ValueError,
                "device_ids can only be None or contain a single element.",
            )

        self.is_multi_device_module = (
            len({p.device for p in self._module_parameters}) > 1
        )
        distinct_device_types = {
            p.device.type for p in self._module_parameters if p.device is not None
        }
        if len(distinct_device_types) != 1:
            self._log_and_throw(
                ValueError,
                "DistributedDataAsynchronousParallel's input module must be on "
                f"the same type of devices, but input module parameters locate in {distinct_device_types}.",
            )
        
        self.device_type = next(iter(distinct_device_types))

        if (
            device_ids is None
            or len(device_ids) == 0  # For backward compatibility.
            or self.device_type == "cpu"
            or self.is_multi_device_module
        ):
            if device_ids or output_device:
                self._log_and_throw(
                    ValueError,
                    "DistributedDataAsynchronousParallel device_ids and output_device arguments "
                    "only work with single-device/multiple-device GPU modules or CPU modules, "
                    f"but got device_ids {device_ids}, output_device {output_device}, "
                    f"and module parameters { ({p.device for p in self._module_parameters}) }.",  # noqa: E201,E202
                )

            self.device_ids = None
            self.output_device = None
        else:
            self.device_ids = [_get_device_index(x, True) for x in device_ids]

            if output_device is None:
                output_device = device_ids[0]

            self.output_device = _get_device_index(output_device, True)
        
        self.static_graph = False
        self.dim = dim
        self.module = module
        self.optimizer = optimizer
        self.dataloader = DataLoaderCallbackWrapper(dataloader, on_complete_callback=self._perform_synchronization,on_step_complete=self._stepwise_operation)
        self.device = next(iter(self._module_parameters)).device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view

        # Check that a module does not have Uninitialized parameters
        for param in self._module_parameters:
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                self._log_and_throw(
                    RuntimeError,
                    "Modules with uninitialized parameters can't be used with `DistributedDataAsynchronousParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # reduction bucket size
        if bucket_cap_mb is None:
            # default case (bucket cap is 25 MiB)
            bucket_cap_mb = 25
            self.bucket_bytes_cap_default = True
        else:
            self.bucket_bytes_cap_default = False
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )

        # Initialize gradient buffers and register all reduce hook
        self._delay_grad_buffer: Optional[torch.Tensor] = None
        self._delay_grad_views: list[torch.Tensor] = []
        self._delay_all_reduce_all_params = False

        self.skip_all_reduce_unused_params = skip_all_reduce_unused_params

        self._comm_hooks: list[tuple[Callable, object]] = []

        self._lazy_init_ran = False

        self.local_rank = int(os.environ.get("LOCAL_RANK")) if os.environ.get("LOCAL_RANK") != None else 0
        self.global_rank = int(os.environ.get("RANK")) if os.environ.get("RANK") != None else 0
        self.world_size = int(os.environ.get("WORLD_SIZE")) if os.environ.get("WORLD_SIZE") != None else 0

        self.steps_per_allreduce = steps_per_allreduce
        self.steps_per_model_shift = steps_per_model_shift
        self.steps_per_epoch = steps_per_epoch
        self.current_steps = 0
        self.merge_model_fn = self._merge_models
        self.shift_model_fn = self._shift_model_pairs
        #self.optimizer.register_step_post_hook(self._post_optimizer_hook)
        self.reducer = WeightBucketReducer(self.module, self.optimizer, self.bucket_bytes_cap)
        self.current_epoch = 0
        self.exchdeg = 0
        self.epochs = epochs

        self.steps_per_allreduce = self.steps_per_epoch
        self.steps_per_model_shift = 0#self.steps_per_epoch
    
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance, instance.dataloader

    @contextmanager
    @torch._disable_dynamo(recursive=False)
    def _inside_ddap_forward(self):
        DistributedDataAsynchronousParallel._active_ddap_module = self
        try:
            yield
        finally:
            DistributedDataAsynchronousParallel._active_ddap_module = None

    def _run_ddap_forward(self, *inputs, **kwargs):
        if self._use_python_reducer:
            return self.module(*inputs, **kwargs)  # type: ignore[index]
        else:
            with self._inside_ddap_forward():
                return self.module(*inputs, **kwargs)  # type: ignore[index]
            
    def _clear_grad_buffer(self):
        # Making param.grad points to the grad buffers before backward is based on the
        # assumption that the grad accumulation is done in place in autograd engine,
        # for some edge cases, if the grad accumulation in autograd engine is not in
        # place, then the param.grad and grad buffers are detached.
        if self._delay_grad_buffer is not None:
            # We batch zero_grad for all params by resetting the whole grad
            # buffer when the grad of all params is set to None.
            all_param_grad_none = all(
                param.grad is None for param in self._delay_all_reduce_params
            )

            for index, param in enumerate(self._delay_all_reduce_params):
                if param.grad is None:
                    param.grad = self._delay_grad_views[index]
                    if not all_param_grad_none:
                        param.grad.zero_()

            if all_param_grad_none:
                self._delay_grad_buffer.zero_()
    
    def _lazy_init(self):
        # Initialization for DDP that occurs after construction, but lazily
        # before the first forward pass.
        self._lazy_init_ran = True
    
    def _stepwise_operation(self, iteration):
        if len(self.dataloader)/100 > 100:
            if iteration % 100 == 0:
                self.merge_model_fn()

    def _perform_synchronization(self):
        if (self.current_steps % self.steps_per_allreduce == 0) or ((self.steps_per_allreduce < self.steps_per_epoch) and (self.current_steps % self.steps_per_epoch == 0)):
            self.merge_model_fn()
            self.merge_optimizer_state()
            self.current_steps = 0
        elif self.steps_per_model_shift > 0 and self.current_steps % self.steps_per_model_shift == 0:
            self.shift_model_fn()
        if self.current_steps % self.steps_per_epoch == 0:
            self.current_epoch += 1
    
    def _shift_model_pairs(self):
        with torch.no_grad():
            pow2 = next_power_of_2(self.world_size)
            max_degree = log2(pow2)
            current_degree = self.exchdeg % max_degree
            use_rank = floor(self.global_rank / (2**current_degree))
            if use_rank % 2 == 0:
                source = int((self.global_rank + (2**current_degree)) % pow2)
            else:
                source = int((self.global_rank - (2**current_degree) + pow2) % pow2)
            if source >= self.world_size:
                source = -1
            dest = source
            #print(self.global_rank, source, dest)
            self.reducer.exchange_weights(source,dest,self.world_size,self.global_rank)
            self.exchdeg += 1
            
    def _shift_model_ring(self):
        with torch.no_grad():
            source = (self.global_rank - 1 + self.world_size) % self.world_size
            dest = (self.global_rank + 1) % self.world_size
            self.reducer.exchange_weights(source,dest,self.world_size,self.global_rank)
    
    def reverse_higher_portion_inplace(self, nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        
        # In-place reversal using two pointers
        i, j = left, len(nums) - 1
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
        return nums
    
    def _permute_layers(self, root=0):
        encoder = self.module.bert.encoder
        layers = encoder.layer
        my_flats = get_layer_flats(layers, self.local_rank)

        if self.global_rank == root:
            ref_flats = my_flats
        else:
            ref_flats = torch.zeros_like(my_flats)
        
        dist.broadcast(ref_flats, src=root)

        if self.global_rank != root:
            cost = torch.cdist(ref_flats, my_flats, p=2).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            permuted_layers = torch.nn.ModuleList([layers[c] for c in col_ind])
            encoder.layer = permuted_layers
            print(f"Rank {self.global_rank} - Applied permutation: {col_ind}")
        dist.barrier()

    def _generate_model_diffs(self):
        with torch.no_grad():
            #Generate Pairs to calculate diffs
            myrecvpaircount = (self.world_size - 1) / 2
            myrecvpaircount = floor(myrecvpaircount) if self.global_rank >= self.world_size/2 else ceil(myrecvpaircount)
            mysendpaircount = self.world_size - myrecvpaircount - 1
            myrecvpairs = []
            mysendpairs = []
            for i in range(myrecvpaircount):
                myrecvpairs.append((self.global_rank+i+1) % self.world_size)
            for i in range(mysendpaircount):
                mysendpairs.append((self.global_rank-i-1+self.world_size) % self.world_size)
            myrecvpairs = sorted(myrecvpairs)
            mysendpairs = sorted(mysendpairs)
            self.reverse_higher_portion_inplace(myrecvpairs, self.world_size/2)

            #Generate Diffs for Pairs
            diffs = self.reducer.generate_diffs(mysendpairs, myrecvpairs, self.world_size, self.global_rank)

            #Merge the Diffs
            diffs = torch.tensor(diffs, dtype=torch.float32, device=self.global_rank)
            dist.all_reduce(diffs, op=ReduceOp.SUM)
            print("Global Diffs: ",diffs)


    def average_optimizer_state(self):
        self.reducer.average_optimizer_state()

    def merge_optimizer_state(self):
        self.reducer.merge_optimizer_state()

    def bcast_optimizer_state(self, root):
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                if 'momentum_buffer' in state:
                    dist.broadcast(state['momentum_buffer'], src=root)
                if 'exp_avg' in state:
                    dist.broadcast(state['exp_avg'], src=root)
                    dist.broadcast(state['exp_avg_sq'], src=root)

    def _merge_models(self):
        with torch.no_grad():
            #self.reducer.bcast_weights(0)
            #self.reducer.allreduce_weights()
            self.reducer.perform_ties_merge(self.current_epoch, self.epochs)
    
    def _pre_forward(self, *inputs, **kwargs):
        if self._use_python_reducer:
            return input, kwargs
        
        if not self._lazy_init_ran and not torch.compiler.is_compiling():
            self._lazy_init()
        
        if self._delay_all_reduce_all_params:
            return inputs, kwargs
        
        return inputs, kwargs
    
    def _post_forward(self, output):
        if self._use_python_reducer:
            return output
        
        if self._delay_all_reduce_all_params:
            self._clear_grad_buffer()
            return output
        
        if self.training:
            self.current_steps += 1

        self._clear_grad_buffer()
        return output
    
    def forward(self, *inputs, **kwargs):
        with torch.profiler.record_function("DistributedDataAsynchronousParallel"):
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)
            output = (
                self.module.forward(*inputs, **kwargs)
                if self._delay_all_reduce_all_params
                else self._run_ddap_forward(*inputs, **kwargs)
            )
            return self._post_forward(output)
    
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def to_kwargs(self, inputs, kwargs, device_id):
        # Kept for BC
        return _to_kwargs(
            inputs,
            kwargs,
            torch.device(self.device_type, device_id),
            self.use_side_stream_for_tensor_copies,
        )

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim = self.dim)

    def train(self, mode=True):
        super().train(mode)
        return self
    
    def join(
        self,
        divide_by_initial_world_size: bool = True,
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ):
        return Join(
            [self],
            enable,
            throw_on_early_termination,
            divide_by_initial_world_size=divide_by_initial_world_size,
        )
    
    def join_hook(
        self,
        **kwargs,
    ):
        divide_by_initial_world_size = kwargs.get("divide_by_initial_world_size", True)
        return _DDAPJoinHook(
            self, divide_by_initial_world_size=divide_by_initial_world_size
        )
    
    @property
    def join_device(self):
        return self.device

    @property
    def join_process_group(self):
        return self.process_group
