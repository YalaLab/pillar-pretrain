"""
Distributed training utilities for PyTorch.

This module provides utilities for setting up and managing distributed training across multiple GPUs.
It supports both single-node multi-GPU training and multi-node training via SLURM or torchrun.

Key features:
- Automatic device management for distributed training
- Support for both SLURM and torchrun environments
- Utilities for rank-based operations and device selection
- Object broadcasting and gathering across processes

Example usage:
    ```python
    args = argparse.Namespace()
    device = init_distributed_device(args)
    if is_master(args):
        # Do something only on the master process
        pass
    ```
"""

import os
import warnings
from typing import Optional

import torch
import torch.distributed as dist


def is_global_master(args):
    """Check if the current process is the global master (rank 0).
    
    Args:
        args: Arguments namespace containing rank information
        
    Returns:
        bool: True if this is the global master process
    """
    return args.rank == 0


def is_local_master(args):
    """Check if the current process is the local master (local_rank 0).
    
    Args:
        args: Arguments namespace containing local_rank information
        
    Returns:
        bool: True if this is the local master process
    """
    return args.local_rank == 0


def is_master(args, local=False):
    """Check if the current process is a master process.
    
    Args:
        args: Arguments namespace containing rank information
        local: If True, check for local master, otherwise check for global master
        
    Returns:
        bool: True if this is the specified master process
    """
    return is_local_master(args) if local else is_global_master(args)


def is_device_available(device):
    """Check if the specified device is available and known.
    
    Args:
        device: Device string (e.g., 'cuda' or 'cpu')
        
    Returns:
        tuple: (is_available, is_known) indicating device status
    """
    device_type = torch.device(device).type
    is_avail = False
    is_known = False
    if device_type == 'cuda':
        is_avail = torch.cuda.is_available()
        is_known = True
    elif device_type == 'cpu':
        is_avail = True
        is_known = True
    return is_avail, is_known


def set_device(device):
    """Set the current CUDA device.
    
    Args:
        device: Device string (e.g., 'cuda:0')
    """
    if device.startswith('cuda:'):
        torch.cuda.set_device(device)


def is_using_distributed():
    """Check if distributed training is enabled based on environment variables.
    
    Returns:
        bool: True if distributed training is enabled
    """
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    """Get distributed training information from environment variables.
    
    Returns:
        tuple: (local_rank, global_rank, world_size)
    """
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    """Initialize distributed training and set up the device.
    
    This function sets up distributed training and configures the appropriate device
    for the current process. It updates the args namespace with distributed training
    information.
    
    Args:
        args: Arguments namespace to be updated with distributed training info
        
    Returns:
        torch.device: The device to use for this process
    """
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    result = init_distributed_device_so(
        device=getattr(args, 'device', 'cuda'),
        dist_backend=getattr(args, 'dist_backend', None),
        dist_url=getattr(args, 'dist_url', None),
        horovod=getattr(args, 'horovod', False),
        no_set_device_rank=getattr(args, 'no_set_device_rank', False),
    )
    args.device = result['device']
    args.world_size = result['world_size']
    args.rank = result['global_rank']
    args.local_rank = result['local_rank']
    args.distributed = result['distributed']
    device = torch.device(args.device)
    return device


def init_distributed_device_so(
        device: str = 'cuda',
        dist_backend: Optional[str] = None,
        dist_url: Optional[str] = None,
        horovod: bool = False,
        no_set_device_rank: bool = False,
):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    device_type, *device_idx = device.split(':', maxsplit=1)
    is_avail, is_known = is_device_available(device_type)
    if not is_known:
        warnings.warn(f"Device {device} was not known and checked for availability, trying anyways.")
    elif not is_avail:
        warnings.warn(f"Device {device} was not available, falling back to CPU.")
        device_type = device = 'cpu'

    if is_using_distributed():
        if dist_backend is None:
            dist_backends = {
                "cuda": "nccl",
            }
            dist_backend = dist_backends.get(device_type, 'gloo')

        dist_url = dist_url or 'env://'

        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            local_rank, global_rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(global_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
            )
        else:
            import datetime 
            # DDP via torchrun, torch.distributed.launch
            local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                timeout=datetime.timedelta(seconds=1800),
            )
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        distributed = True

    if distributed and not no_set_device_rank and device_type not in ('cpu'):
        # Ignore manually specified device index in distributed mode and
        # override with resolved local rank, fewer headaches in most setups.
        if device_idx:
            warnings.warn(f'device index {device_idx[0]} removed from specified ({device}).')
        device = f'{device_type}:{local_rank}'
        set_device(device)

    return dict(
        device=device,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )


def broadcast_object(args, obj, src=0):
    """Broadcast a Python object from source rank to all other ranks.
    
    Args:
        args: Arguments namespace containing rank information
        obj: Object to broadcast (must be pickle-able)
        src: Source rank (default: 0)
        
    Returns:
        The broadcasted object
    """
    if args.rank == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_object(args, obj, dst=0):
    """Gather a Python object from all ranks.
    
    Args:
        args: Arguments namespace containing world_size information
        obj: Object to gather (must be pickle-able)
        dst: Destination rank (default: 0)
        
    Returns:
        list: List of gathered objects from all ranks
    """
    objects = [None for _ in range(args.world_size)]
    dist.all_gather_object(objects, obj)
    return objects
