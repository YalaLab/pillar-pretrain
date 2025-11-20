import os
import torch
import torch.nn as nn


def maybe_copy_codebase(args):
    if args.copy_codebase:
        from shutil import copytree, ignore_patterns

        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(
            current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
        )
        print("Done copying code.")
        return 1
    return 0


def get_weight_norm(model, norm="2"):
    """Calculate the weight norm for each parameter in the model."""
    return {
        f"weight_norm/{name}": torch.linalg.norm(param) / torch.numel(param)
        for name, param in model.named_parameters()
    }


def compute_weight_norms(model, norm_type='l2'):
    """
    Compute weight norms for all parameters in the model.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm to compute ('l2', 'l1', 'inf')
    
    Returns:
        dict: Dictionary containing norm statistics with fine-grained tracking
    """
    weight_norms = {}
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if norm_type == 'l2' or norm_type == '2':
                norm = param.data.norm(2).item()
                # Also compute normalized norm (per element)
                weight_norms[f"weight_norm/{name}"] = param.data.norm(2).item() / param.numel()
            elif norm_type == 'l1' or norm_type == '1':
                norm = param.data.norm(1).item()
                weight_norms[f"weight_norm/{name}"] = param.data.norm(1).item() / param.numel()
            elif norm_type == 'inf':
                norm = param.data.norm(float('inf')).item()
                weight_norms[f"weight_norm/{name}"] = norm
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
            
            weight_norms[f"weight_norm_raw/{name}"] = norm
            total_norm += norm ** 2
            param_count += 1
    
    if norm_type in ['l2', '2']:
        total_norm = total_norm ** 0.5
    
    weight_norms['total_weight_norm'] = total_norm
    weight_norms['param_count'] = param_count
    
    return weight_norms


def get_grad_norm(model, check_nan=False, log_weight_norm=False, norm=2):
    """Calculate gradient norms and optionally weight norms for model parameters."""
    if norm != 2:
        raise ValueError("Only l2-norm supported")

    grad_norm_dict = {}
    is_nan_dict = {}
    
    # Get parameters with gradients
    params_with_grad = [(name, param) for name, param in model.named_parameters() if param.grad is not None]
    
    if params_with_grad:
        # Calculate gradient norms
        grad_norm_dict = {
            f"grad_norm/{name}": param.grad.data.norm(norm).item()
            for name, param in params_with_grad
        }
        
        # Check for NaN gradients if required
        if check_nan:
            is_nan_dict = {
                name: param.grad for name, param in params_with_grad
                if torch.isnan(param.grad).any()
            }
        
        # Calculate total gradient norm
        grad_norms_tensor = torch.tensor([v for v in grad_norm_dict.values()])
        grad_norm_dict["total_grad_norm"] = grad_norms_tensor.norm(norm).item()
    
    # Add weight norms if required
    if log_weight_norm:
        grad_norm_dict.update(get_weight_norm(model))
    
    return grad_norm_dict, is_nan_dict


def compute_gradient_norms(model, norm_type='l2', check_nan=False):
    """
    Compute gradient norms for all parameters in the model.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm to compute ('l2', 'l1', 'inf')
        check_nan: Whether to check for NaN gradients
    
    Returns:
        dict: Dictionary containing norm statistics with fine-grained tracking
    """
    grad_norms = {}
    total_norm = 0.0
    param_count = 0
    nan_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if norm_type == 'l2' or norm_type == '2':
                norm = param.grad.data.norm(2).item()
                grad_norms[f"grad_norm/{name}"] = norm
            elif norm_type == 'l1' or norm_type == '1':
                norm = param.grad.data.norm(1).item()
                grad_norms[f"grad_norm/{name}"] = norm
            elif norm_type == 'inf':
                norm = param.grad.data.norm(float('inf')).item()
                grad_norms[f"grad_norm/{name}"] = norm
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
            
            grad_norms[f"grad_norm_raw/{name}"] = norm
            total_norm += norm ** 2
            param_count += 1
            
            # Check for NaN
            if check_nan and torch.isnan(param.grad).any():
                nan_params.append(name)
    
    if norm_type in ['l2', '2']:
        total_norm = total_norm ** 0.5
    
    grad_norms['total_grad_norm'] = total_norm
    grad_norms['param_count'] = param_count
    
    if check_nan:
        grad_norms['nan_params'] = nan_params
        grad_norms['has_nan'] = len(nan_params) > 0
    
    return grad_norms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



