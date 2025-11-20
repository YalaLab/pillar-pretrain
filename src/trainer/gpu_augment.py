"""GPU-based 3D rotation augmentation using Kornia."""

import torch
import kornia.augmentation as K
from typing import Union, Tuple, List


class RandomRotation3D(torch.nn.Module):
    """Apply random 3D rotation using Kornia's RandomRotation3D."""
    
    def __init__(self, degrees: List[float] = [0, 0, 10], resample: str = "bilinear", 
                 same_on_batch: bool = False, p: float = 0.5):
        super().__init__()
        # Use Kornia's RandomRotation3D
        self.transform = K.RandomRotation3D(
            degrees=degrees,  # [x, y, z] rotation ranges
            resample=resample,
            same_on_batch=same_on_batch,
            p=p
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation to input tensor."""
        return self.transform(x)


def apply_3d_rotation_gpu(
    images: torch.Tensor,
    degrees: float = 10.0,
    p: float = 0.5,
    training: bool = True
) -> torch.Tensor:
    """Apply random 3D rotation to medical images on GPU using Kornia.
    
    Args:
        images: Tensor of shape (B, C, D, H, W) or (B, D, H, W)
        degrees: Maximum rotation degrees (applied to Z-axis only by default)
        p: Probability of applying rotation
        training: Whether in training mode (no rotation if False)
        
    Returns:
        Rotated images tensor
    """
    if not training:
        return images
        
    # Handle both 4D and 5D inputs
    need_squeeze = False
    if images.dim() == 4:
        images = images.unsqueeze(1)  # Add channel dimension
        need_squeeze = True
    
    # Create rotation transform
    # Default: only rotate around Z-axis (axial rotation)
    degrees_list = [0, 0, degrees]  # [x, y, z] rotation ranges
    rotation = RandomRotation3D(degrees=degrees_list, resample="bilinear", same_on_batch=False, p=p)
    rotation = rotation.to(images.device)
    rotation.train(training)
    
    # Apply rotation
    rotated = rotation(images)
    
    # Remove channel dimension if input was 4D
    if need_squeeze:
        rotated = rotated.squeeze(1)
        
    return rotated
