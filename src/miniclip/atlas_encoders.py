
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Optional, Union, Tuple
from timm.models.layers import DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp

logger = logging.getLogger(__name__)

class ConvBlock3D(nn.Module):
    """
    3D Conv block adapted from: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention"
    """

    def __init__(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3):
        """
        Args:
            drop_path: drop path probability.
            layer_scale: layer scale coefficient.
            kernel_size: kernel size for 3D convolutions.
        """
        super().__init__()
        logger.debug(f"Initializing ConvBlock3D with dim={dim}, drop_path={drop_path}, kernel_size={kernel_size}")
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        logger.debug(f"ConvBlock3D input shape: {x.shape}")
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1, 1)  # Added extra dimension for 3D
        x = input + self.drop_path(x)
        logger.debug(f"ConvBlock3D output shape: {x.shape}")
        return x




class LayerNorm3d(nn.LayerNorm):
    """ LayerNorm for channels of '3D' spatial NCDHW tensors """
    _fast_norm: torch.jit.Final[bool]

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        logger.debug(f"Initialized LayerNorm3d with num_channels={num_channels}")
        self._fast_norm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Change permutation to handle 3D data: NCDHW -> NDHWC
        logger.debug(f"LayerNorm3d input shape: {x.shape}")
        x = x.permute(0, 2, 3, 4, 1)
        
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        # Return to original format: NDHWC -> NCDHW
        x = x.permute(0, 4, 1, 2, 3)
        logger.debug(f"LayerNorm3d output shape: {x.shape}")
        return x



class Downsample3D(nn.Module):
    """
    3D Down-sampling block adapted from: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention"
    """

    def __init__(
        self,
        dim,
        keep_dim=False,
        kernel_size=(3, 3, 3),
        stride=(3, 2, 2),
        padding=(0, 1, 1)
    ):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.norm = LayerNorm3d(dim)  # Assuming you have a 3D LayerNorm implementation
        self.reduction = nn.Sequential(
            nn.Conv3d(dim, dim_out,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x



class MultiConvEmbed3D(nn.Module):
    def __init__(
        self,
        vol_size=224,  # Changed from img_size to vol_size
        patch_size=16,
        in_chans=1,    # Changed default from 3 to 1 for typical 3D medical data
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        
        # Base embedding class with flexible downsampling
        # Default total downsampling: 8x8x4
        # Strategy:
        # - conv1: 1x2x2 downsampling
        # - conv2: 2x2x2 downsampling
        # - conv3: 1x1x1 downsampling (no downsampling)
        # - ds1: 2x2x2 downsampling (default Downsample3D)
        # Total: 4x8x8
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=2, ## use 1,2,2 for 3,8,8
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                # for 8x8x3
                stride=1,
                ## for 16x16x6
                # stride=(2, 1, 1),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )

        # 3 conv-blocks
        self.conv1 = nn.Sequential(
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
        )
        self.ds1 = Downsample3D(
            embed_dim,
            keep_dim=False)
        
        # # 3 conv-blocks
        # self.conv2 = nn.Sequential(
        #     ConvBlock3D(2 * embed_dim),
        #     ConvBlock3D(2 * embed_dim),
        #     ConvBlock3D(2 * embed_dim),
        # )
        # self.ds2 = Downsample3D(2 * embed_dim)

        self.flatten = flatten

    def forward(self, x):
        # Expected input: (B, C, D, H, W)
        x = self.proj(x)
        
        # Downsampling and feature extraction
        x = self.conv_down(x)  # 4x4x2 downsampling
        x = self.conv1(x)      # Feature extraction, no downsampling
        x = self.ds1(x)        # Additional 2x2x2 downsampling
        
        if self.flatten:
            x = rearrange(x, 'b c d h w -> b (d h w) c')  # Modified for 3D data
        return x

class BMRMultiConvEmbed3D(MultiConvEmbed3D):
    def __init__(
            self,
            vol_size=224,  # Changed from img_size to vol_size
            patch_size=16,
            in_chans=1,    # Changed default from 3 to 1 for typical 3D medical data
            in_dim=64,
            embed_dim=192,
            flatten=False,
            bias=False,
        ):
        super().__init__(embed_dim=embed_dim)
        
        # Breast MR specific downsampling
        # Total downsampling: 6x24x24 (for 8x8x3 output) or 3x12x12 (for 16x16x6 output)
        # Strategy:
        # - conv1: 2x2x2 downsampling
        # - conv2: 1x2x2 downsampling  
        # - conv3: 1x1x1 downsampling (no downsampling)
        # - ds1: 3x3x3 downsampling
        # Total: 6x24x24
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(2, 2, 2), # tweak either this to (2,2,2)
                padding=1,
                bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=(1,2,2), # or this 
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                # for 8x8x3
                stride=1,
                ## for 16x16x6
                # stride=(2, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )
        self.ds1 = Downsample3D(
            embed_dim,
            stride=3,
            padding=0,
            keep_dim=False)


class BrainCTMultiConvEmbed3D(MultiConvEmbed3D):
    def __init__(
            self,
            vol_size=224,  # Changed from img_size to vol_size
            patch_size=16,
            in_chans=11,    # 11 channels for brain CT
            in_dim=64,
            embed_dim=192,
            flatten=False,
            bias=False,
        ):
        super().__init__(in_chans=in_chans, embed_dim=embed_dim)
        
        # Brain CT specific downsampling
        # Uses parent class conv_down (4x8x8) + custom ds1
        # Strategy:
        # - Inherits conv_down from parent: 4x8x8 downsampling
        # - ds1: 2x2x2 downsampling
        # Total: 8x16x16
        self.ds1 = Downsample3D(
            embed_dim,
            stride=2,
            padding=1,
            keep_dim=False)


class ProstateMRMultiConvEmbed3D(nn.Module):
    def __init__(
            self,
            vol_size=768,  # Expected input volume size
            patch_size=24,
            in_chans=1,    # Number of input channels for prostate MR
            in_dim=64,
            embed_dim=192,
            flatten=False,
            bias=False,
        ):
        super().__init__()
        
        # For 768x768x32 input with 24x24x1 downsampling -> 32x32x32 output
        # Total downsampling needed: 1x24x24
        # Strategy: 
        # - conv1: 1x3x3 downsampling
        # - conv2: 1x2x2 downsampling  
        # - conv3: 1x2x2 downsampling
        # - conv4: 1x2x2 downsampling
        # Total: 1x24x24
        
        self.proj = nn.Identity()
        
        # Initial conv block with 1x3x3 downsampling
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(1, 3, 3),  # 1x3x3 downsampling
                padding=1,
                bias=bias),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=(1, 2, 2),  # 1x2x2 downsampling
                padding=1,
                bias=bias),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=(1, 2, 2),  # 1x2x2 downsampling
                padding=1,
                bias=bias),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(inplace=True),
        )
        
        # Conv blocks for feature extraction (no downsampling)
        self.conv1 = nn.Sequential(
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
        )
        
        # Final downsampling to achieve 1x24x24 total
        self.ds1 = Downsample3D(
            embed_dim,
            stride=(1, 2, 2),  # 1x2x2 downsampling
            padding=(1, 1, 1),  # Changed from (0,1,1) to maintain depth=32
            keep_dim=False
        )
        
        self.flatten = flatten

    def forward(self, x):
        # Expected input: (B, 1, 32, 768, 768)
        x = self.proj(x)
        
        # After conv_down: (B, embed_dim, 32, 64, 64)
        x = self.conv_down(x)
        
        # Feature extraction without downsampling
        x = self.conv1(x)
        
        # Final downsampling: (B, embed_dim, 32, 32, 32)
        x = self.ds1(x)
        
        if self.flatten:
            x = rearrange(x, 'b c d h w -> b (d h w) c')
            
        return x


class AbdomenCTEmbed3D(nn.Module):
    def __init__(
        self,
        vol_size=224,
        patch_size=16,
        in_chans=1,
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        
        # Abdomen CT specific downsampling for isotropic volumes
        # Total downsampling: 8x8x8
        # Strategy:
        # - conv1: 2x2x2 downsampling
        # - conv2: 2x2x2 downsampling  
        # - conv3: 1x1x1 downsampling (no downsampling)
        # - ds1: 2x2x2 downsampling
        # Total: 8x8x8
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )

        # 3 conv-blocks
        self.conv1 = nn.Sequential(
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
        )
        self.ds1 = Downsample3D(
            embed_dim,
            keep_dim=False,
            stride=(2, 2, 2),
            padding=(1, 1, 1)
        )
        
        self.flatten = flatten

    def forward(self, x):
        # Expected input: (B, 1, D, H, W) - isotropic CT volume
        x = self.proj(x)
        
        # After conv_down: 4x4x4 downsampling
        x = self.conv_down(x)
        
        # Feature extraction without downsampling
        x = self.conv1(x)
        
        # Final downsampling: total 8x8x8
        x = self.ds1(x)
        
        if self.flatten:
            x = rearrange(x, 'b c d h w -> b (d h w) c')
        return x



class ChestCTEmbed3D(nn.Module):
    ## abdomen ct version 6x6x6
    def __init__(
        self,
        vol_size=224,
        patch_size=16,
        in_chans=1,
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        
        # Chest CT specific downsampling for isotropic volumes
        # Total downsampling: 6x6x6
        # Strategy:
        # - conv_down: 3x3x3 downsampling
        # - conv1: 1x1x1 downsampling (no downsampling)
        # - ds1: 2x2x2 downsampling
        # Total: 6x6x6
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(3, 3, 3),
                padding=0,
                bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )

        # 3 conv-blocks
        self.conv1 = nn.Sequential(
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
        )
        self.ds1 = Downsample3D(
            embed_dim,
            keep_dim=False,
            stride=(2, 2, 2),
            padding=(1, 1, 1)
        )
        
        self.flatten = flatten

    def forward(self, x):
        # Expected input: (B, 1, D, H, W) - isotropic CT volume
        x = self.proj(x)
        
        # After conv_down: 3x3x3 downsampling
        x = self.conv_down(x)
        
        # Feature extraction without downsampling
        x = self.conv1(x)
        
        # Final downsampling: total 6x6x6
        x = self.ds1(x)
        
        if self.flatten:
            x = rearrange(x, 'b c d h w -> b (d h w) c')
        return x


class ChestCT884Embed3D(nn.Module):
    ## chest ct version 8x8x4
    def __init__(
        self,
        vol_size=224,
        patch_size=16,
        in_chans=1,
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        
        # Chest CT specific downsampling for isotropic volumes
        # Total downsampling: 6x6x6
        # Strategy:
        # - conv_down: 3x3x3 downsampling
        # - conv1: 1x1x1 downsampling (no downsampling)
        # - ds1: 2x2x2 downsampling
        # Total: 6x6x6
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )


        # 3 conv-blocks
        self.conv1 = nn.Sequential(
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
        )
        self.ds1 = Downsample3D(
            embed_dim,
            keep_dim=False,
            stride=(2, 2, 1),
            padding=(1, 1, 1)
        )
        
        self.flatten = flatten

    def forward(self, x):
        # Expected input: (B, 1, D, H, W) - isotropic CT volume
        x = self.proj(x)
        
        # After conv_down: 3x3x3 downsampling
        x = self.conv_down(x)
        
        # Feature extraction without downsampling
        x = self.conv1(x)
        
        # Final downsampling: total 6x6x6
        x = self.ds1(x)
        
        if self.flatten:
            x = rearrange(x, 'b c d h w -> b (d h w) c')
        return x



class ConvCXREmbed(nn.Module):
    def __init__(
        self,
        vol_size=224,  # Changed from img_size to vol_size
        patch_size=16,
        in_chans=1,    # Changed default from 3 to 1 for typical 3D medical data
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        
        # CXR (Chest X-ray) specific downsampling
        # Designed for 2D images treated as 3D with depth=1
        # Total downsampling: 1x16x16
        # Strategy:
        # - conv1: 1x2x2 downsampling
        # - conv2: 1x2x2 downsampling  
        # - conv3: 1x2x2 downsampling
        # - ds1: 1x2x2 downsampling
        # Total: 1x16x16
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                # for 1x16x16
                stride=(1, 2, 2),
                ## for 16x16x6
                # stride=(2, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )

        # 3 conv-blocks
        self.conv1 = nn.Sequential(
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
        )
        self.ds1 = Downsample3D(
            embed_dim,
            keep_dim=False,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1
        )


        self.flatten = flatten

    def forward(self, x):
        # Expected input: (B, 1, 1, H, W) - 2D CXR as 3D volume
        x = self.proj(x)
        
        # After conv_down: 1x8x8 downsampling
        x = self.conv_down(x)
        
        # Feature extraction without downsampling
        x = self.conv1(x)
        
        # Final downsampling: total 1x16x16
        x = self.ds1(x)
        
        if self.flatten:
            x = rearrange(x, 'b c d h w -> b (d h w) c')  # Modified for 3D data
        return x




class MultiViewConvCXR(nn.Module):
    """
    Unlike the ConvCXREmbed, for each sample expect two views of CXR.
    Use a learnable view-specific positional embedding to encode the two views.
    """

    def __init__(self,
        vol_size=224,  # Changed from img_size to vol_size
        patch_size=16,
        in_chans=1,    # Changed default from 3 to 1 for typical 3D medical data
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
        merge_views=True,  # If True, merge the two views into one
    ):
        super().__init__()
        
        # Multi-view CXR embedding (frontal + lateral views)
        # Same downsampling as single-view CXR: 1x16x16
        # Strategy:
        # - conv1: 1x2x2 downsampling
        # - conv2: 1x2x2 downsampling  
        # - conv3: 1x2x2 downsampling
        # - ds1: 1x2x2 downsampling
        # Total: 1x16x16 per view
        # Additional: view-specific positional embeddings added
        self.merge_views = merge_views
        
        self.pos_embed = nn.Parameter(torch.randn(1, 2, 2*embed_dim))
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                in_dim,
                embed_dim,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                # for 1x16x16
                stride=(1, 2, 2),
                ## for 16x16x6
                # stride=(2, 2, 2),
                padding=1,
                bias=False),
            nn.BatchNorm3d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )

        # 3 conv-blocks
        self.conv1 = nn.Sequential(
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
            ConvBlock3D(embed_dim),
        )
        self.ds1 = Downsample3D(
            embed_dim,
            keep_dim=False,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1
        )


        self.flatten = flatten

    def forward(self, x):
        # Expected input: 
        # - 6D: (B, V, C, D, H, W) where V=2 views
        # - 5D: (B, C, V, H, W) for vlm-eval

        if len(x.shape) == 6: ## batch x views x channels x depth x height x width
            # If input is 6D, assume it has two views
            num_views = x.shape[1]
            x = rearrange(x, 'b v c d h w -> (b v) c d h w')
        elif len(x.shape) == 5:
            ## for vlm-eval
            num_views = x.shape[2]
            x = rearrange(x, 'b c v h w -> (b v) c 1 h w')

        # Embed the inputs using shared patch embed
        x = self.proj(x)
        
        # Downsampling: 1x16x16 per view
        x = self.conv_down(x)  # 1x8x8
        x = self.conv1(x)      # Feature extraction
        x = self.ds1(x)        # Final 1x2x2 -> total 1x16x16
        # revert to multi-view shape
        x = rearrange(x, '(b v) c d h w -> b v c d h w', v=num_views)

        ## make positional embedding compatible with 3D data
        if self.pos_embed is not None:

            # ensure shape of pos-embed is BVCDHW
            if self.pos_embed.dim() == 3:
                pos_embed = self.pos_embed.unsqueeze(3).unsqueeze(4).unsqueeze(5)  # Add dimensions for depth, height, width

            x = x + pos_embed

        ## if merge views, then put the two views side by side.
        # i.e. the output should be going from BVCDHW to BCDh(VW)
        if self.merge_views:
            x = rearrange(x, 'b v c d h w -> b c d h (v w)')

        if self.flatten:
            x = rearrange(x, 'b c d h w -> b (d h w) c')  # Modified for 3D data
        return x
        