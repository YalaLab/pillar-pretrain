"""
yolo run 1

- prepare tokens for all scales
- prepare QKV for all scales
- all to all in parallel for all scales
- update locals : by running x-attn from locals to globals
- update globals : run x-atnn pairwise to update globals from prev stage locals
- MLP

update scale3
scale3 = scale3 + xattn(scale3, scale3) + xattn_one2one(scale3, scale2)
scale3 = scale3 + mlp(scale3, scale3)

update scale2
scale2 = scale2 + xattn_all2all(scale2, [scale2, scale3]) + xattn_one2one(scale2, [scale1])
scale2 = scale2 + mlp(scale2)

update scale1
scale1 = scale1 + xattn_all2all(scale1, [scale1, scale2, scale3])
scale1 = scale1 + mlp(scale1)


0. get all scale tokens from max pool # at most 2N tokens

## optimized block
1. get all scale QKV  ## 3*2N
2. all2all communication + one2one communication as needed ## needs repeats
3. run x-attn for each scale
4. run MLP for each scale


scale3 = scale3 + xattn(scale3, scale3) + xattn_one2one(scale3, scale2)  ##
scale2 = scale2 + xattn_all2all(scale2, [scale2, scale3]) + xattn_one2one(scale2, [scale1])
scale1 = scale1 + xattn_all2all(scale1, [scale1, scale2, scale3])  ## NKlogN/logK

scale2 = scale2 + mlp(scale2)
scale3 = scale3 + mlp(scale3)
scale1 = scale1 + mlp(scale1)

####
readout : avgpool each scale + concat + linear
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Union, Tuple
from timm.models.layers import DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp
from .atlas_encoders import MultiConvEmbed3D, ConvCXREmbed, MultiViewConvCXR


logger = logging.getLogger(__name__)




class AbstractModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class ConvBlock(nn.Module):
    """
    Conv block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3):
        """
        Args:
            drop_path: drop path.
            layer_scale: layer scale coefficient.
            kernel_size: kernel size.
        """
        super().__init__()
        logger.debug(f"Initializing ConvBlock with dim={dim}, drop_path={drop_path}, kernel_size={kernel_size}")
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        logger.debug(f"ConvBlock input shape: {x.shape}")
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        logger.debug(f"ConvBlock output shape: {x.shape}")
        return x


class Downsample(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(
        self,
        dim,
        keep_dim=False,
    ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        logger.debug(f"Initializing Downsample with dim={dim}, keep_dim={keep_dim}")
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.norm = LayerNorm2d(dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        logger.debug(f"Downsample input shape: {x.shape}")
        x = self.norm(x)
        x = self.reduction(x)
        logger.debug(f"Downsample output shape: {x.shape}")
        return x


class MultiConvEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        in_dim=64,
        embed_dim=96,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        logger.debug(f"Initializing MultiConvEmbed with in_chans={in_chans}, in_dim={in_dim}, embed_dim={embed_dim}")
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, embed_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim, eps=1e-4),
            nn.ReLU(),
        )

        # 3 conv-blocks per fastervit patchify
        self.conv1 = nn.Sequential(
            ConvBlock(embed_dim),
            ConvBlock(embed_dim),
            ConvBlock(embed_dim),
        )
        # ConvBlock(embed_dim)
        self.ds1 = Downsample(embed_dim)
        # 3 conv-blocks per fastervit patchify
        self.conv2 = nn.Sequential(
            ConvBlock(2 * embed_dim),
            ConvBlock(2 * embed_dim),
            ConvBlock(2 * embed_dim),
        )
        self.ds2 = Downsample(2 * embed_dim)

        self.flatten = flatten

    def forward(self, x):
        # breakpoint()
        logger.debug(f"MultiConvEmbed input shape: {x.shape}")
        x = x.squeeze(2)
        logger.debug(f"  After squeeze shape: {x.shape}")
        x = self.proj(x)
        x = self.conv_down(x)
        logger.debug(f"  After conv_down shape: {x.shape}")
        x = self.conv1(x)
        logger.debug(f"  After conv1 shape: {x.shape}")
        x = self.ds1(x)
        logger.debug(f"  After ds1 shape: {x.shape}")
        x = self.conv2(x)
        logger.debug(f"  After conv2 shape: {x.shape}")
        x = self.ds2(x)
        logger.debug(f"  After ds2 shape: {x.shape}")
        if self.flatten:
            x = rearrange(x, "b c h w -> b (h w) c")
        # else:
        #     x = rearrange(x, "b c h w -> b c 1 h w")
        logger.debug(f"MultiConvEmbed final output shape: {x.shape}")
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        logger.debug(f"Initializing PatchEmbed with in_chans={in_chans}, in_dim={in_dim}, dim={dim}")
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(),
        )

        # self.conv_down = nn.Conv2d(in_chans, dim, 4, 4, 0, bias=False)

    def forward(self, x):
        # breakpoint()
        logger.debug(f"PatchEmbed input shape: {x.shape}")
        x = self.proj(x)
        x = self.conv_down(x)
        logger.debug(f"PatchEmbed output shape: {x.shape}")
        return x


class PosEmbMLPSwinv1D_v0(nn.Module):
    def __init__(self, dim, rank=2, seq_length=4, conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(
                nn.Linear(self.rank, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, dim, bias=False),
            )
        else:
            self.cpb_mlp = nn.Sequential(
                nn.Conv1d(self.rank, 512, 1, bias=True),
                nn.ReLU(),
                nn.Conv1d(512, dim, 1, bias=False),
            )
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def forward(self, input_tensor):
        # breakpoint()
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(
                    0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
                )
                relative_coords_h -= seq_length // 2
                relative_coords_h /= seq_length // 2
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(
                    relative_coords_table.unsqueeze(0).unsqueeze(2)
                )
                self.relative_bias = self.pos_emb
            else:
                seq_length = int(seq_length**0.5)
                relative_coords_h = torch.arange(
                    0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
                )
                relative_coords_w = torch.arange(
                    0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype
                )
                relative_coords_table = (
                    torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
                    .contiguous()
                    .unsqueeze(0)
                )
                relative_coords_table -= seq_length // 2
                relative_coords_table /= seq_length // 2
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(
                        relative_coords_table.flatten(2).transpose(1, 2)
                    )
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
        input_tensor = input_tensor + self.pos_emb
        return input_tensor




class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self, dim, rank=2, seq_length=4, conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            logger.debug(f"Initializing PosEmbMLPSwinv1D (Linear) dim={dim}, rank={rank}, seq_length={seq_length}")
            self.cpb_mlp = nn.Sequential(
                nn.Linear(self.rank, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, dim, bias=False),
            )
        else:
            logger.debug(f"Initializing PosEmbMLPSwinv1D (Conv1d) dim={dim}, rank={rank}, seq_length={seq_length}")
            self.cpb_mlp = nn.Sequential(
                nn.Conv1d(self.rank, 512, 1, bias=True),
                nn.ReLU(),
                nn.Conv1d(512, dim, 1, bias=False),
            )
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        # relative_bias = torch.zeros(1, seq_length, dim)
        # self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def forward(self, input_tensor, coordsz=[8,8,5]):
        logger.debug(f"PosEmbMLPSwinv1D input shape: {input_tensor.shape}, coordsz: {coordsz}, rank: {self.rank}, conv: {self.conv}")
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        # if self.deploy:
        #     return input_tensor + self.relative_bias
        # else:
        #     self.grid_exists = False
        
        if not self.grid_exists:
            logger.debug(f"  Generating 3D relative coords for grid {coordsz}")
            self.grid_exists = True
            h, w, d = coordsz

            # Create relative coordinate tensors for each dimension
            relative_coords_h = torch.arange(0, h, device=input_tensor.device, dtype=input_tensor.dtype)
            relative_coords_w = torch.arange(0, w, device=input_tensor.device, dtype=input_tensor.dtype)
            relative_coords_d = torch.arange(0, d, device=input_tensor.device, dtype=input_tensor.dtype)

            # breakpoint()
            # Create 3D meshgrid
            relative_coords_table = torch.stack(
                torch.meshgrid(
                    [relative_coords_h, relative_coords_w, relative_coords_d],
                    indexing='ij'
                )
            ).contiguous().unsqueeze(0)  # Shape: [1, 3, h, w, d]

            # Center and normalize each dimension separately
            if h > 1:
                relative_coords_table[0, 0] -= h // 2  # height dimension
            if w > 1:
                relative_coords_table[0, 1] -= w // 2  # width dimension
            if d > 1:
                relative_coords_table[0, 2] -= d // 2  # depth dimension

            relative_coords_table = relative_coords_table.float()
            if h > 1:
                relative_coords_table[0, 0] /= (h // 2)  # normalize height
            if w > 1:
                relative_coords_table[0, 1] /= (w // 2)  # normalize width
            if d > 1:
                relative_coords_table[0, 2] /= (d // 2)  # normalize depth
            # relative_coords_table[0, 0] /= (h // 2)  # normalize height
            # relative_coords_table[0, 1] /= (w // 2)  # normalize width
            # relative_coords_table[0, 2] /= (d // 2)  # normalize depth

            if not self.conv:
                self.pos_emb = self.cpb_mlp(
                    relative_coords_table.permute(0, 2, 3, 4, 1).reshape(-1, h*w*d, 3)
                )
            else:
                self.pos_emb = self.cpb_mlp(
                    relative_coords_table.squeeze(0).reshape(3, -1)
                )

            self.relative_bias = self.pos_emb
        # breakpoint()
        input_tensor = input_tensor + self.pos_emb
        return input_tensor

class CrossWindowAttention(nn.Module):
    """Cross-window attention where queries come from a separate input."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        # Separate Q projection for query input
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # KV projection for context input
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim)

        # Dropouts
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_q: Query input tensor (B, Nq, C)
            x_kv: Key-value input tensor (B, Nkv, C)
            mask: Optional attention mask
        """
        B, Nq, C = x_q.shape
        _, Nkv, _ = x_kv.shape

        # Generate Q from x_q
        q = (
            self.q(x_q)
            .reshape(B, Nq, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Generate K,V from x_kv
        kv = (
            self.kv(x_kv)
            .reshape(B, Nkv, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # Each shape: (B, num_heads, Nkv, head_dim)

        if (
            torch.backends.cuda.flash_sdp_enabled()
            or torch.backends.cuda.cudnn_sdp_enabled()
            or torch.backends.cuda.mem_efficient_sdp_enabled()
            or torch.backends.cuda.math_sdp_enabled()
        ) and mask is None:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj_drop(self.proj(x))
        return x

    def run_attn(self, q, k, v, mask=None):
        B, H, Nq, D = q.shape
        C = H * D
        if (
            torch.backends.cuda.flash_sdp_enabled()
            or torch.backends.cuda.cudnn_sdp_enabled()
            or torch.backends.cuda.mem_efficient_sdp_enabled()
            or torch.backends.cuda.math_sdp_enabled()
        ) and mask is None:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj_drop(self.proj(x))
        return x

    def get_qkv(self, x_q, x_kv):
        B, Nq, C = x_q.shape
        _, Nkv, _ = x_kv.shape
        q = (
            self.q(x_q)
            .reshape(B, Nq, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x_kv)
            .reshape(B, Nkv, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        return q, k, v

    def get_q(self, x):
        B, Nq, C = x.shape
        q = self.q(x).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return q

    def get_kv(self, x):
        B, Nkv, C = x.shape
        kv = (
            self.kv(x)
            .reshape(B, Nkv, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        return [k, v]


class CrossWindowBlock(nn.Module):
    """Transformer block with cross-window attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        # breakpoint()
        # Cross window attention
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossWindowAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_q: Query input tensor
            x_kv: Key-value input tensor
            mask: Optional attention mask
        """
        # Cross window attention with residual
        x = x_q + self.drop_path(
            self.attn(self.norm1_q(x_q), self.norm1_kv(x_kv), mask)
        )

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def get_qkv(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q
        x_q = self.norm1_q(x_q)
        x_kv = self.norm1_kv(x_kv)
        q, k, v = self.attn.get_qkv(x_q, x_kv)
        return q, k, v

    def get_qkv_tokens(self, x, key="q"):
        if key == "q":
            return self.attn.get_q(self.norm1_q(x))
        if key == "kv":
            return self.attn.get_kv(self.norm1_kv(x))

    def xattn_qkv(self, q, k, v, mask=None):
        x = self.attn.run_attn(q, k, v, mask)
        return x

    def mlp_residual(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def skip_with_drop(self, x, skip):
        x = x + self.drop_path(skip)
        return x


class FastMultiScaleAttentionBlock(nn.Module):
    """
    MultiScaleAttentionBlock: Implements multi-scale attention with various communication protocols.
    Supports weight sharing and different attention strategies including sparse, parallel, and apollo variants.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        sr_ratio=1,
        norm_layer=nn.LayerNorm,
        pool_op="max",
        merge_ratio=16,
        local2global=4,
        window_dims=4,
        window_size=8,
        weight_share=True,
        ignore_registers=False,
        accumulate_window_summary=True,
        multiscale_layout=None,
        **kwargs,
    ):
        super().__init__()
        self._init_basic_config(
            dim,
            num_heads,
            drop,
            attn_drop,
            qkv_bias,
            mlp_ratio,
            drop_path,
            merge_ratio,
            local2global,
            window_dims,
            init_values,
            norm_layer,
            weight_share,
            multiscale_layout,
        )

        self._init_multiscale_attention()
        self._init_multiscale_position_embeddings()

    def _init_basic_config(
        self,
        dim,
        num_heads,
        drop,
        attn_drop,
        qkv_bias,
        mlp_ratio,
        drop_path,
        merge_ratio,
        local2global,
        window_dims,
        init_values,
        norm_layer,
        weight_share,
        multiscale_layout,
    ):
        """Initialize basic configuration parameters."""
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**0.5
        self.merge_ratio = merge_ratio
        self.local2global = local2global
        self.window_dims = window_dims
        self.init_values = init_values
        self.norm_layer = norm_layer
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path

        # Dropout configurations
        self.attn_drop_p = attn_drop
        self.drop = drop
        self.proj_drop = nn.Dropout(drop)

        # Component configurations
        self.qkv_bias = qkv_bias
        self.additional_scale = None
        # self.num_tokens_per_window = math.prod(merge_ratio)
        # self.num_windows = math.prod(window_dims)
        self.communication_protocol = "all2all_sattn__sequential"

        # aggregate information from the lower to higher levels per block
        # currently supports : one2one_xattn, no_l2g
        self.aggregation_protocol = "one2one_xattn"
        self.multiscale_layout = multiscale_layout

        self.out_scales = {}
        self.cache_qkv = {}

        self.weight_share = weight_share

    def _init_multiscale_attention(self):
        """Initialize multiscale attention components, with one x-attn block per window."""
        self.blocks = nn.ModuleList(
            [
                CrossWindowBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop,
                    attn_drop=self.attn_drop_p,
                    drop_path=self.drop_path,
                    norm_layer=self.norm_layer,
                )
                for layout in self.multiscale_layout
            ]
        )

    def _init_multiscale_position_embeddings(self):
        """Initialize position embeddings.

        Args:
            num_scales (int): Number of different scale position embeddings to create.
        """
        self.posemb = nn.ModuleList(
            [
                PosEmbMLPSwinv1D(self.dim, rank=3, seq_length=layout["seq_length"])
                for layout in self.multiscale_layout
            ]
        )

    def propagate_bottom_up(
        self,
        stages: List[torch.Tensor],
        grid_sizes: List[Tuple[int, int]],
        merge_ratio: int,
        local2global: int,
    ) -> List[torch.Tensor]:
        """
        Propagate information from local to global representations in bottom-up pass.

        Args:
            stages: List of tensors at different scales
            grid_sizes: List of grid sizes for each scale
            merge_ratio: Size of merging window
            downscaling_op: Pooling operator for downscaling

        Returns:
            Updated list of stages with propagated information
        """
        # breakpoint()
        # downscaling_op = nn.MaxPool2d(kernel_size=local2global)
        downscaling_op = nn.MaxPool3d(kernel_size=local2global)

        for i in range(len(stages) - 1):
            current_stage = stages[i]
            current_grid_size = grid_sizes[i]
            nw = math.prod(current_grid_size)

            # Downscaling process
            current_stage = rearrange(
                current_stage,
                "bnw (m0 m1 m2) c -> bnw c m0 m1 m2",
                m0=merge_ratio[0],
                m1=merge_ratio[1],
                m2=merge_ratio[2],
            )
            current_stage = downscaling_op(current_stage)

            # Spatial rearrangement
            current_stage = rearrange(
                current_stage, "(b nw) c m0 m1 m2 -> b nw m0 m1 m2 c", nw=nw
            )
            current_stage = rearrange(
                current_stage,
                "b (d h w) m0 m1 m2 c -> b (d m0) (h m1) (w m2) c",
                h=current_grid_size[0],
                w=current_grid_size[1],
                d=current_grid_size[2],
            )

            # Handle different spatial dimensions
            d, h, w = current_stage.shape[1:4]
            # if h < merge_ratio or w < merge_ratio:
            if d == merge_ratio[0] and h == merge_ratio[1] and w == merge_ratio[2]:
                local2global = rearrange(current_stage, "b d h w c -> b (d h w) c")
            elif d >= merge_ratio[0] and h >= merge_ratio[1] and w >= merge_ratio[2]:
                local2global = rearrange(
                    current_stage,
                    "b (d m0) (h m1) (w m2) c -> (b d h w) (m0 m1 m2) c",
                    m0=merge_ratio[0],
                    m1=merge_ratio[1],
                    m2=merge_ratio[2],
                )
            else:
                local2global = rearrange(current_stage, "b d h w c -> b (d h w) c")

            stages[i + 1] = stages[i + 1] + local2global

        return stages


    def forward_sequential(
        self,
        scales: List[torch.Tensor],
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        """
        Implements all variants of apollo communication.

        Args:
            scales: List of tensors for each scale level
            grid_sizes: Optional grid sizes for each scale

        Returns:
            List of processed scale tensors
        """
        # breakpoint()
        # p torch.isnan(scales[0]).any()
        # p torch.isnan(scales[1]).any()
        # p torch.isnan(scales[2]).any()
        merge_ratio = self.merge_ratio
        local2global = self.local2global
        self.num_scales = len(scales)

        # breakpoint()
        # Add position embeddings to all stages
        if self.weight_share:
            # When weights are shared, use local embedding for all stages
            scales = [self.swin_local_pos_embed(scale) for scale in scales]
        else:
            # assume a separate pos-embed for each scale
            for idx in range(self.num_scales):
                scales[idx] = self.posemb[idx](scales[idx], self.multiscale_layout[idx]["window_dims"])

        scales = self.propagate_bottom_up(scales, grid_sizes, merge_ratio, local2global)
        self.out_scales = {}

        # message passing from higher to lower level scales
        for S in range(self.num_scales - 1, -1, -1):
            x_S = scales[S]

            if "all2all_sattn" in self.communication_protocol:
                outs = self._process__sequential__all2all_sattn(x_S, S)
                if S in self.out_scales:
                    self.out_scales[S]["version"] += 1
                    self.out_scales[S]["tokens"] = outs
                else:
                    self.out_scales[S] = {"version": 1, "tokens": outs}
            else:
                raise NotImplementedError

        # breakpoint()
        # # message passing from lower to higher level scales
        if self.aggregation_protocol != "nol2g":
            if self.aggregation_protocol == "one2one_xattn":
                fn = self._aggregate_one2one_xattn
            else:
                raise NotImplementedError
            for S in range(1, self.num_scales):
                outs = fn(S)
                self.out_scales[S]["version"] += 1
                self.out_scales[S]["tokens"] = outs

        # delete the cache and outscales
        out_scales = [self.out_scales[S]["tokens"] for S in range(self.num_scales)]
        self.out_scales = {}
        self.cache_qkv = {}
        return out_scales

    def forward(self, scales, grid_sizes=None):
        if "sequential" in self.communication_protocol:
            return self.forward_sequential(scales, grid_sizes)
        else:
            raise NotImplementedError

    def get_qkv(self, x_S, S, keys=["q", "kv"], update_cache=False):
        """
        implements a minimal QKV cache
        """
        # update if cache version and token version are different
        for key in keys:
            cache_idx = f"{S}-{key}"
            if cache_idx in self.cache_qkv:
                if (
                    self.cache_qkv[cache_idx]["version"]
                    != self.out_scales[S]["version"]
                ):
                    self.cache_qkv[cache_idx] = {
                        "tokens": self.blocks[S].get_qkv_tokens(x_S, key),
                        "version": self.out_scales[S]["version"],
                    }
                    # print(f"------------------------------> Generating new cache for {cache_idx}")
                # else:
                #     print(f"------------------------------> Re-using cache for {cache_idx}")
            else:
                self.cache_qkv[cache_idx] = {
                    "tokens": self.blocks[S].get_qkv_tokens(x_S, key),
                    "version": 0,
                }
                # print(f"------------------------------> Generating new cache for {cache_idx}")

        qkv = []
        if "q" in keys:
            qkv.append(self.cache_qkv[f"{S}-q"]["tokens"])
        if "kv" in keys:
            qkv.extend(self.cache_qkv[f"{S}-kv"]["tokens"])
        return qkv

    def _aggregate_one2one_xattn(self, S):
        """
        Aggregate cross-attention from scale S to T.
        """
        # breakpoint()
        x_S = self.out_scales[S]["tokens"]
        x_Sm1 = self.out_scales[S - 1]["tokens"]

        q_S = self.get_qkv(x_S, S, keys=["q"])[0]
        k_Sm1, v_Sm1 = self.get_qkv(x_Sm1, S - 1, keys=["kv"])

        ## assume Sm1 is B x H x H x C with window size KxK
        ## then Sm is B x [H/K K] [H/K K] x C -> [B x (H/K * H/K)] x K x K x C
        # kD, kH, kW = self.multiscale_layout[S]["grid_size"]
        kH, kW, kD = self.multiscale_layout[S]["grid_size"]
        # m1 = int(math.sqrt(self.multiscale_layout[S]["window_size"]))
        # m1, m2, m0 = self.multiscale_layout[S]["window_dims"]
        mH, mW, mD = self.multiscale_layout[S]["window_dims"]
        q_S = rearrange(
            q_S,
            "(b kD kH kW) h (mD mH mW) c -> b h (kD mD) (kH mH) (kW mW) c",
            kD=kD, kH=kH, kW=kW, mD=mD, mH=mH, mW=mW
        )
        # m1, m2, m0 = self.multiscale_layout[S]["window_dims"]
        mH, mW, mD = self.multiscale_layout[S]["window_dims"]
        sH, sW, sD = self.multiscale_layout[S - 1]["grid_size"]
        # q_S = rearrange(q_S, "b h (sD m0) (sH m1) (sW m2) c -> (b sD sH sW) h m0 m1 m2 c", sD=sD, sH=sH, sW=sW)
        q_S = rearrange(
            q_S,
            "b h (sD mD) (sH mH) (sW mW) c -> (b sD sH sW) h mD mH mW c",
            sD=sD, sH=sH, sW=sW #, mD=mD, mH=mH, mW=mW
        )
        # m1 = int(math.sqrt(q_S.shape[2]))
        m0, m1, m2 = q_S.shape[2:5]
        q_S = rearrange(q_S, "b h m0 m1 m2 c -> b h (m0 m1 m2) c", m0=m0, m1=m1, m2=m2)

        xattn_l2g = self.blocks[S].xattn_qkv(q_S, k_Sm1, v_Sm1)
        xattn_l2g = rearrange(xattn_l2g, "(b sD sH sW) (m0 m1 m2) c -> b (sD m0) (sH m1) (sW m2) c", sD=sD, sH=sH, sW=sW, m0=m0, m1=m1, m2=m2)
        xattn_l2g = rearrange(xattn_l2g, "b (kD m0) (kH m1) (kW m2) c -> (b kD kH kW) (m0 m1 m2) c", kD=kD, kH=kH, kW=kW)

        # xattn_l2g = rearrange(xattn_l2g, "(b k) n c -> b (k n) c", b=b)
        x_S = self.blocks[S].skip_with_drop(x_S, xattn_l2g)
        x_S = self.blocks[S].mlp_residual(x_S)

        return x_S

    def _process__sequential__all2all_sattn(self, x_S, S):
        # get the QKV for x_S
        q_S, k_S, v_S = self.get_qkv(x_S, S)

        k_Sp1, v_Sp1 = [k_S], [v_S]
        if len(self.out_scales) > 0:
            for T, out_t in self.out_scales.items():
                x_t = out_t["tokens"]
                num_repeats = x_S.shape[0] // x_t.shape[0]
                k_t, v_t = self.get_qkv(x_t, T, keys=["kv"])
                k_t = k_t.repeat_interleave(num_repeats, dim=0)
                v_t = v_t.repeat_interleave(num_repeats, dim=0)

                k_Sp1.append(k_t)
                v_Sp1.append(v_t)

        k_Sp1 = torch.cat(k_Sp1, dim=2)
        v_Sp1 = torch.cat(v_Sp1, dim=2)

        x_S = self.blocks[S].skip_with_drop(
            x_S, self.blocks[S].xattn_qkv(q_S, k_Sp1, v_Sp1)
        )
        x_S = self.blocks[S].mlp_residual(x_S)

        return x_S



class AtlasLayer(nn.Module):
    """
    AtlasLayer: A single layer of the AtlasMultiScale architecture that processes
    input features through multiple attention blocks with window-based operations.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        input_resolution: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        only_local: bool = False,
        multiscale_layout=None,
        merge_ratio=8,
        local2global=4,
        **kwargs,
    ):
        """Initialize the ApolloLayer.

        Args:
            dim: Feature dimension size
            depth: Number of attention blocks in the layer
            input_resolution: Input spatial resolution
            num_heads: Number of attention heads
            window_size: Size of local attention windows
            conv: Whether to use convolution-based processing
            mlp_ratio: Expansion ratio for MLP hidden dimension
            qkv_bias: Enable bias terms in QKV projections
            qk_scale: Scaling factor for QK attention
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            only_local: Restrict to local attention only
        """
        super().__init__()

        # Basic configuration
        self.conv = conv
        self.window_size = window_size
        self.grid_size = input_resolution // window_size
        self.num_windows = self.grid_size**2

        # Calculate stride ratio for attention
        sr_ratio = input_resolution // window_size if not only_local else 1

        # Handle drop path scheduling
        if isinstance(drop_path, list):
            drop_path_rates = drop_path
        else:
            drop_path_rates = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                FastMultiScaleAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rates[i],
                    sr_ratio=sr_ratio,
                    window_size=window_size,
                    input_resolution=input_resolution,
                    weight_share=False,
                    merge_ratio=merge_ratio,
                    local2global=local2global,
                    multiscale_layout=multiscale_layout,
                )
                for i in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, grid_sizes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Forward pass for the Atlas Layer.

        Args:
            x: Input tensor
            grid_sizes: List of grid sizes for multi-scale processing

        Returns:
            Processed tensor after attention blocks
        """
        # breakpoint()
        # Process through attention blocks
        for block in self.blocks:
            x = block(x, grid_sizes)

        return x


class AtlasMultiScale(AbstractModel):
    """
    AtlasMultiScale: A multi-scale vision transformer architecture that processes
    images at multiple resolutions using a hierarchical structure.
    """

    def __init__(
        self,
        args,
        dim,
        in_dim,
        num_features,
        depths=8,
        window_size=8,
        mlp_ratio=4,
        num_heads=8,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer=AtlasLayer,
        img_size=[256, 256, 1],
        merge_ratio=8,
        local2global=4,
        patch_size=16,
        bsz=64,
        embed_op="convstem",
        **kwargs,
    ):
        super().__init__(args)
        # Model configuration
        self.num_classes = num_classes
        self.merge_ratio = merge_ratio
        self.local2global = local2global
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.num_features = num_features
        self.readout_norm = "batchnorm"
        self.depths = depths
        self.img_size = img_size
        self.embed_op = embed_op
        self.bsz = bsz

        # Layers initialization
        if self.embed_op == "conv2d":
            self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=96, dim=self.dim)
        elif self.embed_op == "convstem":
            self.patch_embed = MultiConvEmbed3D(
                in_chans=in_chans, embed_dim=self.num_features
            )
        elif self.embed_op == "convCXR":
            self.patch_embed = ConvCXREmbed(
                in_chans=in_chans, embed_dim=self.num_features
            )
        elif self.embed_op == "convCXRMultiView":
            self.patch_embed = MultiViewConvCXR(
                in_chans=in_chans, embed_dim=self.num_features
            )

        self.multiscale_layout = self.prepare_multiscale_layout(
            self.img_size, self.merge_ratio, self.local2global, self.patch_size
        )

        # Drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depths)]

        # Initialize transformer layers
        self.levels = nn.ModuleList()
        layer_args = self._prepare_layer_args(
            kwargs["layer_args"].copy(),
            dim=dim,
            num_features=self.num_features,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dpr=dpr,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            multiscale_layout=self.multiscale_layout,
            merge_ratio=merge_ratio,
            local2global=local2global,
        )
        self.levels.append(layer(**layer_args))

        # Initialize normalization and classification head
        self._init_norm_and_head()

    def prepare_multiscale_layout(
        self, img_size, merge_ratio, local2global, patch_size
    ):
        """
        given the input size, merge_ratio and local2global
        prepare the layout for multiscale attention and config
        for the architecture
        """
        # breakpoint()
        H, W, D = img_size
        pH, pW, pD = patch_size
        lD, lH, lW = local2global
        mD, mH, mW = merge_ratio

        h, w, d = H // pH, W // pW, D // pD
        seqlen = h * w * d
        
        downsampling_ratio = lD * lH * lW
        min_seqlen = mD * mH * mW

        kD, kH, kW = 1, 1, 1

        if seqlen <= min_seqlen:
            return [
                {
                    "grid_size": [1, 1, 1],
                    "window_size": seqlen,
                    "num_windows": 1,
                    "seq_length": seqlen,
                    "window_dims": [h, w, d],
                }
            ]
        num_stages = (
            math.ceil((math.log2(seqlen / min_seqlen) / math.log2(downsampling_ratio)))
            + 1
        )

        multiscale_layout = []

        for scale in range(num_stages):
            h0, w0, d0 = h // kH, w // kW, d // kD

            # if h0 <= mH or w0 <= mW or d0 <= mD:
            if h0 == mH and w0 == mW and d0 == mD:
                # if exact merge is possible, then dont add another scale
                multiscale_layout.append(
                    {
                        "grid_size": [1, 1, 1],
                        "window_size": h0 * w0 * d0,
                        "num_windows": 1,
                        "seq_length": h0 * w0 * d0,
                        "window_dims": [h0, w0, d0],
                    }
                )
                break
            elif h0 >= mH and w0 >= mW and d0 >= mD:
                # grid_size = current_resolution // merge_ratio
                grid_size = [h0//mH, w0//mW, d0//mD]
                multiscale_layout.append(
                    {
                        "grid_size": grid_size,
                        "window_size": mH * mW * mD,
                        "num_windows": grid_size[0] * grid_size[1] * grid_size[2],
                        "seq_length": h0 * w0 * d0,
                        "window_dims": [mH, mW, mD],
                    }
                )
            else:
                multiscale_layout.append(
                    {
                        "grid_size": [1, 1, 1],
                        "window_size": h0 * w0 * d0,
                        "num_windows": 1,
                        "seq_length": h0 * w0 * d0,
                        "window_dims": [h0, w0, d0],
                    }
                )
                break

            kD *= lD
            kH *= lH
            kW *= lW
        
        # breakpoint()

        # assert len(multiscale_layout) == num_stages
        return multiscale_layout

    def _prepare_layer_args(self, layer_args, **kwargs):
        """Prepare arguments for layer initialization."""
        layer_args.update(
            {
                "dim": kwargs["dim"],
                "depth": kwargs["depths"],
                "num_heads": kwargs["num_heads"],
                "window_size": kwargs["window_size"],
                "mlp_ratio": kwargs["mlp_ratio"],
                "drop_path": kwargs["dpr"],
                "qkv_bias": kwargs["qkv_bias"],
                "qk_scale": kwargs["qk_scale"],
                "drop": kwargs["drop_rate"],
                "attn_drop": kwargs["attn_drop_rate"],
                "transformer_blocks": False,
                "conv": False,
                "only_local": False,
                "multiscale_layout": self.multiscale_layout,
                "merge_ratio": kwargs["merge_ratio"],
                "local2global": kwargs["local2global"],
                "input_resolution": layer_args["input_resolution"] // 4,
            }
        )
        return layer_args

    def _init_norm_and_head(self):
        """Initialize normalization layers and classification head."""
        if self.readout_norm == "batchnorm":
            self.norm = nn.BatchNorm2d(self.dim)
        elif self.readout_norm == "layernorm":
            self.norm = LayerNorm2d(self.dim)
        else:  # flatln
            self.norm = nn.LayerNorm(self.dim)

        self.avgpool = (
            nn.AdaptiveAvgPool2d(1) if self.readout_norm != "flatln" else None
        )
        self.head = (
            nn.Linear(self.dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

    def forward_raw(self, x, inflate=True):
        """Process input through the raw forward pass."""
        # breakpoint()
        x = self.patch_embed(x)
        # breakpoint()
        stages = self._build_multiscale_tokens(x)
        return stages

    def _build_multiscale_tokens(self, x_BCDHW):
        """
        Build tokens for all scales in 3D

        Assuming inputs from the patch embed to be x_BCDHW,
        build tokens at all scales using maxpool with
        progressively larger window sizes (aka downscaling)
        """
        seqlen = x_BCDHW.shape[2] * x_BCDHW.shape[3] * x_BCDHW.shape[4]
        downsampling_ratio = self.local2global[0] * self.local2global[1] * self.local2global[2]
        kernel_size = (1, 1, 1)

        min_seqlen = self.merge_ratio[0] * self.merge_ratio[1] * self.merge_ratio[2]

        num_stages = (
            math.ceil((math.log2(seqlen / min_seqlen) / math.log2(downsampling_ratio)))
            + 1
        )

        stages = []
        self.grid_sizes = []

        for scale in range(num_stages):
            local2global_op = nn.MaxPool3d(kernel_size=kernel_size)
            x_scale_BCDHW = local2global_op(x_BCDHW)
            kernel_size = tuple(k * l for k, l in zip(kernel_size, self.local2global))

            # check if windowing is possible
            b, c, d, h, w = x_scale_BCDHW.shape

            ## if exact merge is possible, then dont add another scale
            if d == self.merge_ratio[0] and h == self.merge_ratio[1] and w == self.merge_ratio[2]:
                x_scale_win = rearrange(x_scale_BCDHW, "b c d h w -> b (d h w) c")
                grid_size = [1, 1, 1]
            ## only reduce in dimensions where merge is possible
            elif d >= self.merge_ratio[0] and h >= self.merge_ratio[1] and w >= self.merge_ratio[2]:
                # run windowing
                x_scale_win = rearrange(
                    x_scale_BCDHW,
                    "b c (d m0) (h m1) (w m2) -> b d h w m0 m1 m2 c",
                    m0=self.merge_ratio[0],
                    m1=self.merge_ratio[1],
                    m2=self.merge_ratio[2],
                )
                grid_size = [x_scale_win.shape[2], x_scale_win.shape[3], x_scale_win.shape[1]]
                x_scale_win = rearrange(
                    x_scale_win, "b d h w m0 m1 m2 c -> (b d h w) (m0 m1 m2) c"
                )
            else:
                x_scale_win = rearrange(x_scale_BCDHW, "b c d h w -> b (d h w) c")
                grid_size = [1, 1, 1]

            stages.append(x_scale_win)
            self.grid_sizes.append(grid_size)

            if math.prod(grid_size) == 1:
                break

        # breakpoint()
        return stages

    def _build_multiscale_tokens2d(self, x_BCHW):
        """
        Build tokens for all scales

        Assuming inputs from the patch embed to be x_BCHW,
        build tokens at all scales using maxpool with
        progressively larger window sizes (aka downscaling)
        """
        # breakpoint()
        seqlen = x_BCHW.shape[2] * x_BCHW.shape[3]
        downsampling_ratio = self.local2global**2
        kernel_size = 1

        min_seqlen = self.merge_ratio**2

        num_stages = (
            math.ceil((math.log2(seqlen / min_seqlen) / math.log2(downsampling_ratio)))
            + 1
        )

        stages = []
        self.grid_sizes = []

        for scale in range(num_stages):
            local2global_op = nn.MaxPool2d(kernel_size=kernel_size)
            x_scale_BCHW = local2global_op(x_BCHW)
            kernel_size *= self.local2global

            # check if windowing is possible
            b, c, h, w = x_scale_BCHW.shape
            if h > self.merge_ratio and w > self.merge_ratio:
                # run windowing
                x_scale_win = rearrange(
                    x_scale_BCHW,
                    "b c (h m1) (w m2) -> b h w m1 m2 c",
                    m1=self.merge_ratio,
                    m2=self.merge_ratio,
                )
                grid_size = [x_scale_win.shape[1], x_scale_win.shape[2]]
                x_scale_win = rearrange(
                    x_scale_win, "b h w m1 m2 c -> (b h w) (m1 m2) c"
                )
            else:
                x_scale_win = rearrange(x_scale_BCHW, "b c h w -> b (h w) c")
                grid_size = [1, 1]

            stages.append(x_scale_win)
            self.grid_sizes.append(grid_size)

        return stages

    def forward_features(self, x, process_readout=True, grid_sizes=None):
        """Forward pass through feature extraction layers."""
        grid_sizes = [layout["grid_size"] for layout in self.multiscale_layout]
        # breakpoint()

        for level in self.levels:
            x = level(x, grid_sizes=grid_sizes)

        if process_readout:
            if self.readout_norm == "batchnorm":
                return self._process_batchnorm_features(x)
            elif self.readout_norm == "flatln":
                return self._process_flatln_features(x)
            else:  # layernorm
                return self._process_layernorm_features(x)
        else:
            return x

    def _process_batchnorm_features(self, x):
        """Process features using batch normalization."""
        # breakpoint()
        bsz = self.bsz
        resolution = self.img_size[0]
        # patchembed_downsample = 16
        if self.embed_op == "conv2d":
            patchembed_downsample = 4
        else:
            patchembed_downsample = 16
        downsample = self.local2global

        readout_feats = None
        readout_res = resolution // patchembed_downsample

        m0 = readout_res
        for scale in x:
            x0 = rearrange(scale, "(b nw) k c -> b c (nw k)", b=bsz)
            x0 = rearrange(x0, "b c (m0 m1) -> b c m0 m1", m0=m0, m1=m0)
            x0 = F.interpolate(x0, size=(readout_res, readout_res), mode="nearest")
            m0 = m0 // downsample

            if readout_feats is None:
                readout_feats = x0
            else:
                readout_feats += x0

        readout_feats = self.norm(readout_feats)
        readout_feats = self.avgpool(readout_feats)
        return torch.flatten(readout_feats, 1)

    def _process_flatln_features(self, x):
        """Process features using flat layer normalization."""
        feats = rearrange(x[0], "(b nw) k c -> b (nw k) c", b=64)
        x = self.norm(feats)
        return x.mean(1)

    def _process_layernorm_features(self, x):
        """Process features using layer normalization."""
        x = self.norm(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x, batch=None):
        """Forward pass through the entire network."""
        # breakpoint()
        x = self.forward_raw(x)
        x = self.forward_features(x)
        x = self.head(x)
        return {"logit": x}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Get keywords for parameters that should not use weight decay."""
        return {"rpb"}
