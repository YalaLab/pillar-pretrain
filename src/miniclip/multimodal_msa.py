import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp


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



class RelativePosEmb(nn.Module):
    """
    Learnable relative positional embedding for 3D grids, supporting linear or conv projections.
    Args:
        dim (int): Output embedding dimension.
        rank (int): Number of spatial dims (default: 2).
        conv (bool): Use Conv1d if True, else Linear.
    """

    def __init__(self, dim, rank=2, conv=False):
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

        self.modality_grid_exists = {}
        self.posemb = {}
        self.conv = conv

    def forward(self, x, grid_size=[8, 8, 5], modality="chest_cxr_single_view"):
        """
        Add relative positional embedding to x for a given grid size and modality.
        Args:
            x (Tensor): Input tensor.
            grid_size (list): [H, W, D] grid size.
            modality (str): Modality key for caching.
        Returns:
            Tensor: x with positional embedding added.
        """

        # This prevents reusing tensors from a previous training step's freed graph.
        # The cache will still work correctly during evaluation (model.eval()).
        if self.training:
            self.modality_grid_exists = {}
            self.posemb = {}

        if modality not in self.modality_grid_exists:
            self.modality_grid_exists[modality] = True
            h, w, d = grid_size

            # Create relative coordinate tensors for each dimension
            relative_coords_h = torch.arange(0, h, device=x.device, dtype=x.dtype)
            relative_coords_w = torch.arange(0, w, device=x.device, dtype=x.dtype)
            relative_coords_d = torch.arange(0, d, device=x.device, dtype=x.dtype)

            # Create 3D meshgrid
            relative_coords_table = (
                torch.stack(
                    torch.meshgrid(
                        [relative_coords_h, relative_coords_w, relative_coords_d],
                        indexing="ij",
                    )
                )
                .contiguous()
                .unsqueeze(0)
            )  # Shape: [1, 3, h, w, d]

            # Center and normalize each dimension separately
            if h > 1:
                relative_coords_table[0, 0] -= h // 2  # height dimension
            if w > 1:
                relative_coords_table[0, 1] -= w // 2  # width dimension
            if d > 1:
                relative_coords_table[0, 2] -= d // 2  # depth dimension

            relative_coords_table = relative_coords_table.float()
            if h > 1:
                relative_coords_table[0, 0] /= h // 2  # normalize height
            if w > 1:
                relative_coords_table[0, 1] /= w // 2  # normalize width
            if d > 1:
                relative_coords_table[0, 2] /= d // 2  # normalize depth

            if not self.conv:
                posemb = self.cpb_mlp(
                    relative_coords_table.permute(0, 2, 3, 4, 1).reshape(
                        -1, h * w * d, 3
                    )
                )
            else:
                posemb = self.cpb_mlp(relative_coords_table.squeeze(0).reshape(3, -1))
            self.posemb[modality] = posemb

        x = x + self.posemb[modality]
        return x


class MultiScaleAttentionBlock(nn.Module):
    """
    MultiScaleAttentionBlock: Implements multi-scale attention with various communication protocols.
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
        norm_layer=nn.LayerNorm,
        pool_op="max",
        window_dims=4,
        weight_share=True,
        ignore_registers=False,
        accumulate_window_summary=True,
        num_scales=None,
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
            window_dims,
            init_values,
            norm_layer,
            weight_share,
            num_scales,
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
        window_dims,
        init_values,
        norm_layer,
        weight_share,
        num_scales,
    ):
        """Initialize basic configuration parameters."""
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**0.5
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
        self.communication_protocol = "all2all_sattn__sequential"

        # aggregate information from the lower to higher levels per block
        # currently supports : one2one_xattn
        self.aggregation_protocol = "one2one_xattn"
        self.num_scales = num_scales

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
                for scale_idx in range(self.num_scales)
            ]
        )

    def _init_multiscale_position_embeddings(self):
        """Initialize position embeddings.

        Args:
            num_scales (int): Number of different scale position embeddings to create.
        """
        self.posemb = nn.ModuleList(
            [RelativePosEmb(self.dim, rank=3) for scale_idx in range(self.num_scales)]
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

    def forward(
        self,
        scales,
        grid_sizes=None,
        multiscale_layout=None,
        merge_ratio=None,
        local2global=None,
        modality: str = "chest_cxr_single_view",
    ):
        if "sequential" in self.communication_protocol:
            return self.forward_sequential(
                scales,
                grid_sizes,
                multiscale_layout,
                merge_ratio,
                local2global,
                modality=modality,
            )
        else:
            raise NotImplementedError

    def forward_sequential(
        self,
        scales: List[torch.Tensor],
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
        multiscale_layout=None,
        merge_ratio=None,
        local2global=None,
        modality: str = "chest_xray_single_view",
    ) -> List[torch.Tensor]:
        """
        Implements communication protocol for sequential processing of scales.

        Args:
            scales: List of tensors for each scale level
            grid_sizes: Optional grid sizes for each scale
            multiscale_layout: Layout information for each scale, including window dimensions

        Returns:
            List of processed scale tensors
        """
        self.num_scales = len(scales)

        # assume a separate pos-embed for each scale
        for idx in range(self.num_scales):
            scales[idx] = self.posemb[idx](
                scales[idx],
                grid_size=multiscale_layout[idx]["window_dims"],
                modality=modality,
            )

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

        ## message passing from lower to higher level scales
        # NOTE: removed other protocols for now, only one2one_xattn is supported 
        if self.aggregation_protocol == "one2one_xattn":
            fn = self._aggregate_one2one_xattn
        else:
            raise NotImplementedError

        for S in range(1, self.num_scales):
            outs = fn(S, multiscale_layout=multiscale_layout)
            self.out_scales[S]["version"] += 1
            self.out_scales[S]["tokens"] = outs

        # delete the cache and outscales
        out_scales = [self.out_scales[S]["tokens"] for S in range(self.num_scales)]
        self.out_scales = {}
        self.cache_qkv = {}
        return out_scales

    def _aggregate_one2one_xattn(self, S, multiscale_layout=None):
        """
        Aggregate cross-attention from scale S to T.
        """
        x_S = self.out_scales[S]["tokens"]
        x_Sm1 = self.out_scales[S - 1]["tokens"]

        q_S = self.get_qkv(x_S, S, keys=["q"])[0]
        k_Sm1, v_Sm1 = self.get_qkv(x_Sm1, S - 1, keys=["kv"])

        kH, kW, kD = multiscale_layout[S]["grid_size"]
        mH, mW, mD = multiscale_layout[S]["window_dims"]
        q_S = rearrange(
            q_S,
            "(b kD kH kW) h (mD mH mW) c -> b h (kD mD) (kH mH) (kW mW) c",
            kD=kD,
            kH=kH,
            kW=kW,
            mD=mD,
            mH=mH,
            mW=mW,
        )
        mH, mW, mD = multiscale_layout[S]["window_dims"]
        sH, sW, sD = multiscale_layout[S - 1]["grid_size"]

        q_S = rearrange(
            q_S,
            "b h (sD mD) (sH mH) (sW mW) c -> (b sD sH sW) h mD mH mW c",
            sD=sD,
            sH=sH,
            sW=sW, 
        )

        m0, m1, m2 = q_S.shape[2:5]
        q_S = rearrange(q_S, "b h m0 m1 m2 c -> b h (m0 m1 m2) c", m0=m0, m1=m1, m2=m2)

        xattn_l2g = self.blocks[S].xattn_qkv(q_S, k_Sm1, v_Sm1)
        xattn_l2g = rearrange(
            xattn_l2g,
            "(b sD sH sW) (m0 m1 m2) c -> b (sD m0) (sH m1) (sW m2) c",
            sD=sD,
            sH=sH,
            sW=sW,
            m0=m0,
            m1=m1,
            m2=m2,
        )
        xattn_l2g = rearrange(
            xattn_l2g,
            "b (kD m0) (kH m1) (kW m2) c -> (b kD kH kW) (m0 m1 m2) c",
            kD=kD,
            kH=kH,
            kW=kW,
        )

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
            else:
                self.cache_qkv[cache_idx] = {
                    "tokens": self.blocks[S].get_qkv_tokens(x_S, key),
                    "version": 0,
                }

        qkv = []
        if "q" in keys:
            qkv.append(self.cache_qkv[f"{S}-q"]["tokens"])
        if "kv" in keys:
            qkv.extend(self.cache_qkv[f"{S}-kv"]["tokens"])
        return qkv


class AtlasStage(nn.Module):
    """
    AtlasStage: A single stage of the AtlasMultiScale architecture that processes
    input features through multiple attention blocks with window-based operations.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        num_scales=None,
        **kwargs,
    ):
        """Initialize the AtlasStage.

        Args:
            dim: Feature dimension size
            depth: Number of attention blocks in the layer
            num_heads: Number of attention heads
            window_size: Size of local attention windows
            mlp_ratio: Expansion ratio for MLP hidden dimension
            qkv_bias: Enable bias terms in QKV projections
            qk_scale: Scaling factor for QK attention
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList(
            [
                MultiScaleAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rates[i],
                    weight_share=False,
                    num_scales=num_scales,
                )
                for i in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: List[Tuple[int, int]],
        multiscale_layout=None,
        merge_ratio=None,
        local2global=None,
        modality: str = "chest_xray_single_view",
    ) -> torch.Tensor:
        """Forward pass for the Atlas Stages.

        Args:
            x: Input tensor
            grid_sizes: List of grid sizes for multi-scale processing

        Returns:
            Processed tensor after attention blocks
        """
        # Process through attention blocks
        for block in self.blocks:
            x = block(
                x,
                grid_sizes,
                multiscale_layout,
                merge_ratio,
                local2global,
                modality=modality,
            )

        return x
