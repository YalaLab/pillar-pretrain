from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

norm_layer_map = {
    "batch": nn.BatchNorm3d,
    "instance": nn.InstanceNorm3d,
    "layer": nn.LayerNorm,
    "group": nn.GroupNorm,
}

# https://pytorch.org/vision/0.8/models.html
class Conv2Plus1D(nn.Module):
    """
    Replaces a 3x3x3 (T x H x W) kernel with two sequential convs:
      (1 x 3 x 3) + (3 x 1 x 1)
    This is the key idea of R(2+1)D.
    """
    def __init__(self, in_planes, out_planes, stride=1, *, norm_layer, norm_eps=1e-5, norm_momentum=0.1):
        super().__init__()
        # For R(2+1)D style factorization, the first conv is purely spatial:
        #     kernel_size=(1,3,3), stride=(1, s, s) if stride>1
        # The second conv is purely temporal:
        #     kernel_size=(3,1,1), stride=(s, 1, 1) if stride>1
        # We keep an overall "stride" pattern consistent with how
        # the original code used average pooling for anti-aliasing.
        self.factor = nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, 1, 1),
                bias=False
            ),
            norm_layer(out_planes, eps=norm_eps, momentum=norm_momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_planes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(1, 0, 0),
                bias=False
            )
        )

    def forward(self, x):
        return self.factor(x)

# --------------------------------------
# Minimal attention-pool for 3D
# --------------------------------------
# class AttentionPool3d(nn.Module):
#     """
#     Same logic as the original AttentionPool2d, but flatten T×H×W into
#     a single dimension. We store a positional embedding of length
#     (T×H×W + 1).
#     """
#     def __init__(self, thw_size: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         """
#         thw_size is T*H*W (the product of temporal and spatial).
#         """
#         super().__init__()
#         self.positional_embedding = nn.Parameter(
#             torch.randn(thw_size + 1, embed_dim) / embed_dim ** 0.5
#         )
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads

#     def forward(self, x: torch.Tensor):
#         """
#         x is expected to have shape [N, C, T, H, W].
#         We flatten T×H×W into one dimension => [N, C, T*H*W].
#         Then do multi-head attention as in 2D.
#         """
#         N, C, T, H, W = x.shape
#         x = x.view(N, C, T*H*W)             # [N, C, THW]
#         x = x.permute(2, 0, 1)             # => [THW, N, C]

#         # Prepend the mean pooled "CLS" token
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # => [(THW+1), N, C]

#         # Add learned positional embedding
#         x = x + self.positional_embedding[:, None, :].to(x.dtype)  # => [(THW+1), N, C]

#         # standard PyTorch multi-head attention (inlined)
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0.,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )

#         # x[0] is the pooled representation
#         return x[0]

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedAttentionPool3d(nn.Module):
    """
    A 3D attention pool that uses factorized positional embeddings:
        pos_emb(t, h, w) = time_emb[t] + space_emb[h, w]
    plus a learnable CLS token.

    Then the sequence (CLS + flattened T×H×W) is passed to multi-head attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        frames: int = 32,
        hw: int = 32,
    ):
        """
        Args:
            embed_dim:   the channel dimension C
            num_heads:   number of attention heads
            output_dim:  final projection dimension (defaults to embed_dim)
            frames:  maximum number of frames we expect
            hw:       maximum spatial height

        We create large embeddings for [frames, hw], then index
        into them dynamically in `forward(...)`.
        """
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim or embed_dim

        # Factorized embeddings
        self.time_embed = nn.Parameter(
            torch.randn(frames, embed_dim) / embed_dim**0.5
        )
        self.space_embed = nn.Parameter(
            torch.randn(hw * hw, embed_dim) / embed_dim**0.5
        )
        # Learnable CLS token (1, embed_dim)
        self.cls_embed = nn.Parameter(
            torch.randn(1, embed_dim) / embed_dim**0.5
        )

        # Projections (same usage as in the original AttentionPool3d)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, self.output_dim)

    def forward(self, x: torch.Tensor):
        """
        x shape: [N, C, T, H, W]
        We will:
          1) Flatten T×H×W => shape: [N, C, T*H*W]
          2) Prepend CLS => shape: [(T*H*W + 1), N, C]
          3) Add factorized positional embedding => same shape
          4) Perform multi-head attention => return x[0], the pooled CLS
        """
        N, C, T, H, W = x.shape
        THW = T * H * W

        # Flatten T×H×W => shape: [THW, N, C]
        x = x.view(N, C, THW).permute(2, 0, 1)  # [THW, N, C]

        # Build factorized pos embedding:
        #  - time part:   (T, C)
        #  - space part:  (H*W, C)  (we'll reshape to [H, W, C] below)
        # Then for each t,h,w: pos[t,h,w] = time_embed[t] + space_embed[h,w].
        # Flatten => shape [T*H*W, C].
        time_pos = self.time_embed[:T]      # [T, C]
        # We only support varying T, not H or W
        space_pos = self.space_embed  # [H*W, C]
        space_pos = space_pos.view(H, W, C) # => [H, W, C]

        # Broadcast-add to get shape: (T, H, W, C)
        # time_pos[t, :] + space_pos[h, w, :]
        pos_4d = time_pos[:, None, None, :] + space_pos[None, :, :, :]
        # => [T, H, W, C]

        # Flatten => shape [T*H*W, C]
        pos_flat = pos_4d.view(THW, C)

        # Concatenate the CLS token at the front
        # which has shape (1, C)
        pos_with_cls = torch.cat([self.cls_embed, pos_flat], dim=0)  # => [1 + THW, C]

        # Prepend the mean pooled "CLS" at the front of x
        # x: [THW, N, C]
        x_cls = x.mean(dim=0, keepdim=True)  # => [1, N, C]
        x = torch.cat([x_cls, x], dim=0)     # => [(THW+1), N, C]

        # Now add the factorized pos embedding => broadcast over batch dimension
        # pos_with_cls => shape (THW+1, C), we unsqueeze dim=1 => (THW+1, 1, C)
        x = x + pos_with_cls[:, None, :].to(x.dtype)

        # Standard multi-head attention (inlined)
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,  # We provide separate q/k/v weights
            in_proj_bias=torch.cat([
                self.q_proj.bias, self.k_proj.bias, self.v_proj.bias
            ]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        # x[0] is the pooled representation (the CLS token)
        return x[0]  # => [N, C] after transpose

class AvgPool3d(nn.AvgPool3d):
    """This is a custom AvgPool3d that can handle 1-frame inputs."""
    def forward(self, x):
        # Check if the temporal dimension is 1
        if x.size(2) == 1:  # Assuming x has shape [N, C, T, H, W]
            # Ensure kernel_size, stride, and padding are tuples
            kernel_size = (self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,) * 3)
            stride = (self.stride if isinstance(self.stride, tuple) else (self.stride,) * 3)
            padding = (self.padding if isinstance(self.padding, tuple) else (self.padding,) * 3)
            
            # Only pool over spatial dimensions (H, W)
            return F.avg_pool3d(
                x,
                kernel_size=(1, kernel_size[1], kernel_size[2]),
                stride=(1, stride[1], stride[2]),
                padding=(0, padding[1], padding[2])
            )
        else:
            return super().forward(x)

# --------------------------------------
# R(2+1)D Bottleneck
# --------------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, *, norm_layer, norm_eps=1e-5, norm_momentum=0.1):
        """
        Minimal modifications:
        - Replace 2D ops with 3D ops
        - Factor the "3x3" conv2 into (1x3x3) + (3x1x1) while preserving
          the "anti-aliasing" idea (avgpool if stride>1).
        """
        super().__init__()

        # 1x1x1 (no change except 2D->3D)
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes, eps=norm_eps, momentum=norm_momentum)
        self.act1 = nn.ReLU(inplace=True)

        # factorized 3D conv for “conv2”
        self.conv2 = Conv2Plus1D(planes, planes, stride=1, norm_layer=norm_layer, norm_eps=norm_eps, norm_momentum=norm_momentum)
        self.bn2 = norm_layer(planes, eps=norm_eps, momentum=norm_momentum)
        self.act2 = nn.ReLU(inplace=True)

        # anti-aliasing: if stride>1 => do an AvgPool3d
        self.avgpool = AvgPool3d(stride) if stride > 1 else nn.Identity()

        # 1x1x1 (final projection)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=norm_eps, momentum=norm_momentum)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", AvgPool3d(stride) if stride > 1 else nn.Identity()),
                ("0", nn.Conv3d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", norm_layer(planes * self.expansion, eps=norm_eps, momentum=norm_momentum))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out

# --------------------------------------
# R(2+1)D "ModifiedResNet3D"
# --------------------------------------
class ModifiedResNet3D(nn.Module):
    """
    The same "modified" ResNet design from the 2D version, except:
      - All convolutions are 3D
      - Each Bottleneck uses factorized 3D conv for its middle layer
      - Final pooling is an attention over T×H×W
    """

    def __init__(self, layers, output_dim, heads, image_size=224, frames=200, width=64, input_dim = 3, norm_layer="batch", norm_eps=1e-5, norm_momentum=0.1):
        super().__init__()
        # 3 channel inputs (this is for processing natural images). For CTs we duplicate channels three times so it's equivalent to one channel.
        self.output_dim = output_dim
        self.image_size = image_size
        self.frames = frames
        norm_layer = self.norm_layer = norm_layer_map[norm_layer]

        # --- The 3-layer "stem" in 3D ---
        #   Keep same structure: (conv→bn→relu)×3 + avgpool,
        #   but replace 2D with 3D kernels. The stride is only in spatial dims;
        #   we keep temporal stride=1 so T is unchanged in the stem.
        self.input_dim = input_dim
        self.conv1 = nn.Conv3d(input_dim, width // 2, kernel_size=(3, 3, 3),
                               stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn1 = norm_layer(width // 2, eps=norm_eps, momentum=norm_momentum)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(width // 2, width // 2, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1), bias=False)
        self.bn2 = norm_layer(width // 2, eps=norm_eps, momentum=norm_momentum)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(width // 2, width, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1), bias=False)
        self.bn3 = norm_layer(width, eps=norm_eps, momentum=norm_momentum)
        self.act3 = nn.ReLU(inplace=True)

        # anti-aliasing pool: apply stride=2 in spatial dims only
        self.avgpool = nn.AvgPool3d((1, 2, 2))

        # --- Residual layers ---
        layer_kwargs = dict(norm_layer=norm_layer, norm_eps=norm_eps, norm_momentum=norm_momentum)
        self._inplanes = width
        self.layer1 = self._make_layer(width,  layers[0], stride=1, **layer_kwargs)  # no new downsample
        self.layer2 = self._make_layer(width*2, layers[1], stride=2, **layer_kwargs)
        self.layer3 = self._make_layer(width*4, layers[2], stride=2, **layer_kwargs)
        self.layer4 = self._make_layer(width*8, layers[3], stride=2, **layer_kwargs)

        # embed_dim = width * 32 => same logic as 2D version
        embed_dim = width * 32

        # Instead of flatten & linear, use attention over T×H×W
        # After layer4, T = frames // 8, H and W each //32
        # spatial: stem stride 2, stem avgpool 2; layer2, layer3, layer4 each have stride 2
        # temporal: layer2, layer3, layer4 each have stride 2
        self.attnpool = FactorizedAttentionPool3d(
            embed_dim=embed_dim,
            num_heads=heads,
            output_dim=output_dim,
            frames=self.frames // 8,
            hw=self.image_size // 32,
        )

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1, **kwargs):
        # the first block in each layer may need a stride>1 or channel-dim expansion
        layers = [Bottleneck(self._inplanes, planes, stride, **kwargs)]
        self._inplanes = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        # Zero-init final BN in each Bottleneck
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        # keep original logic
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # no-op here
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        """
        x shape: [N, C, T, H, W] or [N, C, H, W] for 2D samples.
        If x is 2D, it will be unsqueezed to [N, C, 1, H, W].
        """
        # breakpoint()
        modality, x = list(x.items())[0]

        if x.dim() == 4:  # If input is 2D
            x = x.unsqueeze(2)  # Add a temporal dimension
        else:
            assert x.dim() == 5, "Input must be 5D (N, C, T, H, W) or 4D (N, C, H, W)"
            # duplicate C three times
            if self.input_dim == 3:
                x = x.repeat(1, 3, 1, 1, 1)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # attention-pool over (T,H,W)
        x = self.attnpool(x)
        return x


# --------------------------------------
# Example usage:
# --------------------------------------
if __name__ == "__main__":
    # Suppose we have a 16-frame video of size 224×224
    model = ModifiedResNet3D(
        layers=[3, 4, 6, 3],      # standard ResNet-50 pattern
        output_dim=512,
        heads=8,
        image_size=224,
        frames=16,
        width=64
    )

    dummy_video = torch.randn(2, 1, 16, 224, 224)  # (N=2, C=1, T=16, H=224, W=224)
    dummy_image = torch.randn(2, 1, 224, 224)  # (N=2, C=1, H=224, W=224)
    out_video = model(dummy_video)
    out_image = model(dummy_image)
    print("Output shape (video):", out_video.shape)  # [N, output_dim], e.g. [2, 512]
    print("Output shape (image):", out_image.shape)  # [N, output_dim], e.g. [2, 512]