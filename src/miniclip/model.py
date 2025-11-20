"""CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from .atlas import ColossusMultiScale
from .hf_model import HFTextEncoder, CacheTextEncoder
from .modified_resnet import ModifiedResNet
from .modified_resnet3d import ModifiedResNet3D
from .multimodal_atlas import MultiModalAtlas

# from .timm_model import TimmModel
from .transformer import (
    Attention,
    LayerNorm,
    LayerNormFp32,
    QuickGELU,
    TextTransformer,
    VisionTransformer,
    text_global_pool,
)
from .utils import to_2tuple

# from .utils import Cumulative_Probability_Layer


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    merge_ratio: Optional[List[int]] = field(default_factory=lambda: None)
    kwargs_list: Optional[List[Dict]] = field(default_factory=lambda: None)
    in_chans: int = None
    num_classes: int = None
    max_followup: int = None
    pool_name: str = None
    pool_kwargs: Optional[Dict] = None
    model_config: str = None

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.0  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    model_name: str = "vit"  # model name, 'vit' or 'resnet'

    timm_model_name: Optional[str] = (
        None  # a valid model name overrides layers, width, patch_size
    )
    timm_model_pretrained: bool = (
        False  # use (imagenet) pretrained weights for named model
    )
    timm_pool: str = (
        "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    )
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.0  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth

    type_3d: bool = False  # 3D model
    frames: int = 200  # number of frames for 3D model
    input_dim: int = 3  # input dimensions
    norm_layer: str = "batch"
    norm_eps: float = 1e-5
    norm_momentum: float = 0.1


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None
    text_model_embedding_dim: int = 512

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = "argmax"
    proj_bias: bool = False
    proj_type: str = (
        "linear"  # control final text projection, 'none' forces no projection
    )
    output_tokens: bool = False
    output_token_dim: int = None
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = "mlp"
    hf_pooler_type: str = "mean_pooler"  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        raise NotImplementedError("Timm models are not supported in this version. ")
    elif vision_cfg.model_name == "atlas":
        visual = ColossusMultiScale(
            args=vision_cfg,
            kwargs_list=vision_cfg.kwargs_list,
            embed_dim=embed_dim,
            num_classes=2,
        )
    elif vision_cfg.model_name == "multimodal_atlas":
        visual = MultiModalAtlas(
            args=vision_cfg,
            kwargs_list=vision_cfg.kwargs_list,
            embed_dim=embed_dim,
            num_classes=2,
            model_config=vision_cfg.model_config,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        if vision_cfg.type_3d:
            visual = ModifiedResNet3D(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                frames=vision_cfg.frames,
                width=vision_cfg.width,
                input_dim=vision_cfg.input_dim,
                norm_layer=vision_cfg.norm_layer,
                norm_eps=vision_cfg.norm_eps,
                norm_momentum=vision_cfg.norm_momentum,
            )
        else:
            visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width,
            )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            frames=vision_cfg.frames,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            type_3d=vision_cfg.type_3d,
            labels=getattr(vision_cfg, 'labels', []),
            in_chans=getattr(vision_cfg, 'in_chans', None),
        )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if "cache" in text_cfg.hf_model_name:
        text = CacheTextEncoder(
            proj_type=text_cfg.hf_proj_type,
            output_dim=embed_dim,
            embedding_dim=text_cfg.text_model_embedding_dim,
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
        )
    elif text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
            output_token_dim=text_cfg.output_token_dim,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        predict_risk: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.predict_risk = predict_risk

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {"positional_embedding"}
        if hasattr(self.visual, "no_weight_decay"):
            for n in self.visual.no_weight_decay():
                no_wd.add("visual." + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):

        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.predict_risk:
                out_dict["predicted_risk"] = image_features

            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return (
                image_features,
                text_features,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        predict_risk: bool = False,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        self.predict_risk = predict_risk
        if predict_risk:
            self.risk_head = Cumulative_Probability_Layer(embed_dim, max_followup=6)
        self.predict_risk = predict_risk

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if hasattr(self.text, "lock"):
            self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = set()
        if hasattr(self.visual, "no_weight_decay"):
            for n in self.visual.no_weight_decay():
                no_wd.add("visual." + n)
        if hasattr(self.text, "no_weight_decay"):
            for n in self.visual.no_weight_decay():
                no_wd.add("text." + n)
        return no_wd

    def encode_image(
        self, image, normalize: bool = False, info: str = None, output_dict=False
    ):
        if info != None:
            features = self.visual(image, info=info)
        else:
            features = self.visual(image)
        if output_dict:
            features["image_features"] = (
                F.normalize(features["hidden"], dim=-1)
                if normalize
                else features["hidden"]
            )
            return features

        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        if isinstance(text, torch.Tensor):
            features = self.text(text)
            features = F.normalize(features, dim=-1) if normalize else features
        elif isinstance(text, dict):
            # make a batch for all keys to do single pass
            keys = list(text.keys())
            batch = torch.stack([text[k] for k in keys], dim=0)
            batch = rearrange(batch, "k b d -> (k b) d")  # flatten batch for text tower
            features = self.text(batch)
            features = F.normalize(features, dim=-1) if normalize else features
            features = rearrange(features, "(k b) d -> k b d", k=len(keys))
            features = {k: v for k, v in zip(keys, features)}
        else:
            raise ValueError(f"Unsupported text input type: {type(text)}")

        return features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        info=None,
        batch=None,
    ):
        image_features = (
            self.encode_image(image, normalize=True, info=info)
            if image is not None
            else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            out_dict = {}
            if isinstance(image_features, dict):
                for k, v in image_features.items():
                    out_dict[k] = v
            else:
                out_dict["image_features"] = image_features

            out_dict["text_features"] = text_features
            out_dict["logit_scale"] = self.logit_scale.exp()

            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias

            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()




def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if "text_projection" in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(
                k.startswith(p)
                for p in (
                    "text_projection",
                    "positional_embedding",
                    "token_embedding",
                    "transformer",
                    "ln_final",
                )
            ):
                k = "text." + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
    state_dict: dict,
    quick_gelu=True,
    cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(
        model
    )  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device("cpu")):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,),
        ),
    )
    model.visual.image_size = image_size
    return model


def resize_pos_embed(
    state_dict, model, interpolation: str = "bicubic", antialias: bool = True
):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get("visual.positional_embedding", None)
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = (
        1  # FIXME detect different token configs (ie no class token, or more)
    )
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = (
            old_pos_embed[:extra_tokens],
            old_pos_embed[extra_tokens:],
        )
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info(
        "Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size
    )
    pos_emb_img = pos_emb_img.reshape(
        1, old_grid_size[0], old_grid_size[1], -1
    ).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, grid_size[0] * grid_size[1], -1
    )[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict["visual.positional_embedding"] = new_pos_embed


def resize_text_pos_embed(
    state_dict, model, interpolation: str = "linear", antialias: bool = False
):
    old_pos_embed = state_dict.get("positional_embedding", None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, "positional_embedding", None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, "positional_embedding", None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, "text pos_embed width changed!"
    if old_num_pos == num_pos:
        return

    logging.info(
        "Resizing text position embedding num_pos from %s to %s", old_num_pos, num_pos
    )
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict["positional_embedding"] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, "visual", model)
    preprocess_cfg = getattr(module, "preprocess_cfg", {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, "image_size")
        if size is not None:
            preprocess_cfg["size"] = size
        mean = getattr(module, "image_mean", None)
        if mean is not None:
            preprocess_cfg["mean"] = mean
        std = getattr(module, "image_std", None)
        if std is not None:
            preprocess_cfg["std"] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, "visual", model)
    module.image_mean = preprocess_cfg[
        "mean"
    ]  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg["std"]  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(
        preprocess_cfg
    )  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, "text", model)
    cfg = {}
    context_length = getattr(module, "context_length", None)
    if context_length is not None:
        cfg["context_length"] = context_length
    vocab_size = getattr(module, "vocab_size", None)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return cfg


# ===== SigLIP Model - Sigmoid Loss CLIP Variant =====

class SigLIP(nn.Module):
    """SigLIP model - CLIP variant using sigmoid loss instead of softmax.
    
    Key differences from CustomTextCLIP:
    1. Includes built-in distributed feature gathering
    2. Computes logits internally for the sigmoid loss function
    3. Always outputs logits in the output dictionary
    """
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = -10.0,  # SigLIP typically uses bias
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        predict_risk: bool = False,
        output_dict: bool = True,  # SigLIP always outputs dict
        **kwargs
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        self.predict_risk = predict_risk
        if predict_risk:
            self.risk_head = Cumulative_Probability_Layer(embed_dim, max_followup=6)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if hasattr(self.text, "lock"):
            self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd = set()
        if hasattr(self.visual, "no_weight_decay"):
            for n in self.visual.no_weight_decay():
                no_wd.add("visual." + n)
        if hasattr(self.text, "no_weight_decay"):
            for n in self.visual.no_weight_decay():
                no_wd.add("text." + n)
        return no_wd

    def encode_image(
        self, image, normalize: bool = False, info: str = None, output_dict=False
    ):
        if info != None:
            features = self.visual(image, info=info)
        else:
            features = self.visual(image)
        if output_dict:
            features["image_features"] = (
                F.normalize(features["hidden"], dim=-1)
                if normalize
                else features["hidden"]
            )
            return features

        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        if isinstance(text, torch.Tensor):
            features = self.text(text)
            features = F.normalize(features, dim=-1) if normalize else features
        elif isinstance(text, dict):
            # make a batch for all keys to do single pass
            keys = list(text.keys())
            batch = torch.stack([text[k] for k in keys], dim=0)
            batch = rearrange(batch, "k b d -> (k b) d")  # flatten batch for text tower
            features = self.text(batch)
            features = F.normalize(features, dim=-1) if normalize else features
            features = rearrange(features, "(k b) d -> k b d", k=len(keys))
            features = {k: v for k, v in zip(keys, features)}
        else:
            raise ValueError(f"Unsupported text input type: {type(text)}")

        return features

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        info=None,
        batch=None,
    ):
        image_features = (
            self.encode_image(image, normalize=True, info=info)
            if image is not None
            else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            out_dict = {}
            if isinstance(image_features, dict):
                for k, v in image_features.items():
                    out_dict[k] = v
            else:
                out_dict["image_features"] = image_features

            out_dict["text_features"] = text_features
            out_dict["logit_scale"] = self.logit_scale.exp()

            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias

            # Compute logits for SigLIP loss
            if text_features is not None and image_features is not None:
                if isinstance(text_features, dict):
                    logits = {}
                    for text_category, text_feat in text_features.items():
                        # GATHER FEATURES INSIDE THE MODEL
                        if dist.is_initialized() and self.training:
                            all_image_features, all_text_features = gather_features(
                                image_features, text_feat, gather_with_grad=True, world_size=dist.get_world_size()
                            )
                        else:
                            all_image_features, all_text_features = image_features, text_feat

                        # CALCULATE FINAL LOGITS INSIDE THE MODEL
                        category_logits = self.logit_scale.exp() * (all_image_features @ all_text_features.T)
                        if self.logit_bias is not None:
                            category_logits += self.logit_bias
                        
                        logits[text_category] = category_logits
                    
                    out_dict["logits"] = logits
                else:
                    # Single text tensor case
                    if dist.is_initialized() and self.training:
                        all_image_features, all_text_features = gather_features(
                            image_features, text_features, gather_with_grad=True, world_size=dist.get_world_size()
                        )
                    else:
                        all_image_features, all_text_features = image_features, text_features

                    logits = self.logit_scale.exp() * (all_image_features @ all_text_features.T)
                    if self.logit_bias is not None:
                        logits += self.logit_bias
                    
                    out_dict["logits"] = logits

            return out_dict

        # Non-dict output not supported for SigLIP
        raise ValueError("SigLIP requires output_dict=True")


# ===== TextDino Model - Cross-Modal Knowledge Distillation =====

@dataclass
class TextDinoCfg:
    """Configuration for TextDino model."""
    # Teacher (Text) config
    teacher_width: int = 768
    teacher_layers: int = 12
    teacher_heads: int = 12
    teacher_mlp_ratio: float = 4.0
    teacher_hf_model_name: Optional[str] = None  # If using HF model as teacher
    teacher_hf_model_pretrained: bool = True
    teacher_proj_dim: Optional[int] = None  # Optional projector for teacher
    teacher_proj_bias: bool = True
    
    # Student (Vision) config - uses existing CLIPVisionCfg
    student_proj_dim: int = 768  # Projector output dimension (should match teacher)
    student_proj_bias: bool = True
    student_proj_layers: int = 2  # Number of MLP layers in projector
    student_hidden_dim: Optional[int] = None  # Hidden dimension for DINO head MLP
    student_bottleneck_dim: Optional[int] = None  # Bottleneck dimension for DINO head
    student_use_bn: bool = False  # Whether to use batch norm in DINO head
    
    # DINO specific params
    teacher_temp: float = 0.04  # Low temperature for teacher (sharp)
    student_temp: float = 0.1   # Higher temperature for student (soft)
    center_momentum: float = 0.9  # EMA momentum for centering
    
    # Training params
    freeze_teacher: bool = True  # Whether to freeze teacher weights
    

class TextDino(nn.Module):
    """TextDino model - uses text as teacher for vision student via DINO-style distillation.
    
    Key features:
    - Text encoder (teacher) provides semantic guidance
    - Vision encoder (student) learns to match text semantics
    - DINO-style centering and sharpening
    - Cross-entropy distillation loss
    """
    output_dict: torch.jit.Final[bool]
    
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        textdino_cfg: TextDinoCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = True,
        **kwargs,  # Accept additional kwargs like textdino, siglip, etc.
    ):
        super().__init__()
        self.output_dict = output_dict
        
        # Convert dict configs to proper config objects
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)
        if isinstance(textdino_cfg, dict):
            textdino_cfg = TextDinoCfg(**textdino_cfg)
        
        self.textdino_cfg = textdino_cfg
        
        # Build vision encoder (student)
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        
        # Build text encoder (teacher)
        if textdino_cfg.teacher_hf_model_name:
            # Check if this is a cached embedding model
            if "cache" in textdino_cfg.teacher_hf_model_name:
                # Use CacheTextEncoder for pre-computed embeddings
                self.text = CacheTextEncoder(
                    proj_type='none',  # We'll use our own projector
                    output_dim=embed_dim,
                    embedding_dim=text_cfg.text_model_embedding_dim,
                    context_length=text_cfg.context_length,
                    vocab_size=text_cfg.vocab_size,
                )
            else:
                # Use HuggingFace model as teacher
                self.text = HFTextEncoder(
                    textdino_cfg.teacher_hf_model_name,
                    output_dim=embed_dim,
                    proj_type='none',  # We'll use our own projector
                    pretrained=textdino_cfg.teacher_hf_model_pretrained,
                    output_tokens=False,
                )
        else:
            # Use standard text transformer
            self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        
        self.context_length = getattr(self.text, 'context_length', text_cfg.context_length)
        self.vocab_size = getattr(self.text, 'vocab_size', text_cfg.vocab_size)
        
        # Student projector (vision -> common space)
        student_input_dim = embed_dim
        student_proj_dim = textdino_cfg.student_proj_dim
        
        # Use DINO head if bottleneck and hidden dimensions are specified
        if textdino_cfg.student_bottleneck_dim is not None and textdino_cfg.student_hidden_dim is not None:
            from .dino_head import DINOHead
            self.student_projector = DINOHead(
                in_dim=student_input_dim,
                out_dim=student_proj_dim,
                use_bn=textdino_cfg.student_use_bn,
                nlayers=textdino_cfg.student_proj_layers,
                hidden_dim=textdino_cfg.student_hidden_dim,
                bottleneck_dim=textdino_cfg.student_bottleneck_dim,
                mlp_bias=textdino_cfg.student_proj_bias,
            )
        elif textdino_cfg.student_proj_layers == 1:
            self.student_projector = nn.Linear(student_input_dim, student_proj_dim, bias=textdino_cfg.student_proj_bias)
        else:
            layers = []
            for i in range(textdino_cfg.student_proj_layers):
                if i == 0:
                    layers.append(nn.Linear(student_input_dim, student_proj_dim, bias=textdino_cfg.student_proj_bias))
                else:
                    layers.append(nn.Linear(student_proj_dim, student_proj_dim, bias=textdino_cfg.student_proj_bias))
                if i < textdino_cfg.student_proj_layers - 1:
                    layers.append(nn.GELU())
            self.student_projector = nn.Sequential(*layers)
        
        # Teacher projector (text -> common space)
        if textdino_cfg.teacher_proj_dim is not None:
            # Use actual text embedding dimension, not embed_dim
            if textdino_cfg.teacher_hf_model_name and "cache" in textdino_cfg.teacher_hf_model_name:
                teacher_input_dim = text_cfg.text_model_embedding_dim  # 2560 for Qwen3
            else:
                teacher_input_dim = embed_dim
            self.teacher_projector = nn.Linear(
                teacher_input_dim, 
                textdino_cfg.teacher_proj_dim, 
                bias=textdino_cfg.teacher_proj_bias
            )
        else:
            self.teacher_projector = None
            
        # Center for DINO (prevents mode collapse)
        self.register_buffer("center", torch.zeros(1, student_proj_dim))
        
        # Freeze teacher if requested
        if textdino_cfg.freeze_teacher:
            for param in self.text.parameters():
                param.requires_grad = False
            if self.teacher_projector is not None:
                for param in self.teacher_projector.parameters():
                    param.requires_grad = False
    
    def update_center(self, teacher_output):
        """Update center with EMA."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.textdino_cfg.center_momentum + \
                      batch_center * (1 - self.textdino_cfg.center_momentum)
    
    @torch.no_grad()
    def encode_text_teacher(self, text, normalize: bool = False):
        """Encode text with teacher (no gradients)."""
        # Get text features
        if hasattr(self.text, 'forward'):
            features = self.text(text)
        else:
            # For CLIP-style text encoder
            features = self.text.encode_text(text)
            
        # Apply teacher projector if exists
        if self.teacher_projector is not None:
            features = self.teacher_projector(features)
            
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_image_student(self, image, normalize: bool = False):
        """Encode image with student (with gradients)."""
        # Vision encoder expects dict input and returns dict output
        vision_output = self.visual(image)
        
        # Extract the actual features from the vision output dict
        if isinstance(vision_output, dict):
            features = vision_output["hidden"]  # The actual feature tensor
        else:
            features = vision_output
            
        features = self.student_projector(features)
        return F.normalize(features, dim=-1) if normalize else features
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        return_all_features: bool = False,
    ):
        """Forward pass for TextDino.
        
        Args:
            image: Image tensor for student
            text: Text tensor for teacher
            return_all_features: If True, return intermediate features
            
        Returns:
            Dictionary with teacher_logits, student_logits, and other info
        """
        assert image is not None and text is not None, "TextDino requires both image and text inputs"
        
        # Get teacher output (no gradients)
        with torch.no_grad():
            teacher_features = self.encode_text_teacher(text, normalize=True)
            # Apply temperature scaling to get teacher logits
            teacher_logits = teacher_features / self.textdino_cfg.teacher_temp
            
            # Update center for DINO
            if self.training:
                self.update_center(teacher_features)
        
        # Get student output (with gradients) 
        student_features = self.encode_image_student(image, normalize=True)
        # Apply temperature scaling to get student logits
        student_logits = student_features / self.textdino_cfg.student_temp
        
        if self.output_dict:
            out_dict = {
                'teacher_logits': teacher_logits,
                'student_logits': student_logits,
                'teacher_temp': self.textdino_cfg.teacher_temp,
                'student_temp': self.textdino_cfg.student_temp,
                'center': self.center,
            }
            
            if return_all_features:
                out_dict['teacher_features'] = teacher_features
                out_dict['student_features'] = student_features
                
            return out_dict
        
        return teacher_logits, student_logits
