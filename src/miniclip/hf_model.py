""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType
from .cached_text_embedding import CachedTextEmbedding

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# Additional poolers for GPT-like models can be registered dynamically via register_pooler.
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]


class CacheTextEncoder(nn.Module):
    """Cache text encoder"""
    def __init__(self, proj_type: str = "mlp", output_dim: int = 512, embedding_dim: int = 512, context_length: int = 512, vocab_size: int = 49408):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj_type = proj_type
        self.output_dim = output_dim
        self.context_length = context_length
        self.vocab_size = vocab_size

        if self.proj_type == 'linear':
            self.proj = nn.Linear(self.embedding_dim, self.output_dim, bias=False)
        elif self.proj_type == 'mlp':
            hidden_size = (self.embedding_dim + self.output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, self.output_dim, bias=False),
            )
        else:
            self.proj = nn.Identity()   
    
    def forward(self, x: TensorType):
        return self.proj(x)


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj_type: str = None,
            pretrained: bool = True,
            output_tokens: bool = False,
            output_token_dim: int = None,
            device_map: str = None,  # Add device_map support for tensor parallelism
            torch_dtype: torch.dtype = None,  # Add dtype support
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # Infer whether we should ask the HF model to include its own pooling layer.
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (AutoModel.from_pretrained, {"pretrained_model_name_or_path": model_name_or_path}) if pretrained else (
                AutoModel.from_config, {"config": self.config})
            
            # Add device_map and torch_dtype to model_args if provided
            if device_map is not None:
                model_args["device_map"] = device_map
            if torch_dtype is not None:
                model_args["torch_dtype"] = torch_dtype
            
            # PretrainedConfig exposes is_encoder_decoder on the text encoders we support.
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(**model_args)
                self.transformer = self.transformer.encoder
            else:
                ## check if the function supports the pooling layer
                if hasattr(create_func, "add_pooling_layer"):
                    self.transformer = create_func(**model_args, add_pooling_layer=uses_transformer_pooler)
                else:
                    self.transformer = create_func(**model_args)
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
        if pooler_type is None:  # get default arch pooler
            pooler_type = (arch_dict[self.config.model_type]["pooler"])

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        # Handle MedGemma and other models with text_config
        if hasattr(self.config, 'text_config'):
            # For models like MedGemma that have a text_config attribute
            config_for_attrs = self.config.text_config
        else:
            # For regular models
            config_for_attrs = self.config
            
        self.vocab_size = getattr(config_for_attrs, 'vocab_size', 0)
        self.context_length = getattr(config_for_attrs, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        # Handle MedGemma and other models with text_config
        if hasattr(self.config, 'text_config'):
            # For models like MedGemma that have a text_config attribute
            config_for_width = self.config.text_config
        else:
            # For regular models
            config_for_width = self.config
            
        d_model = getattr(config_for_width, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )
        
        # Move projection layer to the same device and dtype as the transformer
        if hasattr(self.transformer, 'hf_device_map'):
            # For tensor parallelism, put projection on the first GPU with bfloat16
            self.proj = self.proj.to("cuda:0").to(torch.bfloat16)
        elif hasattr(self.transformer, 'device'):
            # For single GPU, put projection on the same device and dtype
            self.proj = self.proj.to(self.transformer.device)
            if torch_dtype is not None:
                self.proj = self.proj.to(torch_dtype)
        
        if output_tokens:
            if output_token_dim != None:
                self.token_proj = nn.Linear(d_model, output_token_dim, bias=False)
            else:
                self.token_proj = nn.Identity()


    def forward(self, x: TensorType):
        if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
            pad_token_id = self.config.pad_token_id
        else:
            pad_token_id = arch_dict[self.config.model_type]["pad_token_id"] \
                if "pad_token_id" in arch_dict[self.config.model_type] else None
        if pad_token_id is None:
            raise ValueError("pad_token_id is not defined in the model config. "
                             "Please set it to a valid value (e.g., 0 for BERT-like models).")

        attn_mask = (x != pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :] 
            if type(self.pooler) == ClsPooler 
            else out.last_hidden_state
        )
        
        if self.output_tokens:
            tokens = self.token_proj(tokens)
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
