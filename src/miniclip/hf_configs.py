# HF architecture dict:
arch_dict = {
    # https://huggingface.co/docs/transformers/en/model_doc/qwen3
    "qwen3": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layers",
            "token_embeddings_attr": "embed_tokens"
        },
        "pooler": "mean_pooler",
        "pad_token_id": 151643 # Qwen3 models use 151643 as pad token id
    },
    # https://huggingface.co/docs/transformers/model_doc/mt5#mt5
    "mt5": {
        "config_names": {
            # unlimited seqlen
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/273
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/t5/modeling_t5.py#L374
            "context_length": "",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "num_heads",
            "layers": "num_layers",
            "layer_attr": "block",
            "token_embeddings_attr": "embed_tokens"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    "bert": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
        },
        "pooler": "cls_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/m2m_100
    "m2m_100": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "encoder_attention_heads",
            "layers": "encoder_layers",
        },
        "pooler": "cls_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/gemma3#gemma3
    "gemma3": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layers",
            "token_embeddings_attr": "embed_tokens"
        },
        "pooler": "mean_pooler",
        "pad_token_id": 0  # Gemma3 models use 0 as pad token id
    },
}
