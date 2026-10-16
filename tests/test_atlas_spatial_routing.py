"""Where Atlas's cross-scale messages land, traced with coordinate-valued tokens.

Each test pushes tokens that carry their own grid coordinates through the real
MultiScaleAttentionBlock methods (weights do not change where a message goes) and checks that
every message reaches the token or window that covers the same place:

  bottom-up    a fine window's max is added to the mid token for that window
  top-down     a fine window attends to the mid window that contains it
  cross-attn   a mid token cross-attends to the fine window it summarises

The token grids are those of the four released Pillar-0 models.
"""
import math
from types import MethodType, SimpleNamespace

import pytest
import torch
from einops import rearrange

from miniclip.multimodal_atlas import MultiModalAtlas
from miniclip.multimodal_msa import MultiScaleAttentionBlock, parent_window_index

MERGE, L2G = [4, 4, 4], [4, 4, 4]
CONFIGS = {  # image_size, patch_size
    "chest_ct": ([256, 256, 256], [8, 8, 4]),
    "abdomen_ct": ([384, 384, 384], [6, 6, 6]),
    "head_ct": ([256, 256, 128], [8, 8, 4]),
    "breast_mr": ([384, 384, 192], [12, 12, 6]),
}


def coords(shape):
    return torch.stack(torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij")).float()[None]


def windowed(x, merge=MERGE):
    """(1, C, A0, A1, A2) -> tokens as _build_multiscale_tokens writes them."""
    shape = list(x.shape[2:])
    if shape == list(merge) or any(s < m for s, m in zip(shape, merge)):
        return rearrange(x, "b c d h w -> b (d h w) c")
    return rearrange(x, "b c (d m0) (h m1) (w m2) -> (b d h w) (m0 m1 m2) c", m0=merge[0], m1=merge[1], m2=merge[2])


def unwindowed(t, shape, merge=MERGE):
    if t.shape[0] == 1 and t.shape[1] == math.prod(shape):
        return rearrange(t, "b (d h w) c -> b c d h w", d=shape[0], h=shape[1], w=shape[2])[0]
    g = [s // m for s, m in zip(shape, merge)]
    return rearrange(t, "(b d h w) (m0 m1 m2) c -> b c (d m0) (h m1) (w m2)", b=1, d=g[0], h=g[1], w=g[2],
                     m0=merge[0], m1=merge[1], m2=merge[2])[0]


def setup(name, spatial):
    image_size, patch_size = CONFIGS[name]
    atlas = SimpleNamespace(spatial_routing=spatial)
    layout = MultiModalAtlas.prepare_multiscale_layout(atlas, image_size, MERGE, L2G, patch_size)
    tokens = [s // p for s, p in zip(image_size, patch_size)]
    mid = [t // m for t, m in zip(tokens, MERGE)]
    block = SimpleNamespace(spatial_routing=spatial)
    block._one2one_spatial = MethodType(MultiScaleAttentionBlock._one2one_spatial, block)
    return block, layout, tokens, mid


def bottom_up_share(name, spatial):
    block, layout, tokens, mid = setup(name, spatial)
    fine = windowed(coords(tokens) // 4 + 1)                       # each fine token: its window's coords + 1
    zeros = torch.zeros_like(windowed(coords(mid)))
    out = MultiScaleAttentionBlock.propagate_bottom_up(
        block, [fine, zeros], [l["grid_size"] for l in layout[:2]], MERGE, L2G)
    landed = unwindowed(out[1], mid) - 1                            # at each mid cell: which window landed
    return (landed == coords(mid)[0]).all(0).float().mean().item()


def top_down_share(name, spatial):
    block, layout, tokens, mid = setup(name, spatial)
    grids = [l["grid_size"] for l in layout]
    x_fine = windowed(coords(tokens))
    x_mid = windowed(coords(mid) // 4)                              # each mid token: its mid window's coords
    n_own = x_fine.shape[1]
    stub = SimpleNamespace(
        xattn_qkv=lambda q, k, v: k[:, 0, n_own:].mean(1, keepdim=True).expand(-1, q.shape[2], -1),
        skip_with_drop=lambda x, skip: skip, mlp_residual=lambda x: x)
    block.out_scales = {1: {"tokens": x_mid}}
    block._grid_sizes = grids
    block.blocks = {0: stub}
    block.get_qkv = lambda x, S, keys=("q", "kv"): ([x[:, None]] * 3)[: (3 if list(keys) == ["q", "kv"] else 2)]
    read = MultiScaleAttentionBlock._process__sequential__all2all_sattn(block, x_fine, 0)[:, 0]  # per fine window
    fine_windows = rearrange(coords(grids[0]), "b c d h w -> (b d h w) c")
    return (read == fine_windows // 4).all(1).float().mean().item()


def cross_attn_share(name, spatial):
    block, layout, tokens, mid = setup(name, spatial)
    fine = windowed(coords(tokens) // 4)                           # each fine token: its window's coords
    x_mid = windowed(coords(mid))                                   # each mid token: its own cell
    stub = SimpleNamespace(
        xattn_qkv=lambda q, k, v: k[:, 0].mean(1, keepdim=True).expand(-1, q.shape[2], -1),
        skip_with_drop=lambda x, skip: skip, mlp_residual=lambda x: x)
    block.out_scales = {1: {"tokens": x_mid}, 0: {"tokens": fine}}
    block.blocks = {1: stub}
    block.get_qkv = lambda x, S, keys: [x[:, None]] if list(keys) == ["q"] else [x[:, None], x[:, None]]
    got = unwindowed(MultiScaleAttentionBlock._aggregate_one2one_xattn(block, 1, multiscale_layout=layout), mid)
    return (got == coords(mid)[0]).all(0).float().mean().item()


@pytest.mark.parametrize("name", CONFIGS)
def test_spatial_routing_sends_every_message_to_the_same_place(name):
    assert bottom_up_share(name, True) == 1.0
    assert top_down_share(name, True) == 1.0
    assert cross_attn_share(name, True) == 1.0


def test_legacy_routing_is_unchanged():
    """The default keeps the routing the released checkpoints were trained with."""
    assert bottom_up_share("chest_ct", False) == 0.0625
    assert top_down_share("chest_ct", False) == 0.125
    assert cross_attn_share("chest_ct", False) == 0.0625


@pytest.mark.parametrize("name", CONFIGS)
def test_spatial_layout_matches_legacy_numbers(name):
    """With uniform merge ratios the per-axis layout has the same sizes as the legacy one."""
    image_size, patch_size = CONFIGS[name]
    legacy = MultiModalAtlas.prepare_multiscale_layout(SimpleNamespace(spatial_routing=False), image_size, MERGE, L2G, patch_size)
    spatial = MultiModalAtlas.prepare_multiscale_layout(SimpleNamespace(spatial_routing=True), image_size, MERGE, L2G, patch_size)
    assert legacy == spatial


@pytest.mark.parametrize("name", CONFIGS)
def test_tokenizer_and_layout_agree(name):
    image_size, patch_size = CONFIGS[name]
    atlas = SimpleNamespace(spatial_routing=True)
    tokens = [s // p for s, p in zip(image_size, patch_size)]
    _, grids = MultiModalAtlas._build_multiscale_tokens(atlas, torch.zeros(1, 1, *tokens), MERGE, L2G)
    layout = MultiModalAtlas.prepare_multiscale_layout(atlas, image_size, MERGE, L2G, patch_size)
    assert [list(g) for g in grids] == [l["grid_size"] for l in layout]


def test_parent_window_index():
    fine, coarse = [8, 8, 16], [2, 2, 4]
    idx = parent_window_index(fine, coarse, batch_size=2)
    for b in range(2):
        for i in range(math.prod(fine)):
            a, r = divmod(i, fine[1] * fine[2])
            h, w = divmod(r, fine[2])
            want = b * 16 + ((a // 4) * coarse[1] + h // 4) * coarse[2] + w // 4
            assert idx[b * math.prod(fine) + i].item() == want


@pytest.mark.parametrize("spatial", [False, True])
def test_small_chest_model_runs(spatial):
    config = {
        # this repo's ChestCTEmbed3D downsamples 6 x 6 x 6; a non-cubic volume gives a 16 x 16 x 32 grid
        "modalities": {"chest_ct": {"image_size": [96, 96, 192], "in_chans": 11, "local2global": L2G,
                                    "merge_ratio": MERGE, "num_features": 12, "patch_size": [6, 6, 6]}},
        "model": {"atlas_args": {"dim": 24, "num_heads": 2, "mlp_ratio": 2.0, "qkv_bias": True,
                                 "spatial_routing": spatial}, "stages": [1, 1, 1]},
    }
    model = MultiModalAtlas(args=None, embed_dim=24, model_config=config).eval()
    with torch.no_grad():
        out = model({"chest_ct": torch.rand(2, 11, 96, 96, 192)})
    assert out.shape == (2, 72)
