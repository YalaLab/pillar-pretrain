import logging
import math

import torch
import torch.nn as nn
import yaml
from einops import rearrange

from .atlas_encoders import ConvCXREmbed, MultiConvEmbed3D, MultiViewConvCXR, AbdomenCTEmbed3D, BMRMultiConvEmbed3D, BrainCTMultiConvEmbed3D, ProstateMRMultiConvEmbed3D, ChestCTEmbed3D
from .multimodal_msa import AtlasStage

logger = logging.getLogger(__name__)


class MultiModalAtlas(nn.Module):
    def __init__(
        self,
        args,
        kwargs_list=None,
        embed_dim=384,
        num_classes=100,
        fc_norm=nn.LayerNorm,
        multiscale_feats=True,
        model_config={},
        modalities={"single_view": {"num_views": 1, "view_pos_embed": None}},
        **kwargs,
    ):
        """
        Args:
            args: full argument config
            kwargs_list : generate different scales
            embed_dim : embedding dimension
            num_classes : number of classes
            fc_norm : normalization layer
            multiscale_feats : whether to use multiscale features
            modalities : dictionary of modalities
                - single_view_2d
                    - num_views : number of views
                    - view_pos_embed : view-specific positional embeddings
                - multi_view_2d
                    - num_views : number of views
                    - view_pos_embed : view-specific positional embeddings
                    - view_pos_embed_type : type of view-specific positional embeddings
                - single_view_3d
                    - num_views
        """

        super().__init__()
        # assert kwargs_list is not None, "kwargs_list should be provided"
        self.args = args
        # self.kwargs_list = kwargs_list
        self.num_classes = num_classes
        self.image_size = 256
        self.multiscale_feats = multiscale_feats
        self.embed_dim = embed_dim
        self.model_config = model_config
        assert self.model_config is not None, "model_config should be provided"

        self._init_patch_embeds()
        self._init_atlas_stages()
        self._init_pools()

    def build_scales(self, x, batch=None):
        """
        Expects input to be dictionary of tensors,
        """
        # breakpoint()
        modality, image = list(x.items())[0]
        modal_config = self.model_config["modalities"]

        patch_embed_fn = self.patch_embeds[modality]
        patch_embed_xBCDHW = patch_embed_fn(image)
        multiscale_feats, grid_sizes = self._build_multiscale_tokens(
            patch_embed_xBCDHW,
            merge_ratio=modal_config[modality]["merge_ratio"],
            local2global=modal_config[modality]["local2global"],
        )
        return multiscale_feats, grid_sizes

    def _build_multiscale_tokens(self, x_BCDHW, merge_ratio, local2global):
        seqlen = x_BCDHW.shape[2] * x_BCDHW.shape[3] * x_BCDHW.shape[4]
        downsampling_ratio = local2global[0] * local2global[1] * local2global[2]
        kernel_size = (1, 1, 1)

        min_seqlen = merge_ratio[0] * merge_ratio[1] * merge_ratio[2]

        num_stages = (
            math.ceil((math.log2(seqlen / min_seqlen) / math.log2(downsampling_ratio)))
            + 1
        )

        stages = []
        grid_sizes = []

        for scale in range(num_stages):
            local2global_op = nn.MaxPool3d(kernel_size=kernel_size)
            x_scale_BCDHW = local2global_op(x_BCDHW)
            kernel_size = tuple(k * l for k, l in zip(kernel_size, local2global))

            # check if windowing is possible
            b, c, d, h, w = x_scale_BCDHW.shape

            ## if exact merge is possible, then dont add another scale
            if d == merge_ratio[0] and h == merge_ratio[1] and w == merge_ratio[2]:
                x_scale_win = rearrange(x_scale_BCDHW, "b c d h w -> b (d h w) c")
                grid_size = [1, 1, 1]
            ## only reduce in dimensions where merge is possible
            elif d >= merge_ratio[0] and h >= merge_ratio[1] and w >= merge_ratio[2]:
                # run windowing
                x_scale_win = rearrange(
                    x_scale_BCDHW,
                    "b c (d m0) (h m1) (w m2) -> b d h w m0 m1 m2 c",
                    m0=merge_ratio[0],
                    m1=merge_ratio[1],
                    m2=merge_ratio[2],
                )
                grid_size = [
                    x_scale_win.shape[2],
                    x_scale_win.shape[3],
                    x_scale_win.shape[1],
                ]
                x_scale_win = rearrange(
                    x_scale_win, "b d h w m0 m1 m2 c -> (b d h w) (m0 m1 m2) c"
                )
            else:
                x_scale_win = rearrange(x_scale_BCDHW, "b c d h w -> b (d h w) c")
                grid_size = [1, 1, 1]

            stages.append(x_scale_win)
            grid_sizes.append(grid_size)

            if math.prod(grid_size) == 1:
                break

        return stages, grid_sizes

    def prepare_multiscale_layout(
        self, img_size, merge_ratio, local2global, patch_size
    ):
        """
        given the input size, merge_ratio and local2global
        prepare the layout for multiscale attention and config
        for the architecture
        """
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
                grid_size = [h0 // mH, w0 // mW, d0 // mD]
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
        return multiscale_layout

    def forward(self, x, batch=None):
        # breakpoint()
        modality, image = list(x.items())[0]
        bsz = image.shape[0]
        logger.debug(
            f"Entering MultiModalAtlas forward with modality: {modality}, batch size: {bsz}"
        )
        x, grid_sizes = self.build_scales(x, batch)

        modal_config = self.model_config["modalities"][modality]
        image_size = modal_config["image_size"]
        merge_ratio = modal_config["merge_ratio"]
        local2global = modal_config["local2global"]
        patch_size = modal_config["patch_size"]
        multiscale_layout = self.prepare_multiscale_layout(
            img_size=image_size,
            merge_ratio=merge_ratio,
            local2global=local2global,
            patch_size=patch_size,
        )
        grid_sizes = [layout["grid_size"] for layout in multiscale_layout]

        multiscale_feats = []
        for level, atlas_model in enumerate(self.atlas_models):
            is_last = len(x) == 1 or level == len(self.atlas_models) - 1
            x = atlas_model(
                x,
                grid_sizes=grid_sizes[level:],
                multiscale_layout=multiscale_layout[level:],
                merge_ratio=merge_ratio,
                local2global=local2global,
                modality=modality,
            )
            if not is_last:
                multiscale_feats.append(x[0])
                x = x[1:]

        multiscale_feats.append(x[0])

        feats_for_head = []
        for i, scale_tokens in enumerate(multiscale_feats):
            # Need to know if scale_tokens is windowed (B*NW, K, C) or flattened (B, N, C)
            # Assuming it's (B*NW, K, C) if NW > 1 based on layout
            if (
                math.prod(grid_sizes[i]) > 1 and len(scale_tokens.shape) == 3
            ):  # Check if likely windowed format
                rearranged_scale = rearrange(
                    scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz
                )
                logger.debug(
                    f"  Rearranging scale {i} (windowed) {scale_tokens.shape} -> {rearranged_scale.shape}"
                )
                feats_for_head.append(rearranged_scale)
            elif (
                len(scale_tokens.shape) == 3 and scale_tokens.shape[0] == bsz
            ):  # Already (B, N, C)
                logger.debug(
                    f"  Using scale {i} (already BNC) shape: {scale_tokens.shape}"
                )
                feats_for_head.append(scale_tokens)
            else:
                logger.warning(
                    f"  Unexpected shape for scale {i}: {scale_tokens.shape}. Attempting BNC rearrange."
                )
                # Try a generic rearrange, might fail if batch dim isn't divisible
                try:
                    rearranged_scale = rearrange(
                        scale_tokens, "(b nw) k c -> b (nw k) c", b=bsz
                    )
                    feats_for_head.append(rearranged_scale)
                except Exception as e:
                    logger.error(
                        f"    Failed to rearrange scale {i}: {e}. Skipping this scale for readout."
                    )
                    continue

        if self.multiscale_feats:
            x = []
            for i, scale in enumerate(feats_for_head):
                feats = self.maxpool(scale.transpose(1, 2)).squeeze(2)
                x.append(feats)
            x = torch.cat(x, dim=1)
        else:
            ## return maxpool on last scale features only
            x = feats_for_head[0]
            x = self.maxpool(x.transpose(1, 2)).squeeze(2)
        return x

    def _init_patch_embeds(self):

        modalities = self.model_config["modalities"]
        MODALITY_TO_PATCH_EMBED = {
            "chest_xray_single_view": ConvCXREmbed,
            "chest_xray_two_view": MultiViewConvCXR,
            "chest_ct": ChestCTEmbed3D,
            "abdomen_ct": ChestCTEmbed3D,
            "breast_mr": BMRMultiConvEmbed3D,
            "brain_ct": BrainCTMultiConvEmbed3D,
            "prostate_mr": ProstateMRMultiConvEmbed3D,
        }

        self.patch_embeds = nn.ModuleDict()
        for modality in modalities:
            
            image_size = modalities[modality].get("image_size", self.image_size)
            patch_size = modalities[modality].get("patch_size", 16)
            in_chans = modalities[modality].get("in_chans", 1)
            num_features = modalities[modality].get("num_features", 192)

            logging.info(f"Initializing patch embed for modality: {modality}, image_size: {image_size}, patch_size: {patch_size}, in_chans: {in_chans}, num_features: {num_features}")

            self.patch_embeds[modality] = MODALITY_TO_PATCH_EMBED[modality](
                vol_size=image_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=num_features,
            )

    def _init_atlas_stages(self):
        atlas_models = []
        atlas_args = self.model_config["model"]["atlas_args"]
        num_scales = len(self.model_config["model"]["stages"])
        for stage_idx, depth in enumerate(self.model_config["model"]["stages"]):
            atlas_args["depth"] = depth
            atlas_args["num_scales"] = num_scales
            atlas_models.append(AtlasStage(**atlas_args))

        self.atlas_models = nn.ModuleList(atlas_models)

    def _init_pools(self):
        self.maxpool = nn.AdaptiveMaxPool1d(1)
