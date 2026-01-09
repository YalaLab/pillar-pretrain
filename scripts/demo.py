import os
import warnings
from collections import OrderedDict
import subprocess
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import yaml
import pandas as pd
import rve

from tqdm import tqdm

from transformers import AutoModel

def build_inline_rave_config(anatomy: str) -> dict:
    """
    Build an inline vision-engine config equivalent to the RAVE YAMLs,
    so we don't need external RAVE config files.
    """
    base = {
        "modality": "CT",
        "processing": {
            "crop_pad": {"size": [256, 256]},
            "resampling": {"target_spacing": [1.25, 1.25, 1.25]},
            "conversion_backend": "sitk",
            "slice_selection": {"enabled": True, "slices": 256},
        },
        "exporter_config": "video_hevc",
        "logging": {"level": "INFO", "file": None},
    }
    if anatomy == "chest_ct":
        cfg = dict(base)
        cfg["anatomy"] = "chest"
        cfg["processing"] = dict(base["processing"])
        cfg["processing"]["crop_pad"] = {"size": [256, 256]}
        cfg["processing"]["slice_selection"] = {"enabled": True, "slices": 256}
        return cfg
    if anatomy == "abdomen_ct":
        cfg = dict(base)
        cfg["anatomy"] = "abdomen"
        cfg["processing"] = dict(base["processing"])
        cfg["processing"]["crop_pad"] = {"size": [384, 384]}
        cfg["processing"]["slice_selection"] = {"enabled": True, "slices": 384}
        cfg["exporter_config"] = "video_hevc"
        return cfg
    if anatomy == "head_ct" or anatomy == "brain_ct":
        cfg = dict(base)
        cfg["anatomy"] = "brain"
        cfg["processing"] = dict(base["processing"])
        cfg["processing"]["crop_pad"] = {"size": [512, 512]}
        cfg["processing"]["resampling"] = {"target_spacing": [0.5, 0.5, 1.25]}
        cfg["processing"]["slice_selection"] = {"enabled": True, "slices": 128}
        return cfg
    return build_inline_rave_config("chest_ct")

def build_video_hevc_exporter_config() -> dict:
    """
    Build an inline exporter configuration for HEVC (H.265) video codec.
    """
    return {
        "compression": "video",
        "video": {
            "codec": "libx265",
            "bit_depth": 10,
            "crf": 6,
            "gop_size": 128,
            "hu_min": -1024,
            "hu_max": 3071,
            "preset": "ultrafast",
            "archive": False,
        },
        "parallel": {
            "workers": 32,
        },
        "logging": {
            "level": "INFO",
        },
        "output": {
            "extension": ".mp4",
            "overwrite": False,
        },
    }

def run_vision_engine_process(
    config_path: str,
    input_series_csv: str,
    output_dir: str = "rve-output",
    workers: int = 4,
    extra_args: Optional[List[str]] = None,
) -> None:
    cmd = [
        "vision-engine",
        "process",
        "--config", config_path,
        "--input-series-csv", input_series_csv,
        "--output", output_dir,
        "--workers", str(workers),
    ]
    if extra_args:
        cmd += list(extra_args)
    subprocess.run(cmd, check=True)

def _write_config_to_file(config: dict, output_dir: str) -> str:
    """
    Write the inline config dict to a YAML file inside output_dir
    and return its path.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, "inline_rave_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)
    return cfg_path

def _write_exporter_config_to_file(config: dict, output_dir: str) -> str:
    """
    Write the inline exporter config to a YAML file and return its path.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, "inline_exporter_video_hevc.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)
    return cfg_path

def preprocess_inputs(
    input_csv_path: str,
    anatomy: str,
    output_dir: str = "rve-output",
    workers: int = 4,
    extra_args: Optional[List[str]] = None,
) -> str:
    # Build exporter config and write to file
    exporter_cfg = build_video_hevc_exporter_config()
    exporter_cfg_path = _write_exporter_config_to_file(exporter_cfg, output_dir)
    # Build main inline config and reference exporter path
    inline_cfg = build_inline_rave_config(anatomy)
    inline_cfg["exporter_config"] = exporter_cfg_path
    config_path = _write_config_to_file(inline_cfg, output_dir)
    run_vision_engine_process(
        config_path=config_path,
        input_series_csv=input_csv_path,
        output_dir=output_dir,
        workers=workers,
        extra_args=extra_args,
    )
    return os.path.join(output_dir, "mapping.csv")

anatomy_mapping = {
    "chest_ct": "YalaLab/Pillar0-ChestCT",
    "abdomen_ct": "YalaLab/Pillar0-AbdomenCT",
    "head_ct": "YalaLab/Pillar0-HeadCT",
}

class Pillar:
    def __init__(self,
        anatomy="chest_ct",
        model_revision="main",
        local_dir="logs/checkpoints",
        **kwargs
    ):
        self.anatomy = anatomy
        self.model_repo_id = anatomy_mapping[anatomy]
        self.model_revision = kwargs.pop("model_revision", "main")
        # Keep remaining kwargs to build the underlying model architecture
        self._base_model_kwargs = dict(kwargs)

        self.model = AutoModel.from_pretrained(self.model_repo_id, revision=self.model_revision, trust_remote_code=True)

        # Load target dimensions from inline config (formerly from RAVE YAML)
        inline_cfg = build_inline_rave_config(self.anatomy)
        processing_cfg = (inline_cfg or {}).get("processing", {})
        crop_pad_cfg = (processing_cfg or {}).get("crop_pad", {})
        slice_sel_cfg = (processing_cfg or {}).get("slice_selection", {})
        size_hw = crop_pad_cfg.get("size", [256, 256])
        self.target_h = int(size_hw[0]) if isinstance(size_hw, (list, tuple)) and len(size_hw) == 2 else 256
        self.target_w = int(size_hw[1]) if isinstance(size_hw, (list, tuple)) and len(size_hw) == 2 else 256
        self.target_d = int(slice_sel_cfg.get("slices", 256))


    def predict(self, inputs_csv_path=None, **extras):
        embeddings = {"input": [], "embedding": []}

        inputs = pd.read_csv(inputs_csv_path)
        mapping_csv = preprocess_inputs(
            input_csv_path=inputs_csv_path,
            anatomy=self.anatomy,
            output_dir="rve-output",
            workers=4,
        )
        processed = pd.read_csv(mapping_csv)
        inputs = inputs.merge(processed, left_on="series_path", right_on="source_path")
        progress_bar = tqdm(inputs.iterrows(), total=len(inputs), desc="Generating Embeddings")

        batch = {"anatomy": [self.anatomy]}

        for row in progress_bar:
            if len(row) == 2:
                row = row[1]
            embeddings["input"].append(row.get('series_path', None))

            processed_series = rve.load_sample(row['output_path'],  use_hardware_acceleration=False)

            D, H, W = processed_series.shape
            # Center-crop or pad depth (D) to target_d
            if D > self.target_d:
                crop_front = (D - self.target_d) // 2
                crop_back = D - self.target_d - crop_front
                processed_series = processed_series[crop_front:D - crop_back, :, :]
            elif D < self.target_d:
                pad_total = self.target_d - D
                pad_front = pad_total // 2
                pad_back = pad_total - pad_front
                processed_series = F.pad(processed_series, (0, 0, 0, 0, pad_front, pad_back))
            # Update dims after D adjustment
            _, H, W = processed_series.shape
            # Center-crop or pad height (H) to target_h
            if H > self.target_h:
                crop_top = (H - self.target_h) // 2
                crop_bottom = H - self.target_h - crop_top
                processed_series = processed_series[:, crop_top:H - crop_bottom, :]
            elif H < self.target_h:
                pad_total_h = self.target_h - H
                pad_top = pad_total_h // 2
                pad_bottom = pad_total_h - pad_top
                processed_series = F.pad(processed_series, (0, 0, pad_top, pad_bottom, 0, 0))
            # Update dims after H adjustment
            _, _, W = processed_series.shape
            # Center-crop or pad width (W) to target_w
            if W > self.target_w:
                crop_left = (W - self.target_w) // 2
                crop_right = W - self.target_w - crop_left
                processed_series = processed_series[:, :, crop_left:W - crop_right]
            elif W < self.target_w:
                pad_total_w = self.target_w - W
                pad_left = pad_total_w // 2
                pad_right = pad_total_w - pad_left
                processed_series = F.pad(processed_series, (pad_left, pad_right, 0, 0, 0, 0))

            x = rve.apply_windowing(processed_series, "all", "CT").unsqueeze(0)
            with torch.no_grad():
                image = torch.as_tensor(x)
                x_dict = {self.anatomy: image}
                embeddings["embedding"].append(self.model.forward(x_dict, batch=batch, **extras)[0])
                
        return embeddings

if __name__ == "__main__":
    # Minimal CLI to run preprocessing without hardcoded values.
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess inputs with vision-engine using RAVE configs.")
    parser.add_argument("--input-csv", required=True, help="Path to input series CSV.")
    parser.add_argument("--anatomy", default="chest_ct", choices=["chest_ct", "abdomen_ct", "head_ct"], help="Anatomy to preprocess.")
    parser.add_argument("--output-dir", default="rve-output", help="Output directory for vision-engine.")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for vision-engine.")
    args, unknown = parser.parse_known_args()

    preprocess_inputs(
        input_csv_path=args.input_csv,
        anatomy=args.anatomy,
        output_dir=args.output_dir,
        workers=args.workers,
        extra_args=unknown if unknown else None,
    )


