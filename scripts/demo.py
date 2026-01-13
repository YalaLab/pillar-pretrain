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
    if anatomy == "head_ct":
        cfg = dict(base)
        cfg["anatomy"] = "brain"
        cfg["processing"] = dict(base["processing"])
        cfg["processing"]["crop_pad"] = {"size": [512, 512]}
        cfg["processing"]["resampling"] = {"target_spacing": [0.5, 0.5, 1.25]}
        cfg["processing"]["slice_selection"] = {"enabled": True, "slices": 128}
        return cfg
    if anatomy == "breast_mr":
        cfg = dict(base)
        cfg["anatomy"] = "breast"
        cfg["processing"] = dict(base["processing"])
        cfg["processing"]["crop_pad"] = {"size": [384, 384]}
        cfg["processing"]["resampling"] = {"target_spacing": [1.0, 1.0, 1.0]}
        cfg["processing"]["slice_selection"] = {"enabled": True, "slices": 192}
        return cfg
    return build_inline_rave_config("chest_ct")

def build_video_hevc_exporter_config(modality: str) -> dict:
    """
    Build an inline exporter configuration for HEVC (H.265) video codec.
    """
    if modality == "CT":
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
    elif modality == "MR":
        return {
            "compression": "video",
            "video": {
                "codec": "libx265",
                "bit_depth": 10,
                "crf": 6,
                "gop_size": 128,
                "hu_min": 0,
                "hu_max": 65535,
                "preset": "ultrafast",
                "archive": False,
                "lossless": False,
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
    else: 
        raise ValueError(f"Unsupported modality: {modality}")

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
    exporter_cfg = build_video_hevc_exporter_config(modality=anatomy.split("_")[1].upper())
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
    "breast_mr": "YalaLab/Pillar0-BreastMRI",
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
        embeddings = {"sample_name": [], "output_path": [], "embedding": []}

        inputs = pd.read_csv(inputs_csv_path)
        mapping_csv = preprocess_inputs(
            input_csv_path=inputs_csv_path,
            anatomy=self.anatomy,
            output_dir="rve-output",
            workers=4,
        )
        processed = pd.read_csv(mapping_csv)
        inputs = inputs.merge(processed, left_on="series_path", right_on="source_path")
        if "sample_name" in inputs and self.anatomy == "breast_mr":
            inputs = inputs[["sample_name", "series", "output_path"]].groupby("sample_name").agg(list).reset_index()[["sample_name", "series", "output_path"]]
            inputs['output_path'] = inputs.apply(lambda x: {pair[0]: pair[1] for pair in zip(x['series'], x['output_path'])}, axis=1)
        progress_bar = tqdm(inputs.iterrows(), total=len(inputs), desc="Generating Embeddings")

        batch = {"anatomy": [self.anatomy]}

        for row in progress_bar:
            if len(row) == 2:
                row = row[1]
            embeddings["sample_name"].append(row.get('sample_name', None))
            embeddings["output_path"].append(row.get('output_path', None))

            if "ct" in self.anatomy.lower():
                processed_series = rve.load_sample(row['output_path'],  use_hardware_acceleration=False)
                processed_series = processed_series.unsqueeze(0)
            elif self.anatomy == "breast_mr":
                processed_series = []
                for serie in ["T1FS", "T2FS", "Ph2"]:
                    processed_series.append(rve.load_sample(row['output_path'][serie],  use_hardware_acceleration=False))
                processed_series = torch.stack(processed_series, dim=0)
                print(processed_series.shape)
            else:
                raise ValueError(f"Unsupported modality: {self.anatomy}")

            _, D, H, W = processed_series.shape
            if H > self.target_h:
                crop_side = (H - self.target_h) // 2
                processed_series = processed_series[:, :, crop_side:-crop_side, crop_side:-crop_side]
            if D < self.target_d:
                pad_total = self.target_d - D
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left  # Handles odd padding amounts
                processed_series = F.pad(processed_series, (0, 0, 0, 0, pad_left, pad_right, 0, 0))

            if "ct" in self.anatomy.lower():
                x = rve.apply_windowing(processed_series[0], "all", "CT").unsqueeze(0)
            elif self.anatomy == "breast_mr":
                x = torch.zeros((1, 3, D, H, W), device=processed_series.device, dtype=torch.float32)
                for i in range(3):
                    x[:, i] = rve.apply_windowing(processed_series[i], "high_contrast", "MR").to(device=x.device, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported modality: {self.anatomy}")

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


