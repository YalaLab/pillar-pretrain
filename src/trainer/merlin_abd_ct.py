"""Merlin ABD CT dataset for miniclip trainer with cache support.

This dataset supports:
- Loading from RVE torch tensor exports (optimized for deep learning)
- Multiple loading backends: RVE tar files, cached directory structure (sample/volume.pt), original .nii.gz
- Configuration for CT windowing (applied on GPU after collation)

The dataset returns raw CT values without windowing or normalization.
These transforms are applied on GPU after collation for better performance.
The dataset keeps data on CPU and lets the DataLoader handle device placement.
This allows proper use of pin_memory for efficient GPU transfers during training.
"""

import os
import hashlib
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Any, Optional, List, Dict, Union
from torch.utils.data import Dataset
import torchvision
from skimage.transform import resize
from miniclip.hf_cache_utils import create_embedder_from_hub

# Import RVE for loading tar files
try:
    import rve
    HAS_RVE = True
except ImportError:
    HAS_RVE = False
    print("Warning: rad-vision-engine (rve) not installed. RVE loading will be disabled.")


def load_nii_volume(filepath: str, target_shape=(256, 256, 192)) -> torch.Tensor:
    """
    Load a 3D volume from a .nii.gz file and resize to target_shape (D, H, W).
    """
    img = nib.load(filepath)
    volume = img.get_fdata()
    # Normalize to [0, 1] if needed (optional, depends on downstream use)
    volume = np.clip(volume, 0, np.percentile(volume, 99))
    volume = volume / np.max(volume) if np.max(volume) > 0 else volume
    # Resize to (256, 256, 192)
    volume_resized = resize(volume, target_shape, order=1, mode='constant', cval=0, anti_aliasing=True)
    volume_resized = torch.from_numpy(volume_resized).float()
    # Add channel dimension: (D, H, W) -> (1, D, H, W)
    if volume_resized.dim() == 3:
        volume_resized = volume_resized.unsqueeze(0)
    return volume_resized


def load_cached_volume(filepath: str) -> torch.Tensor:
    """
    Load a preprocessed volume from a cached .pt file.
    
    Args:
        filepath: Path to the cached .pt file
    
    Returns:
        Volume tensor on CPU
    """
    try:
        volume = torch.load(filepath, map_location='cpu')
        return volume
    except Exception as e:
        print(f"Failed to load cached volume from {filepath}: {e}")
        raise


def load_rve_volume(filepath: str, 
                   target_slices: Optional[int] = None,
                   target_h: Optional[int] = None,
                   target_w: Optional[int] = None,
                   pad_value: float = -1024.0,
                   crop_mode: str = 'center') -> torch.Tensor:
    """
    Load a 3D volume from an RVE tar file with optional cropping/padding to target dimensions.
    
    Args:
        filepath: Path to the .tar file from RVE
        target_slices: Target number of slices (D dimension). If provided, will pad or crop.
        target_h: Target height (H dimension). If provided, will pad or crop.
        target_w: Target width (W dimension). If provided, will pad or crop.
        pad_value: Value to use for padding (default: -1024 for CT)
        crop_mode: How to crop when volume is larger than target ('center' or 'random')
        
    Returns:
        Volume tensor with shape (1, D, H, W)
    """
    if not HAS_RVE:
        raise RuntimeError("rad-vision-engine (rve) is not installed. Please install it to use RVE loading.")
    
    # breakpoint()
    try:
        # Load the tar file using RVE with CPU decode (26x faster than GPU decode)
        # RVE returns a torch tensor already
        # breakpoint()
        volume = rve.load_sample(filepath, use_hardware_acceleration=False)
        
        # Convert bfloat16 to float16 if needed, otherwise keep as is
        # if volume.dtype == torch.bfloat16:
        #     volume = volume.to(torch.float16)
        
        # RVE should return shape (D, H, W), add channel dimension if needed
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)  # Add channel dimension: (D, H, W) -> (1, D, H, W)
        elif volume.dim() != 4:
            raise ValueError(f"Unexpected volume shape from RVE: {volume.shape}")
        
        # Pad Z dimension if needed
        if target_slices is not None and volume.shape[1] < target_slices:
            z_diff = target_slices - volume.shape[1]
            z_pad_before = z_diff // 2
            z_pad_after = z_diff - z_pad_before
            # Pad only the Z dimension (dimension 1 after channel)
            padding = (0, 0, 0, 0, z_pad_before, z_pad_after)  # (W, H, D) padding in reverse order
            volume = torch.nn.functional.pad(volume, padding, mode='constant', value=pad_value)
        elif target_slices is not None and volume.shape[1] > target_slices:
            # Crop D dimension
            if crop_mode == 'random':
                # Random crop for training
                start_d = torch.randint(0, volume.shape[1] - target_slices + 1, (1,)).item()
                volume = volume[:, start_d:start_d + target_slices]
            else:
                # Center crop for validation/inference
                start_d = (volume.shape[1] - target_slices) // 2
                volume = volume[:, start_d:start_d + target_slices]
        
        # Handle H dimension (height)
        if target_h is not None:
            current_h = volume.shape[2]
            if current_h < target_h:
                # Pad H dimension
                h_diff = target_h - current_h
                h_pad_before = h_diff // 2
                h_pad_after = h_diff - h_pad_before
                padding = (0, 0, h_pad_before, h_pad_after, 0, 0)  # (W, H, D) padding in reverse order
                volume = torch.nn.functional.pad(volume, padding, mode='constant', value=pad_value)
            elif current_h > target_h:
                # Crop H dimension
                # if crop_mode == 'random':
                #     start_h = torch.randint(0, current_h - target_h + 1, (1,)).item()
                #     volume = volume[:, :, start_h:start_h + target_h, :]
                # else:
                start_h = (current_h - target_h) // 2
                volume = volume[:, :, start_h:start_h + target_h, :]
        
        # Handle W dimension (width)
        if target_w is not None:
            current_w = volume.shape[3]
            if current_w < target_w:
                # Pad W dimension
                w_diff = target_w - current_w
                w_pad_before = w_diff // 2
                w_pad_after = w_diff - w_pad_before
                padding = (w_pad_before, w_pad_after, 0, 0, 0, 0)  # (W, H, D) padding in reverse order
                volume = torch.nn.functional.pad(volume, padding, mode='constant', value=pad_value)
            elif current_w > target_w:
                # Crop W dimension
                # if crop_mode == 'random':
                #     start_w = torch.randint(0, current_w - target_w + 1, (1,)).item()
                #     volume = volume[:, :, :, start_w:start_w + target_w]
                # else:
                start_w = (current_w - target_w) // 2
                volume = volume[:, :, :, start_w:start_w + target_w]

        return volume
    except Exception as e:
        print(f"Failed to load RVE volume from {filepath}: {e}")
        raise


def apply_merlin_gpu_transforms(batch: torch.Tensor, 
                               window_type: Optional[Union[str, List[str]]] = None,
                               modality: str = 'CT',
                               normalize_mean: float = 0.289,
                               normalize_std: float = 0.198,
                               normalize: bool = True) -> torch.Tensor:
    """
    Apply windowing and normalization transforms to a batch on GPU.
    This is a vectorized implementation for efficiency.
    
    Args:
        batch: Tensor of shape (B, C, D, H, W)
        window_type: Window type(s) to apply (e.g., 'lung', ['lung', 'bone'], 'all')
        modality: Modality for windowing (default: 'CT')
        normalize_mean: Mean for normalization
        normalize_std: Std for normalization
        
    Returns:
        Transformed batch tensor
    """
    if not HAS_RVE or window_type is None:
        # Just apply normalization
        return (batch - normalize_mean) / normalize_std
    
    B, C, D, H, W = batch.shape
    device = batch.device

    # Log once per modality per device
    log_key = f"{modality}_{device}"
    if not hasattr(apply_merlin_gpu_transforms, '_logged'):
        apply_merlin_gpu_transforms._logged = set()
    if log_key not in apply_merlin_gpu_transforms._logged:
        print(f"Applying Merlin GPU transforms: modality={modality}, window_type={window_type}, device={device}")
        apply_merlin_gpu_transforms._logged.add(log_key)

    # Vectorized windowing implementation
    if window_type == 'all':
        # Get all available windows
        windows = rve.get_available_windows(modality)
        # Allocate output tensor
        windowed = torch.zeros((B, len(windows) * C, D, H, W), device=device, dtype=torch.bfloat16)
        
        # Apply each window type to the entire batch
        for i, window in enumerate(windows):
            if C == 1:
                # Apply window to batch without channel dim
                windowed[:, i] = rve.apply_windowing(batch.squeeze(1), window, modality)
            else:
                ## loop over the channels
                for j in range(C):
                    windowed[:, i * C + j] = rve.apply_windowing(batch[:, j], window, modality)
                # windowed[:, i] = rve.apply_windowing(batch[:, 0], window, modality)

        batch = windowed
    elif isinstance(window_type, list):
        # Multiple specific windows
        windowed = torch.zeros((B, len(window_type), D, H, W), device=device, dtype=torch.bfloat16)
        for i, window in enumerate(window_type):
            if C == 1:
                windowed[:, i] = rve.apply_windowing(batch.squeeze(1), window, modality)
            else:
                windowed[:, i] = rve.apply_windowing(batch[:, 0], window, modality)
        batch = windowed
    else:
        # Single window
        if C == 1:
            batch = rve.apply_windowing(batch.squeeze(1), window_type, modality).unsqueeze(1)
        else:
            windowed = torch.zeros((B, C, D, H, W), device=device, dtype=torch.bfloat16)
            for j in range(C):
                windowed[:, j] = rve.apply_windowing(batch[:, j], window_type, modality)
            batch = windowed
    
    # Apply normalization after windowing
    # Windowing returns values in [0, 1] range
    # convert the list of normalize_mean and normalize_std to a tensor
    if normalize:
        normalize_mean = torch.tensor(normalize_mean, device=device).to(torch.bfloat16)
        normalize_std = torch.tensor(normalize_std, device=device).to(torch.bfloat16)
        batch = (batch - normalize_mean) / normalize_std
    
    return batch


class MerlinAbdCTDataset(Dataset):
    """
    Merlin ABD CT dataset for medical image analysis.
    
    Supports loading from:
    - RVE tar files with torch tensors (recommended for best performance)
    - Cached .pt files in directory structure: cache_dir/sample_name/volume.pt
    - Original .nii.gz files
    
    Example usage with RVE and windowing:
        dataset = MerlinAbdCTDataset(
            input_filename="train.json",
            window_type='all',  # Apply all 11 CT windows
            use_rve=True,
            rve_manifest="/path/to/rve/mapping.csv",
            # ... other parameters
        )
    """
    def __init__(self, input_filename, transforms, img_paths_key, csv_caption_key, 
                 sep=None, data_root=".", tokenizer=None, caption_transform=None, 
                 target_d=192, target_hw=None, pad_value=-1.0, transform_option="pad,nearest",
                 use_cache=True, cache_dir="data/merlin/cache",
                 cache_manifest="/scratch/code/miniclip/merlin_abd_ct_cache_manifest.csv",
                 text_cache_dir=None,
                 use_rve=False, rve_dir=None, rve_manifest=None,
                 window_type: Optional[Union[str, List[str]]] = None, 
                 modality: str = 'CT',
                 image_size: Optional[int] = None,
                 is_train: bool = True):
        
        self.input_filename = input_filename
        self.transforms = transforms
        self.img_paths_key = img_paths_key
        self.csv_caption_key = csv_caption_key
        self.sep = sep
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.caption_transform = caption_transform
        self.target_d = target_d
        self.target_hw = target_hw
        self.pad_value = pad_value
        self.transform_option = transform_option
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_manifest_path = cache_manifest
        self.text_cache_dir = text_cache_dir
        
        # RVE configuration
        self.use_rve = use_rve
        self.rve_dir = rve_dir
        self.rve_manifest_path = rve_manifest
        
        # Windowing configuration
        self.window_type = window_type
        self.modality = modality
        
        # Image size and training mode
        self.image_size = image_size
        self.is_train = is_train
        
        if self.window_type and HAS_RVE:
            available = rve.get_available_windows(self.modality)
            print(f"Windowing enabled - Available windows: {available}")
            if isinstance(self.window_type, str) and self.window_type != 'all':
                if self.window_type not in available:
                    print(f"Warning: Window '{self.window_type}' not available. Available: {available}")

        ## add 3d tranform for permute and normalizaiton
        # self.transforms_3d = torchvision.transforms.Compose([
        #     torchvision.transforms.Normalize(mean=[0.289], std=[0.198])
        # ])
        self.transforms_3d = None
        
        # Load the JSONL data
        self.df = pd.read_json(input_filename, lines=True)
        print(f"Loaded {len(self.df)} samples from {input_filename}")
        
        # Setup RVE manifest if using RVE
        self.rve_manifest = None
        if self.use_rve and self.rve_manifest_path and os.path.exists(self.rve_manifest_path):
            try:
                rve_df = pd.read_csv(self.rve_manifest_path)
                # self.rve_manifest = rve_df
                # Create mapping from sample_name to output_path
                # Extract sample name from source path (e.g., AC421363e from /path/AC421363e.nii.gz)
                ## for the mapping.csv
                # rve_df['sample_name'] = rve_df['source_path'].apply(
                #     lambda x: os.path.basename(x).replace('.nii.gz', '')
                # )
                # self.rve_manifest = rve_df.set_index('sample_name')['output_path'].to_dict()
                self.rve_manifest = rve_df.set_index('sample_name')['image_cache_path'].to_dict()
                print(f"Loaded RVE manifest with {len(self.rve_manifest)} entries")
                if HAS_RVE:
                    print("RVE is available and will be used for loading")
                else:
                    print("Warning: RVE manifest loaded but rve module not available")
                    self.use_rve = False
            except Exception as e:
                print(f"Failed to load RVE manifest: {e}")
                self.rve_manifest = None
                self.use_rve = False
        
        # Setup volume cache
        self.cache_manifest = None
        if self.use_cache and os.path.exists(self.cache_manifest_path):
            try:
                self.cache_manifest = pd.read_csv(self.cache_manifest_path)
                self.cache_manifest = self.cache_manifest.set_index('sample_name')['image_cache_path'].to_dict()
                print(f"Loaded volume cache manifest with {len(self.cache_manifest)} entries")
            except Exception as e:
                print(f"Failed to load volume cache manifest: {e}")
                self.cache_manifest = None
        
        # Setup text cache
        self.build_text_cache()

    def build_text_cache(self):
        """
        Build text cache for the dataset.
        """
        if self.text_cache_dir is None:
            self.text_embedder = None
        else:
            self.text_embedder = create_embedder_from_hub(
                local_cache_dir=self.text_cache_dir,
                revision=None,
                token=None,
                force_download=False
            )

    def _get_cache_key(self, sample: Dict[str, Any], key: str = "sample_name") -> str:
        """Generate a cache key for a text."""
        text_cache_key = str(sample[key])
        text_cache_key = hashlib.sha256(text_cache_key.encode()).hexdigest()
        return text_cache_key

    def apply_windowing(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply windowing to a volume using RVE's torch-based implementation.
        
        Args:
            volume: Input volume tensor (C, D, H, W) or (1, D, H, W)
            
        Returns:
            Windowed volume tensor with values in [0, 1]
        """
        if not self.window_type or not HAS_RVE:
            return volume
        
        # Store original dtype
        orig_dtype = volume.dtype
        
        # Remove batch dimension if present (we handle single volumes)
        squeeze_batch = False
        if volume.dim() == 5:  # (B, C, D, H, W)
            squeeze_batch = True
            volume = volume.squeeze(0)
        
        # Remove channel dimension for windowing (RVE expects 3D volumes)
        if volume.dim() == 4 and volume.shape[0] == 1:  # (1, D, H, W)
            volume = volume.squeeze(0)  # (D, H, W)
        
        # Ensure we have 3D volume for windowing
        if volume.dim() != 3:
            print(f"Warning: Expected 3D volume for windowing, got shape {volume.shape}")
            return volume.unsqueeze(0) if volume.dim() == 3 else volume
        
        try:
            # Apply windowing using RVE's torch implementation
            # RVE handles both single and multiple windows properly
            windowed = rve.apply_windowing(volume, self.window_type, self.modality)
            
            # Ensure output is torch tensor (should already be)
            if not isinstance(windowed, torch.Tensor):
                windowed = torch.from_numpy(windowed)
            
            # Handle output shape
            if windowed.dim() == 3:  # Single window: (D, H, W)
                windowed = windowed.unsqueeze(0)  # Add channel: (1, D, H, W)
            elif windowed.dim() == 4:  # Multiple windows: (C, D, H, W)
                pass  # Already has channel dimension
            
            # Restore batch dimension if it was removed
            if squeeze_batch:
                windowed = windowed.unsqueeze(0)
            
            # Preserve original dtype if needed
            if windowed.dtype != orig_dtype:
                windowed = windowed.to(orig_dtype)
            
            return windowed
            
        except Exception as e:
            print(f"Windowing failed: {e}, returning original volume")
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)
            if squeeze_batch:
                volume = volume.unsqueeze(0)
            return volume

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # print(f"Row: {row}")
        sample_name = row.get('sample_name', str(idx))
        
        # Load image volume
        volume = None
        
        # Try RVE first if enabled
        if self.use_rve and self.rve_manifest is not None:
            rve_path = self.rve_manifest.get(sample_name)
            # print(f"RVE path: {rve_path}")
            if rve_path and os.path.exists(rve_path):
                try:
                    # Pass target dimensions and crop mode based on training/validation
                    volume = load_rve_volume(
                        rve_path, 
                        target_slices=self.target_d, 
                        target_h=self.target_hw,
                        target_w=self.target_hw,
                        pad_value=self.pad_value,
                        crop_mode='random' if self.is_train else 'center'
                    )
                    # Keep on CPU - DataLoader will handle device placement
                except Exception as e:
                    print(f"Failed to load {sample_name} from RVE, trying cache: {e}")
                    volume = None

        # Try cache if RVE failed or not used
        if volume is None and self.use_cache and self.cache_manifest is not None:
            # Use directory structure: sample_name/volume.pt
            cache_path = os.path.join(self.cache_dir, sample_name, "volume.pt")
            
            if os.path.exists(cache_path):
                try:
                    volume = load_cached_volume(cache_path)
                except Exception as e:
                    print(f"Failed to load {sample_name} from cache, falling back to .nii.gz: {e}")
                    volume = None
        

        # Fall back to loading from nii.gz if all else fails
        if volume is None:
            volume = self._load_from_nii(row)
        
        # Apply transforms if provided
        # if self.transforms:
        #     volume = self.transforms(volume)

        # make the volume to be in the format of (C, D, H, W)
        # check if last two dimensions are (256, 256)
        # if volume.shape[-2:] != (256, 256):
        #     volume = volume.permute(0, 3, 1, 2)

        ## check if the H=W in the volume, if not need to do a permute
        if volume.shape[-2] != volume.shape[-1]:
            volume = volume.permute(0, 3, 1, 2)
        
        # Load text features (similar to MIMIC implementation)
        text_cache_key = self._get_cache_key(row, key="sample_name")
        
        if self.text_embedder is not None:
            # Load from text cache
            text_features = self.text_embedder.load_embedding_from_cache_key(text_cache_key)
        else:
            # Load and process caption directly
            caption = row[self.csv_caption_key]
            if self.caption_transform:
                caption = self.caption_transform(caption)
            
            # Tokenize caption if tokenizer is provided
            if self.tokenizer:
                text_features = self.tokenizer([caption])[0]
            else:
                text_features = caption
        
        # Ensure proper shape (add channel dimension if needed)
        if len(volume.shape) == 3:
            volume = volume.unsqueeze(0)
        
        # flip the volume along the first dimension
        volume = torch.flip(volume, dims=[1])
        # print(f"Flipped volume: {volume.shape}")
        
        # Skip windowing and normalization - will be done on GPU after collation
        # This is more efficient for batched operations
        # make it float32
        volume = volume.to(torch.float32)
        
        return volume, text_features

    def _load_from_nii(self, row):
        """Load volume from .nii.gz file."""
        nii_path = row.get(self.img_paths_key, None)
        if nii_path is None:
            # Fallback: assume sample_name is the filename
            nii_path = f"{row.get('sample_name', 'unknown')}.nii.gz"
        volume_path = os.path.join(self.data_root, nii_path)
        # Use image_size if provided, otherwise default to 256
        h_w_size = self.image_size if self.image_size is not None else 256
        volume = load_nii_volume(volume_path, target_shape=(h_w_size, h_w_size, self.target_d))
        # Keep on CPU - DataLoader will handle device placement
        # Note: normalization is applied in __getitem__ after windowing
        return volume


def get_merlin_abd_ct_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    Get Merlin ABD CT dataset with cache support.
    """
    # breakpoint()
    input_filename = args.train_data if is_train else args.val_data
    data_root = args.medcsv_data_root
    
    # Cache configuration
    use_cache = getattr(args, 'use_cache', True)
    cache_dir = getattr(args, 'cache_dir', 'data/merlin/cache')
    cache_manifest = getattr(args, 'cache_manifest', '/scratch/code/miniclip/merlin_abd_ct_cache_manifest.csv')
    
    # Text cache configuration
    text_cache_dir = getattr(args, 'text_cache_dir', None)
    
    # Dataset parameters
    target_d = getattr(args, 'medcsv_target_d', 192)
    pad_value = getattr(args, 'medcsv_pad_value', -1.0)
    transform_option = getattr(args, 'transform_option', 'pad,nearest')
    
    # Get image size from args
    image_size = getattr(args, 'image_size', None)
    target_hw = getattr(args, 'target_hw', None)
    
    # RVE configuration
    use_rve = getattr(args, 'use_rve', False)
    rve_dir = getattr(args, 'rve_dir', None)
    rve_manifest = getattr(args, 'rve_manifest', None)
    
    # Windowing configuration
    window_type = getattr(args, 'window_type', None)
    modality = getattr(args, 'modality', 'CT')
    
    dataset = MerlinAbdCTDataset(
        input_filename=input_filename,
        transforms=preprocess_fn,
        img_paths_key=args.csv_img_key,
        csv_caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        data_root=data_root,
        tokenizer=tokenizer,
        caption_transform=getattr(args, 'medcsv_caption_transform', None),
        target_d=target_d,
        target_hw=target_hw,
        pad_value=pad_value,
        transform_option=transform_option,
        use_cache=use_cache,
        cache_dir=cache_dir,
        cache_manifest=cache_manifest,
        text_cache_dir=text_cache_dir,
        use_rve=use_rve,
        rve_dir=rve_dir,
        rve_manifest=rve_manifest,
        window_type=window_type,
        modality=modality,
        image_size=image_size,
        is_train=is_train
    )
    
    return dataset 
