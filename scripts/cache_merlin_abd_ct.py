#!/usr/bin/env python3
"""
Cache preprocessed Merlin ABD CT volumes for fast training (Ray parallel version).
- Scans data/merlin for .nii.gz files
- Loads, resizes to 256x256x192, normalizes, saves as .pt
- Outputs a manifest CSV: sample_name,cache_path
- Uses Ray for parallel processing
"""
import os
import glob
import argparse
import nibabel as nib
import torch
import numpy as np
from skimage.transform import resize
import pandas as pd
from tqdm import tqdm
import ray


def process_volume_local(nii_path, cache_path, target_shape=(256, 256, 192)):
    try:
        img = nib.load(nii_path)
        vol = img.get_fdata()
        vmin, vmax = np.min(vol), np.percentile(vol, 99)
        vol = np.clip(vol, vmin, vmax)
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)
        vol_resized = resize(vol, target_shape, order=1, mode='constant', cval=0, anti_aliasing=True)
        tensor = torch.from_numpy(vol_resized).float().unsqueeze(0)  # (1, D, H, W)
        torch.save(tensor, cache_path)
        return {'sample_name': os.path.splitext(os.path.basename(nii_path))[0].replace('.nii', ''), 'cache_path': cache_path, 'status': 'ok'}
    except Exception as e:
        return {'sample_name': os.path.splitext(os.path.basename(nii_path))[0].replace('.nii', ''), 'cache_path': cache_path, 'status': f'fail: {e}'}

@ray.remote
def process_volume_ray(nii_path, cache_path, target_shape=(256, 256, 192)):
    return process_volume_local(nii_path, cache_path, target_shape)


def main():
    parser = argparse.ArgumentParser(description="Cache Merlin ABD CT volumes for fast training (Ray parallel version).")
    parser.add_argument('--data-dir', type=str, default='data/merlin', help='Directory with .nii.gz files')
    parser.add_argument('--cache-dir', type=str, default='data/merlin_cache', help='Where to save cached .pt files')
    parser.add_argument('--manifest', type=str, default='merlin_abd_ct_cache_manifest.csv', help='Output manifest CSV')
    parser.add_argument('--pattern', type=str, default='**/*.nii.gz', help='Glob pattern for nii.gz files')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of Ray workers (default: all CPUs)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process (for debugging)')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    nii_files = glob.glob(os.path.join(args.data_dir, args.pattern), recursive=True)
    if args.max_samples is not None:
        nii_files = nii_files[:args.max_samples]
    print(f"Found {len(nii_files)} nii.gz files to process.")

    ray.init(num_cpus=args.num_workers)

    tasks = []
    for nii_path in nii_files:
        sample_name = os.path.splitext(os.path.basename(nii_path))[0].replace('.nii', '')
        cache_path = os.path.join(args.cache_dir, f"{sample_name}.pt")
        if os.path.exists(cache_path):
            # Already cached, skip processing
            continue
        tasks.append(process_volume_ray.remote(nii_path, cache_path))

    manifest = []
    remaining = list(tasks)
    with tqdm(total=len(tasks), desc="Processing volumes (Ray)") as pbar:
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            out = ray.get(done[0])
            if out['status'] == 'ok':
                manifest.append({'sample_name': out['sample_name'], 'cache_path': out['cache_path']})
            else:
                print(f"Failed: {out['sample_name']} - {out['status']}")
            pbar.update(1)

    # Add already-cached files to manifest
    for nii_path in nii_files:
        sample_name = os.path.splitext(os.path.basename(nii_path))[0].replace('.nii', '')
        cache_path = os.path.join(args.cache_dir, f"{sample_name}.pt")
        if os.path.exists(cache_path) and not any(m['sample_name'] == sample_name for m in manifest):
            manifest.append({'sample_name': sample_name, 'cache_path': cache_path})

    # Save manifest
    df = pd.DataFrame(manifest)
    df.to_csv(args.manifest, index=False)
    print(f"Manifest saved to {args.manifest}")
    print(f"Cached {len(manifest)} volumes.")

    ray.shutdown()

if __name__ == "__main__":
    main()
