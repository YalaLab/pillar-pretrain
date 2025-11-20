#!/usr/bin/env python3
"""
Cache MERLIN Embeddings Script

This script extracts text embeddings from a Hugging Face model for the MERLIN ABD CT dataset,
using the existing CachedTextEmbedding system for better integration with miniclip.

This is an adaptation of cache_mimic_embeddings.py to work with the MERLIN dataset format.

Usage examples:
  # Basic usage with train JSON file
  python cache_merlin_embeddings.py --model-name qwen/qwen3-embedding-8b --json-file data/merlin/dataset-splits/merlin_abd_ct/train.json

  # With custom cache directory and batch size
  python cache_merlin_embeddings.py --model-name qwen/qwen3-embedding-8b --json-file data/merlin/dataset-splits/merlin_abd_ct/train.json --cache-dir ./merlin_cache --batch-size 64
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from tqdm import tqdm

# Add the miniclip source to the path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miniclip import cache_captions_parallel


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_merlin_captions(
    json_file: str,
    text_key: str = "report_metadata",
    sample_id_key: str = "sample_name",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load captions from a MERLIN JSON file."""
    print(f"📊 Loading MERLIN metadata from: {json_file}")
    
    # Read JSONL file (one JSON object per line)
    df = pd.read_json(json_file, lines=True)
    
    print(f"📈 Metadata size: {len(df)}")
    print(f"📝 Text column: {text_key}")
    print(f"🔑 Sample ID key: {sample_id_key}")


    captions = []
    for _, row in df.iterrows():
        report_text = row.get(text_key)
        sample_name = row.get(sample_id_key)
        
        if report_text and isinstance(report_text, str) and sample_name:
            # For MERLIN, we use the full report text as the caption
            captions.append({
                "study_id": sample_name,  # Use sample_name as study_id
                "category_findings": {"all_findings": report_text},  # Use the full report text
            })

    if max_samples and len(captions) > max_samples:
        captions = captions[:max_samples]
        print(f"📊 Limited to {max_samples} samples")

    print(f"📝 Valid texts: {len(captions)}/{len(df)}")
    return captions


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache text embeddings for the MERLIN ABD CT dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--model-name", 
        required=True, 
        help="Hugging Face model name (e.g., 'qwen/qwen3-embedding-8b')"
    )
    parser.add_argument(
        "--json-file", 
        required=True, 
        help="Path to the MERLIN JSON file (train.json, valid.json, or test.json)."
    )
    
    # Optional arguments
    parser.add_argument(
        "--cache-dir", 
        default="./merlin_embeddings_cache", 
        help="Directory to store cached embeddings (default: ./merlin_embeddings_cache)"
    )
    parser.add_argument(
        "--text-key", 
        default="report_metadata", 
        help="Key in the JSON file for the text to embed (default: report_metadata)"
    )
    parser.add_argument(
        "--sample-id-key", 
        default="sample_name", 
        help="Key in the JSON file for the sample ID (default: sample_name)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512, 
        help="Maximum sequence length for tokenization (default: 512)"
    )
    parser.add_argument(
        "--no-normalize", 
        action="store_true", 
        help="Don't normalize embeddings (default: normalize)"
    )
    parser.add_argument(
        "--num-processes", 
        type=int, 
        help="Number of processes for parallel processing (default: auto-detect)"
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        help="Device to use ('auto', 'cpu', 'cuda', etc.) (default: auto)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("🚀 MERLIN Embedding Cache Script")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"JSON file: {args.json_file}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Load dataset texts
        captions = load_merlin_captions(
            json_file=args.json_file,
            text_key=args.text_key,
            sample_id_key=args.sample_id_key,
            max_samples=args.max_samples
        )
        
        if not captions:
            print("✗ No valid texts found in the JSON file.")
            sys.exit(1)
        
        # Extract embeddings
        stats = cache_captions_parallel(
            captions=captions,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            num_processes=args.num_processes,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            normalize_embeddings=not args.no_normalize
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n✅ Extraction completed successfully!")
        print(f"⏱️  Total time: {processing_time:.2f} seconds")
        print(f"📊 Processed: {stats['total_processed']}/{stats['total_captions']} texts")
        if processing_time > 0:
            print(f"⚡ Speed: {stats['total_processed']/processing_time:.2f} texts/second")
        
    except KeyboardInterrupt:
        print("\n⛔ Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during extraction: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
