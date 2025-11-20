"""
Hugging Face Hub utilities for cached embeddings with compression support.

This module provides functions to upload compressed cache directories to HF Hub
and download/decompress them for use.
"""

import os
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, hf_hub_download, create_repo
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface_hub")


def compress_cache_directory(cache_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Compress a cache directory into a tar.gz archive.
    
    Args:
        cache_path: Path to the cache directory to compress
        output_path: Optional path for the output archive. If None, creates {cache_path}.tar.gz
        
    Returns:
        Path to the created archive
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_path}")
    
    if output_path is None:
        output_path = cache_path.with_suffix('.tar.gz')
    else:
        output_path = Path(output_path)
    
    # Count total files for progress
    total_files = sum(1 for f in cache_path.rglob('*') if f.is_file())
    logger.info(f"Compressing {total_files} files from {cache_path} to {output_path}")
    
    # Import tqdm here to avoid import issues
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_files, desc="Compressing", unit="files")
    except ImportError:
        progress_bar = None
    
    def progress_filter(tarinfo):
        """Filter function that updates progress bar."""
        if progress_bar and tarinfo.isfile():
            progress_bar.update(1)
        return tarinfo
    
    try:
        with tarfile.open(output_path, 'w:gz') as tar:
            # Add the cache directory contents, preserving structure
            tar.add(cache_path, arcname=cache_path.name, filter=progress_filter)
    finally:
        if progress_bar:
            progress_bar.close()
    
    # Get compression stats
    original_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    compressed_size = output_path.stat().st_size
    compression_ratio = compressed_size / original_size if original_size > 0 else 0
    
    logger.info(f"Compression complete: {original_size / 1024 / 1024:.2f} MB → {compressed_size / 1024 / 1024:.2f} MB "
               f"(ratio: {compression_ratio:.2f})")
    
    return output_path


def decompress_cache_archive(archive_path: Union[str, Path], extract_to: Optional[Union[str, Path]] = None) -> Path:
    """
    Decompress a cache archive.
    
    Args:
        archive_path: Path to the tar.gz archive
        extract_to: Directory to extract to. If None, extracts to archive's parent directory
        
    Returns:
        Path to the extracted cache directory
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    if extract_to is None:
        extract_to = archive_path.parent
    else:
        extract_to = Path(extract_to)
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Decompressing {archive_path} to {extract_to}")
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    
    # Find the extracted cache directory (should be the only top-level directory)
    extracted_items = list(extract_to.iterdir())
    cache_dirs = [item for item in extracted_items if item.is_dir()]
    
    if len(cache_dirs) != 1:
        raise ValueError(f"Expected exactly one directory in archive, found: {cache_dirs}")
    
    extracted_cache_dir = cache_dirs[0]
    logger.info(f"Decompression complete: {extracted_cache_dir}")
    
    return extracted_cache_dir


def upload_cache_to_hub(
    cache_path: Union[str, Path],
    repo_id: str,
    compress: bool = True,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    cleanup_archive: bool = True
) -> str:
    """
    Upload a cache directory to Hugging Face Hub, with optional compression.
    
    Args:
        cache_path: Path to the cache directory
        repo_id: Repository ID on Hugging Face Hub
        compress: Whether to compress the cache before uploading
        revision: Git revision (branch/tag) to upload to
        token: Hugging Face token
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        cleanup_archive: Whether to delete the compressed archive after upload
        
    Returns:
        URL of the uploaded repository
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    cache_path = Path(cache_path)
    api = HfApi()
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id, private=private, repo_type="dataset", exist_ok=True, overwrite=True)
        api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True, repo_type="dataset")
        logger.info(f"Repository {repo_id} created/verified at branch {revision}")
    except Exception as e:
        logger.warning(f"Could not create repository: {e}")
    
    if compress:
        # Create compressed archive
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / f"{cache_path.name}.tar.gz"
            compress_cache_directory(cache_path, archive_path)
            
            # Upload the compressed archive
            logger.info(f"Uploading compressed cache to {repo_id}")
            api.upload_file(
                path_or_fileobj=archive_path,
                path_in_repo="cache.tar.gz",
                repo_id=repo_id,
                revision=revision,
                repo_type="dataset",
                commit_message=commit_message or f"Upload compressed cache from {cache_path.name}"
            )
            
            # Upload metadata file with compression info
            metadata = {
                "compressed": True,
                "original_cache_name": cache_path.name,
                "compression_format": "tar.gz",
                "upload_info": {
                    "original_size_mb": sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / 1024 / 1024,
                    "compressed_size_mb": archive_path.stat().st_size / 1024 / 1024
                }
            }
            
            metadata_path = Path(temp_dir) / "cache_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="cache_metadata.json", 
                repo_id=repo_id,
                revision=revision,
                repo_type="dataset",
                commit_message="Add cache metadata"
            )
            
    else:
        # Upload directory directly (slower for large caches)
        logger.info(f"Uploading cache directory to {repo_id}")
        api.upload_folder(
            folder_path=cache_path,
            repo_id=repo_id,
            revision=revision,
            repo_type="dataset",
            commit_message=commit_message or f"Upload cache from {cache_path.name}"
        )
    
    repo_url = f"https://huggingface.co/{repo_id}"
    if revision:
        repo_url += f"/tree/{revision}"
    
    logger.info(f"Upload complete: {repo_url}")
    return repo_url


def download_cache_from_hub(
    repo_id: str,
    cache_dir: Union[str, Path],
    revision: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False
) -> Path:
    """
    Download and decompress a cache from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID on Hugging Face Hub
        cache_dir: Local directory to save the cache
        revision: Git revision to download from
        token: Hugging Face token
        force_download: Whether to force re-download if cache already exists
        
    Returns:
        Path to the downloaded cache directory
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if cache metadata exists to determine if it's compressed
    try:
        metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename="cache_metadata.json",
            revision=revision,
            repo_type="dataset",
            token=token
        )
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        is_compressed = metadata.get("compressed", False)
        
    except Exception as e:
        logger.warning(f"Could not download metadata, assuming uncompressed: {e}")
        is_compressed = False
    
    if is_compressed:
        # Download and decompress
        logger.info(f"Downloading compressed cache from {repo_id}")
        
        archive_path = hf_hub_download(
            repo_id=repo_id,
            filename="cache.tar.gz",
            revision=revision,
            token=token,
            repo_type="dataset",
            force_download=force_download
        )
        
        # Extract to cache directory
        extracted_cache = decompress_cache_archive(archive_path, cache_dir)
        
        # If the extracted directory has a different name, move/rename it
        final_cache_path = cache_dir / "cache"
        if extracted_cache != final_cache_path:
            if final_cache_path.exists():
                shutil.rmtree(final_cache_path)
            shutil.move(str(extracted_cache), str(final_cache_path))
            extracted_cache = final_cache_path
        
        logger.info(f"Cache downloaded and extracted to: {extracted_cache}")
        return extracted_cache
        
    else:
        # Download directory directly
        logger.info(f"Downloading uncompressed cache from {repo_id}")
        # This would require downloading all files individually
        # For now, assume compressed format is preferred
        raise NotImplementedError("Direct directory download not implemented. Use compressed format.")


def create_embedder_from_hub(
    repo_id: Optional[str] = None,
    local_cache_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
    categories: Optional[List[str]] = ["all_findings"]
):
    """
    Create a CachedTextEmbedding instance from a Hub repository.
    
    Args:
        repo_id: Repository ID on Hugging Face Hub
        local_cache_dir: Local directory for cache. If None, uses ./embeddings_cache/{repo_id}
        revision: Git revision to download from
        token: Hugging Face token
        force_download: Whether to force re-download
        
    Returns:
        CachedTextEmbedding instance
    """
    from .cached_text_embedding import CachedTextEmbedding
    
    if local_cache_dir is None:
        local_cache_dir = Path("./embeddings_cache") / repo_id.replace("/", "_")
    else:
        local_cache_dir = Path(local_cache_dir)
    
    # Download cache if not exists or force_download
    if not local_cache_dir.exists() or force_download and repo_id is not None:
        cache_path = download_cache_from_hub(
            repo_id=repo_id,
            cache_dir=local_cache_dir.parent,
            revision=revision,
            token=token,
            force_download=force_download
        )
    else:
        cache_path = local_cache_dir / "embeddings"
    
    # Load metadata to get model info
    metadata_file = local_cache_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Cache metadata not found: {metadata_file}")
    
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError("Model name not found in cache metadata")
    
    # Create embedder instance
    embedder = CachedTextEmbedding(
        model_name=model_name,
        cache_dir=str(local_cache_dir),
        normalize_embeddings=metadata.get("normalize_embeddings", True),
        max_length=metadata.get("max_length", 512),
        categories=categories or ["all_findings"]
    )
    
    logger.info(f"Created embedder from Hub cache: {local_cache_dir}")
    return embedder 