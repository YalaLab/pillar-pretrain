"""
CachedTextEmbedding - A distributed text embedding caching system with lazy loading

This module provides functionality to generate and cache text embeddings from any Hugging Face model
with support for parallel/distributed processing and lazy loading for PyTorch DataLoaders.

The embeddings are stored as individual torch tensor files and loaded on-demand, making it
memory-efficient for large datasets.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from huggingface_hub import HfApi
import multiprocessing as mp
import tempfile
import tarfile

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)



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




class CachedTextEmbedding:
    """
    A class for generating and caching text embeddings from Hugging Face models
    with lazy loading support for PyTorch DataLoaders.
    
    Features:
    - Stores embedding paths for lazy loading (memory efficient)
    - Supports any Hugging Face text model
    - Parallel processing for batch generation
    - Distributed processing support
    - Resume/fix functionality for missing embeddings
    - Compatible with PyTorch DataLoaders
    """
    
    def __init__(
        self,
        model_name: str,
        cache_dir: str = "./embeddings_cache",
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "auto",
        normalize_embeddings: bool = True,
        categories: List[str] = ["all_findings"],
        **model_kwargs
    ):
        """
        Initialize the CachedTextEmbedding class.
        
        Args:
            model_name: Model identifier (Hugging Face for transformers, vLLM-compatible for vllm)
            cache_dir: Directory to store cached embeddings
            batch_size: Batch size for processing
            max_length: Maximum sequence length for tokenization (transformers only)
            device: Device to run the model on ("auto", "cpu", "cuda", etc.)
            normalize_embeddings: Whether to normalize embeddings
            **model_kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.categories = categories
            
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize model and tokenizer (only when needed for generation)
        self.model = None
        self.tokenizer = None
        self.model_kwargs = model_kwargs
        
        # Create model-specific cache subdirectory
        # model_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        # self.model_cache_dir = self.cache_dir / f"{model_hash}_{self.model_name.replace('/', '_')}"
        # self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings subdirectory
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
        
    def _init_model(self):
        """Initialize the model and tokenizer (lazy initialization)."""
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.model_kwargs)
                
                # Check if we should use tensor parallelism
                use_tensor_parallelism = (
                    "device_map" in self.model_kwargs and 
                    self.model_kwargs["device_map"] == "auto"
                )
                
                if use_tensor_parallelism:
                    # Use tensor parallelism - don't call .to() since device_map handles it
                    self.model = AutoModel.from_pretrained(self.model_name, **self.model_kwargs)
                    logger.info(f"Loaded model with tensor parallelism: {self.model.hf_device_map}")
                else:
                    # Standard loading
                    self.model = AutoModel.from_pretrained(self.model_name, **self.model_kwargs)
                    self.model.to(self.device)
                
                self.model.eval()
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
                    
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
                
    def _load_metadata(self):
        """Load or create metadata for the cache."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
                
            # Validate model compatibility
            if self.metadata.get("model_name") != self.model_name:
                logger.warning(f"Model name mismatch: cache has {self.metadata.get('model_name')}, "
                              f"requested {self.model_name}")
        else:
            self.metadata = {
                "model_name": self.model_name,
                "normalize_embeddings": self.normalize_embeddings,
                "max_length": self.max_length,
                "text_to_path": {},  # Maps text hash to tensor file path
                "embedding_dim": None,
                "version": "1.0",
                "dataset_name": None,  # Will be set by CLI
                "split": None  # Will be set by CLI
            }
            
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _get_cache_key(self, text: Dict[str, Any], key: str = "study_id") -> str:
        """Generate a cache key for a text."""
        cache_text = str(text[key])
        text_hash = hashlib.sha256(cache_text.encode()).hexdigest()
        return text_hash
        
    def _get_tensor_path(self, cache_key: str) -> Path:
        """Get the tensor file path for a given cache key."""
        return self.embeddings_dir / f"{cache_key}.pt"
        
    def _save_embedding(self, cache_key: str, embedding: Dict[str, torch.Tensor], update_metadata: bool = True) -> Path:
        """Save an embedding tensor to cache and return its path."""
        tensor_path = self._get_tensor_path(cache_key)
        
        # Detach per-sample tensors from any shared (batch) backing storage before save.
        # Per-batch embeddings are row VIEWS into a [batch, dim] tensor; torch.save persists
        # the whole underlying storage, bloating each .pt to batch_size*dim (e.g.
        # 128*4096*4B ~= 2MB vs 16KB). clone() gives each row its own contiguous storage.
        def _compact(t):
            return t.detach().cpu().clone().contiguous() if torch.is_tensor(t) else t
        if isinstance(embedding, dict):
            embedding = {k: _compact(v) for k, v in embedding.items()}
        else:
            embedding = _compact(embedding)

        # Save tensor
        torch.save(embedding, tensor_path)
        
        # Optionally update metadata (can be disabled for distributed processing)
        if update_metadata:
            relative_path = tensor_path.relative_to(self.cache_dir)
            self.metadata["text_to_path"][cache_key] = str(relative_path)
        
        return tensor_path
        
    def rebuild_metadata_from_cache(self):
        """Rebuild metadata by scanning cached tensor files (thread-safe)."""
        logger.info("Rebuilding metadata from cached files...")
        
        # Scan for all .pt files in the embeddings directory
        tensor_files = list(self.embeddings_dir.glob("*.pt"))
        
        # Clear and rebuild text_to_path mapping
        self.metadata["text_to_path"] = {}
        
        for tensor_path in tensor_files:
            cache_key = tensor_path.stem  # filename without .pt extension
            relative_path = tensor_path.relative_to(self.cache_dir)
            self.metadata["text_to_path"][cache_key] = str(relative_path)
        
        # Update embedding dimension from first valid tensor
        if tensor_files and self.metadata["embedding_dim"] is None:
            try:
                sample_embedding = torch.load(tensor_files[0], map_location='cpu')
                if sample_embedding:
                    first_key = list(sample_embedding.keys())[0]
                    self.metadata["embedding_dim"] = sample_embedding[first_key].shape[0]
            except Exception as e:
                logger.warning(f"Could not determine embedding dimension: {e}")
        
        # Save the rebuilt metadata
        self._save_metadata()
        logger.info(f"Rebuilt metadata with {len(self.metadata['text_to_path'])} entries")
        
    def _get_embedding_path(self, cache_key: str) -> Optional[Path]:
        """Get the path to a cached embedding tensor."""
        if cache_key in self.metadata["text_to_path"]:
            relative_path = self.metadata["text_to_path"][cache_key]
            full_path = self.cache_dir / relative_path
            if full_path.exists():
                return full_path
            else:
                logger.warning(f"Cached embedding file missing: {full_path}")
                # Remove from metadata since file is missing
                del self.metadata["text_to_path"][cache_key]
                return None
        return None

    def load_embedding_from_cache_key(self, cache_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Lazily load an embedding tensor for the given cache key.
        """
        embedding_path = self._get_embedding_path(cache_key)
        return torch.load(embedding_path, map_location='cpu')

    def load_embedding(self, text: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Lazily load an embedding tensor for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor if cached, None otherwise
        """
        cache_key = self._get_cache_key(text)
        embedding_path = self._get_embedding_path(cache_key)
        
        if embedding_path is None:
            return None
            
        try:
            return torch.load(embedding_path, map_location='cpu')
        except Exception as e:
            logger.warning(f"Failed to load embedding from {embedding_path}: {e}")
            return None
            
    def get_embedding_path(self, text: Dict[str, Any]) -> Optional[str]:
        """
        Get the file path for a text's embedding without loading it.
        
        Args:
            text: Input text
            
        Returns:
            Path to embedding file if cached, None otherwise
        """
        cache_key = self._get_cache_key(text, key="study_id")
        embedding_path = self._get_embedding_path(cache_key)
        return str(embedding_path) if embedding_path else None
            
    def _encode_texts(self, texts: List[Dict[str, Any]]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Encode a batch of texts with nested findings into embeddings.

        Args:
            texts: A list of dictionaries, where each dictionary has a 'study_id' 
                   and a 'category_findings' dictionary (e.g., {'C1': 'text', ...}).

        Returns:
            A dictionary mapping each study_id to a nested dictionary of 
            {category_name: embedding_tensor}.
        """
        self._init_model()
        
        captions_key = "category_findings"
        study_id_key = "study_id"
        
        all_finding_texts = []
        mapping_info = []

        for text_item in texts:
            study_id = text_item.get(study_id_key)
            findings = text_item.get(captions_key)
            # keep only certain categories
            if findings is not None:
                findings = {k: v for k, v in findings.items() if k in self.categories}
            else:
                findings = {}
            
            if not (isinstance(study_id, int) or isinstance(study_id, str)) or not isinstance(findings, dict):
                logger.warning(f"Skipping invalid item in batch: {text_item}")
                continue

            for category, finding_text in findings.items():
                if isinstance(finding_text, str):
                    all_finding_texts.append(finding_text)
                    mapping_info.append({'study_id': study_id, 'category': category})

        if not all_finding_texts:
            return {}

        # Tokenize all texts in one batch
        inputs = self.tokenizer(
            all_finding_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Handle device placement for tensor parallelism
        use_tensor_parallelism = (
            "device_map" in self.model_kwargs and 
            self.model_kwargs["device_map"] == "auto"
        )
        
        if use_tensor_parallelism:
            # For tensor parallelism, put inputs on the first GPU
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        else:
            # Standard device placement
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
                embeddings = torch.sum(embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs[0].mean(dim=1)
                
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        embeddings_on_cpu = embeddings.cpu()
        
        # Reconstruct the nested dictionary
        output_dict = {}
        for i, info in enumerate(mapping_info):
            study_id = info['study_id']
            category = info['category']
            embedding = embeddings_on_cpu[i]
            
            if study_id not in output_dict:
                output_dict[study_id] = {}
            output_dict[study_id][category] = embedding
            
        return output_dict
        
    def encode_single(self, text: str, use_cache: bool = True) -> torch.Tensor:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Input text to encode
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding as torch tensor
        """
        # Convert string to expected format
        text_dict = {
            'study_id': 0,
            'category_findings': {'all_findings': text}
        }
        
        cache_key = self._get_cache_key(text_dict)
        
        # Try to load from cache first
        if use_cache:
            cached_embedding = self.load_embedding(text_dict)
            if cached_embedding is not None:
                return cached_embedding['all_findings']
                
        # Generate embedding
        embeddings_dict = self._encode_texts([text_dict])
        if 0 in embeddings_dict:
            embedding = embeddings_dict[0]['all_findings']
        else:
            raise ValueError("Failed to generate embedding")
        
        # Save to cache
        if use_cache:
            self._save_embedding(cache_key, embeddings_dict[0])
            if self.metadata["embedding_dim"] is None:
                self.metadata["embedding_dim"] = embedding.shape[0]
            self._save_metadata()
            
        return embedding
        
    def cache_text(self, text: str, force_regenerate: bool = False) -> str:
        """
        Cache a text embedding and return the path to the cached file.
        
        Args:
            text: Input text to cache
            force_regenerate: Whether to regenerate even if cached
            
        Returns:
            Path to the cached embedding file
        """
        # Convert string to expected format
        text_dict = {
            'study_id': 0,
            'category_findings': {'all_findings': text}
        }
        
        cache_key = self._get_cache_key(text_dict)
        
        # Check if already cached and not forcing regeneration
        if not force_regenerate:
            existing_path = self._get_embedding_path(cache_key)
            if existing_path is not None:
                return str(existing_path)
        
        # Generate and save embedding
        embeddings_dict = self._encode_texts([text_dict])
        if 0 in embeddings_dict:
            embedding_dict = embeddings_dict[0]
            tensor_path = self._save_embedding(cache_key, embedding_dict)
            
            # Update metadata
            if self.metadata["embedding_dim"] is None:
                first_embedding = list(embedding_dict.values())[0]
                self.metadata["embedding_dim"] = first_embedding.shape[0]
            self._save_metadata()
            
            return str(tensor_path)
        else:
            raise ValueError("Failed to generate embedding")
        
    def cache_texts_batch(
        self,
        texts: Iterable[Dict[str, Any]],
        show_progress: bool = True,
        force_regenerate: bool = False,
        update_metadata: bool = True
    ) -> List[str]:
        """
        Cache a batch of texts and return their embedding file paths.
        
        Args:
            texts: An iterable of dictionary-like objects (e.g., pandas dataframe rows)
            show_progress: Whether to show progress bar
            force_regenerate: Whether to regenerate embeddings even if cached
            update_metadata: Whether to update metadata during processing (disable for distributed)
            
        Returns:
            List of paths to cached embedding files
        """
        paths = []
        texts_to_process = []
        indices_to_process = []
        
        # Check which texts need processing
        iterator = texts.iterrows() if hasattr(texts, 'iterrows') else enumerate(texts)
        for i, text in tqdm(iterator, total=len(texts), disable=not show_progress):
            if not force_regenerate:
                existing_path = self.get_embedding_path(text)
                if existing_path is not None:
                    paths.append(existing_path)
                    continue
            
            # Need to process this text
            texts_to_process.append(text)
            indices_to_process.append(i)
            paths.append(None)  # Placeholder
        
        # Process texts in batches
        if texts_to_process:
            batches = [
                texts_to_process[i:i + self.batch_size]
                for i in range(0, len(texts_to_process), self.batch_size)
            ]
            
            batch_iterator = tqdm(batches, desc="Caching embeddings", disable=not show_progress)
            
            processed_count = 0
            for batch_texts in batch_iterator:
                embeddings_by_study_id = self._encode_texts(batch_texts)

                if not embeddings_by_study_id:
                    processed_count += len(batch_texts)
                    continue
                
                for j, text_item in enumerate(batch_texts):
                    study_id = text_item.get("study_id")
                    
                    if study_id in embeddings_by_study_id:
                        embedding_dict = embeddings_by_study_id[study_id]
                        original_idx = indices_to_process[processed_count + j]
                        cache_key = self._get_cache_key(text_item)
                        # Pass update_metadata flag to _save_embedding
                        tensor_path = self._save_embedding(cache_key, embedding_dict, update_metadata=update_metadata)
                        paths[original_idx] = str(tensor_path)
                    else:
                        logger.warning(f"Embedding not generated for study_id: {study_id}")
                        
                processed_count += len(batch_texts)

        # Update metadata only if requested and there were new embeddings
        if update_metadata and texts_to_process:
            if self.metadata["embedding_dim"] is None and paths:
                # Load one embedding to get dimension
                first_valid_path = next(p for p in paths if p is not None)
                sample_embedding = torch.load(first_valid_path, map_location='cpu')
                self.metadata["embedding_dim"] = list(sample_embedding.values())[0].shape[0]
            self._save_metadata()
            
        return paths
        
    def get_text_paths(self, texts: List[Dict[str, Any]]) -> List[Optional[str]]:
        """
        Get the cached embedding paths for a list of texts without generating them.
        
        Args:
            texts: List of input text dictionaries
            
        Returns:
            List of paths (None for uncached texts)
        """
        return [self.get_embedding_path(text) for text in texts]
        
    def check_cache_health(self) -> Dict[str, Any]:
        """
        Check the health of the cache and identify missing files.
        
        Returns:
            Dictionary with cache health information
        """
        total_cached = len(self.metadata["text_to_path"])
        missing_files = []
        existing_files = []
        
        for cache_key, relative_path in self.metadata["text_to_path"].items():
            full_path = self.cache_dir / relative_path
            if full_path.exists():
                existing_files.append(cache_key)
            else:
                missing_files.append(cache_key)
                
        return {
            "total_cached": total_cached,
            "existing_files": len(existing_files),
            "missing_files": len(missing_files),
            "missing_cache_keys": missing_files,
            "health_percentage": (len(existing_files) / total_cached * 100) if total_cached > 0 else 100
        }
        
    def fix_missing_embeddings(self, texts_for_missing_keys: Optional[Dict[str, str]] = None) -> int:
        """
        Fix missing embedding files by regenerating them.
        
        Args:
            texts_for_missing_keys: Optional mapping of cache_key -> original_text
                                  If not provided, missing entries will be removed from metadata
                                  
        Returns:
            Number of embeddings fixed/removed
        """
        health_info = self.check_cache_health()
        missing_keys = health_info["missing_cache_keys"]
        
        if not missing_keys:
            logger.info("No missing embeddings found")
            return 0
            
        fixed_count = 0
        
        if texts_for_missing_keys:
            # Regenerate embeddings for missing files
            texts_to_regenerate = []
            keys_to_regenerate = []
            
            for key in missing_keys:
                if key in texts_for_missing_keys:
                    texts_to_regenerate.append(texts_for_missing_keys[key])
                    keys_to_regenerate.append(key)
                else:
                    # Remove from metadata if we can't regenerate
                    del self.metadata["text_to_path"][key]
                    fixed_count += 1
                    
            if texts_to_regenerate:
                logger.info(f"Regenerating {len(texts_to_regenerate)} missing embeddings")
                
                # Process in batches
                batches = [
                    texts_to_regenerate[i:i + self.batch_size]
                    for i in range(0, len(texts_to_regenerate), self.batch_size)
                ]
                
                batch_keys = [
                    keys_to_regenerate[i:i + self.batch_size]
                    for i in range(0, len(keys_to_regenerate), self.batch_size)
                ]
                
                for batch_texts, batch_keys_subset in zip(batches, batch_keys):
                    batch_embeddings_list = self._encode_texts(batch_texts)
                    
                    for embedding_dict in batch_embeddings_list:
                        # Assuming one key-value pair
                        study_id, embedding = list(embedding_dict.items())[0]
                        self._save_embedding(study_id, embedding)
                        fixed_count += 1
        else:
            # Just remove missing entries from metadata
            for key in missing_keys:
                del self.metadata["text_to_path"][key]
                fixed_count += 1
                
        self._save_metadata()
        logger.info(f"Fixed {fixed_count} missing embedding entries")
        return fixed_count
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        health_info = self.check_cache_health()
        
        return {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
            "cached_count": len(self.metadata["text_to_path"]),
            "existing_files": health_info["existing_files"],
            "missing_files": health_info["missing_files"],
            "health_percentage": health_info["health_percentage"],
            "embedding_dim": self.metadata["embedding_dim"],
            "total_cache_size_mb": sum(
                f.stat().st_size for f in self.embeddings_dir.glob("*.pt")
                if f.is_file()
            ) / (1024 * 1024)
        }
        
    def clear_cache(self):
        """Clear the entire cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "model_name": self.model_name,
            "normalize_embeddings": self.normalize_embeddings,
            "max_length": self.max_length,
            "text_to_path": {},
            "embedding_dim": None,
            "version": "1.0",
            "dataset_name": None,
            "split": None
        }
        self._save_metadata()


def process_texts_distributed(
    texts: List[Dict[str, Any]],
    model_name: str,
    cache_dir: str,
    rank: int,
    world_size: int,
    barrier,
    results_queue: mp.Queue,
    **kwargs
) -> Tuple[int, int]:
    """
    Process texts in a distributed manner.
    
    Args:
        texts: List of text dictionaries to process
        model_name: Hugging Face model name
        cache_dir: Cache directory
        rank: Process rank
        world_size: Total number of processes
        barrier: Multiprocessing barrier for synchronization
        results_queue: Queue for returning results
        **kwargs: Additional arguments for CachedTextEmbedding
        
    Returns:
        Tuple of (rank, number_of_processed_texts)
    """
    try:
        # Calculate chunk for this process
        chunk_size = len(texts) // world_size
        start_idx = rank * chunk_size
        if rank == world_size - 1:
            end_idx = len(texts)  # Last process gets remaining texts
        else:
            end_idx = start_idx + chunk_size
            
        process_texts = texts[start_idx:end_idx]
        
        # Determine device for this process
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = f"cuda:{rank % torch.cuda.device_count()}"
            torch.cuda.set_device(rank % torch.cuda.device_count())
        else:
            device = "cpu"
            
        logger.info(f"Rank {rank}: Processing {len(process_texts)} texts on {device}")
        
        # Initialize embedding generator for this process
        embedder = CachedTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            **kwargs
        )
        
        # Wait for all processes to be ready
        barrier.wait()
        
        # Process texts and cache them
        embedder.cache_texts_batch(process_texts, show_progress=(rank == 0), update_metadata=False)
        
        # Wait for all processes to finish processing
        barrier.wait()
        
        # Send results back to parent
        results_queue.put((rank, len(process_texts)))
        
        logger.info(f"Rank {rank}: Successfully processed {len(process_texts)} texts")
        return rank, len(process_texts)
        
    except Exception as e:
        logger.error(f"Rank {rank}: Error during processing: {e}")
        # Still need to participate in barriers to avoid deadlock
        try:
            barrier.wait()  # First barrier
            barrier.wait()  # Second barrier
        except:
            pass
        # Send error result
        results_queue.put((rank, 0))
        return rank, 0


def cache_captions_parallel(
    captions: List[Dict[str, Any]],
    model_name: str,
    cache_dir: str = "./embeddings_cache",
    num_processes: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Cache captions using parallel processing.
    
    Args:
        captions: List of caption dictionaries to cache
        model_name: Hugging Face model name
        cache_dir: Cache directory
        num_processes: Number of processes (defaults to GPU count or CPU count)
        **kwargs: Additional arguments for CachedTextEmbedding
        
    Returns:
        Dictionary with processing statistics
    """
    if num_processes is None:
        if torch.cuda.is_available():
            num_processes = torch.cuda.device_count()
        else:
            num_processes = min(mp.cpu_count(), 8)  # Reasonable default
    
    # Ensure num_processes is an int for type safety
    num_processes = int(num_processes)
    
    # Pop device from kwargs to handle it manually for parallel processing
    kwargs.pop('device', None)
        
    logger.info(f"Caching {len(captions)} captions with {num_processes} processes")
    
    if num_processes == 1:
        # Single process fallback
        embedder = CachedTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            device="auto",
            **kwargs
        )
        embedder.cache_texts_batch(captions, show_progress=True)
        return {
            "total_captions": len(captions),
            "total_processed": len(captions),
            "num_processes": 1,
            "model_name": model_name,
            "cache_dir": cache_dir
        }

    # Multi-process distributed processing
    mp.set_start_method('spawn', force=True)
    barrier = mp.Barrier(num_processes)
    results_queue = mp.Queue()
    processes = []
    
    try:
        # Start all worker processes
        for rank in range(num_processes):
            p = mp.Process(
                target=process_texts_distributed,
                args=(captions, model_name, cache_dir, rank, num_processes, barrier, results_queue),
                kwargs=kwargs
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete and collect results
        total_processed = 0
        completed_ranks = set()
        
        # Collect results with timeout
        for _ in range(num_processes):
            try:
                rank, processed_count = results_queue.get(timeout=6000)  # 100 minute timeout
                total_processed += processed_count
                completed_ranks.add(rank)
                logger.info(f"Process {rank} completed, processed {processed_count} texts")
            except:
                logger.error("Timeout waiting for worker process results")
                break
        
        # Wait for all processes to actually terminate
        for i, p in enumerate(processes):
            p.join(timeout=30)  # 30 second timeout per process
            if p.is_alive():
                logger.warning(f"Process {i} still alive, terminating...")
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    logger.error(f"Process {i} could not be terminated, killing...")
                    p.kill()
                    p.join()
        
        # Verify all processes completed successfully
        if len(completed_ranks) != num_processes:
            logger.warning(f"Only {len(completed_ranks)}/{num_processes} processes completed successfully")
        
        # Rebuild metadata from cache files after distributed processing
        if num_processes > 1:
            logger.info("Rebuilding metadata from distributed cache files...")
            embedder = CachedTextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                device="auto",
                **kwargs
            )
            embedder.rebuild_metadata_from_cache()
            
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        # Clean up any remaining processes
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
        raise
            
    return {
        "total_captions": len(captions),
        "total_processed": total_processed,
        "num_processes": num_processes,
        "model_name": model_name,
        "cache_dir": cache_dir
    }



if __name__ == "__main__":
    # Example usage
    import argparse
    from pathlib import Path
    
    # Optional pandas import
    try:
        import pandas as pd
        PANDAS_AVAILABLE = True
    except ImportError:
        PANDAS_AVAILABLE = False
    
    parser = argparse.ArgumentParser(
        description="Cache text embeddings with lazy loading and Hugging Face Hub integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with CSV file
  python -m miniclip.cached_text_embedding \\
    --data-file data.csv \\
    --model-name sentence-transformers/all-MiniLM-L6-v2 \\
    --dataset-name medical-reports \\
    --split train

  # With Hugging Face Hub upload
  python -m miniclip.cached_text_embedding \\
    --data-file data.csv \\
    --model-name sentence-transformers/all-MiniLM-L6-v2 \\
    --dataset-name medical-reports \\
    --split train \\
    --upload-to-hub \\
    --repo-id myuser/medical-embeddings \\
    --private

  # Process multiple splits
  for split in train test val; do
    python -m miniclip.cached_text_embedding \\
      --data-file data_$split.csv \\
      --model-name sentence-transformers/all-MiniLM-L6-v2 \\
      --dataset-name medical-reports \\
      --split $split \\
      --upload-to-hub \\
      --repo-id myuser/medical-embeddings
  done
"""
    )
    
    # Data input arguments
    parser.add_argument(
        "--data-file", 
        required=True, 
        help="File containing data (CSV, JSON, or text file with one item per line)"
    )
    parser.add_argument(
        "--text-column", 
        default="text", 
        help="Column name containing text data (for CSV/JSON files)"
    )
    parser.add_argument(
        "--study-id-column", 
        default="study_id", 
        help="Column name containing study IDs (for CSV/JSON files)"
    )
    parser.add_argument(
        "--findings-column", 
        default="category_findings", 
        help="Column name containing findings dictionary (for CSV/JSON files)"
    )
    
    # Model and processing arguments
    parser.add_argument(
        "--model-name", 
        required=True, 
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--dataset-name", 
        required=True, 
        help="Dataset name for organizing embeddings"
    )
    parser.add_argument(
        "--split", 
        required=True, 
        choices=["train", "test", "val", "validation", "dev"],
        help="Data split (train/test/val)"
    )
    parser.add_argument(
        "--cache-dir", 
        default="./embeddings_cache", 
        help="Base cache directory"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num-processes", 
        type=int, 
        help="Number of processes for parallel processing"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--no-normalize", 
        action="store_true", 
        help="Don't normalize embeddings"
    )
    parser.add_argument(
        "--categories", 
        nargs="+", 
        default=["all_findings"],
        help="Categories to process from findings"
    )
    
    # Hugging Face Hub arguments
    parser.add_argument(
        "--upload-to-hub", 
        action="store_true", 
        help="Upload cache to Hugging Face Hub after processing"
    )
    parser.add_argument(
        "--repo-id", 
        help="Repository ID on Hugging Face Hub (e.g., 'username/dataset-embeddings')"
    )
    parser.add_argument(
        "--revision", 
        help="Revision name on Hugging Face Hub (e.g., 'dataset-embeddings')"
    )
    parser.add_argument(
        "--token", 
        help="Hugging Face token (if not logged in via CLI)"
    )
    parser.add_argument(
        "--private", 
        action="store_true", 
        help="Create a private repository"
    )
    parser.add_argument(
        "--commit-message", 
        help="Custom commit message for Hub upload"
    )
    
    # Utility arguments
    parser.add_argument(
        "--check-health", 
        action="store_true", 
        help="Check cache health"
    )
    parser.add_argument(
        "--fix-missing", 
        action="store_true", 
        help="Fix missing embedding files"
    )
    parser.add_argument(
        "--force-regenerate", 
        action="store_true", 
        help="Force regeneration of existing embeddings"
    )
    
    args = parser.parse_args()
    
    # Normalize split name
    split_map = {"validation": "val", "dev": "val"}
    split_name = split_map.get(args.split, args.split)
    
    # Create organized cache directory structure
    cache_dir = Path(args.cache_dir) / args.dataset_name / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Cache directory: {cache_dir}")
    print(f"📊 Dataset: {args.dataset_name}")
    print(f"🔀 Split: {split_name}")
    print(f"🤖 Model: {args.model_name}")
    
    # Initialize embedder
    embedder = CachedTextEmbedding(
        model_name=args.model_name,
        cache_dir=str(cache_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize_embeddings=not args.no_normalize,
        categories=args.categories
    )
    
    # Update metadata with dataset and split information
    embedder.metadata["dataset_name"] = args.dataset_name
    embedder.metadata["split"] = split_name
    embedder._save_metadata()
    
    if args.check_health:
        health_info = embedder.check_cache_health()
        print("\n🏥 Cache Health Report:")
        print("-" * 40)
        for key, value in health_info.items():
            print(f"  {key}: {value}")
        
        if health_info["missing_files"] > 0 and args.fix_missing:
            print(f"\n🔧 Fixing {health_info['missing_files']} missing files...")
            fixed_count = embedder.fix_missing_embeddings()
            print(f"✅ Fixed {fixed_count} entries")
        exit()
    
    # Load data
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Error: Data file not found: {data_file}")
        exit(1)
    
    print(f"📖 Loading data from: {data_file}")
    
    # Load data based on file type
    if data_file.suffix.lower() == '.csv':
        if not PANDAS_AVAILABLE:
            print("❌ Error: pandas is required for CSV files. Install with: pip install pandas")
            exit(1)
        
        df = pd.read_csv(data_file)
        print(f"📊 Loaded {len(df)} rows from CSV")
        
        # Convert DataFrame to list of dictionaries
        data = []
        for _, row in df.iterrows():
            item = {
                'study_id': row.get(args.study_id_column),
                'category_findings': row.get(args.findings_column, {})
            }
            data.append(item)
            
    elif data_file.suffix.lower() == '.json':
        import json
        with open(data_file, 'r') as f:
            data = json.load(f)
        print(f"📊 Loaded {len(data)} items from JSON")
        
    else:
        # Assume text file with one item per line
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Convert to expected format
        data = []
        for i, line in enumerate(lines):
            item = {
                'study_id': i,
                'category_findings': {'all_findings': line}
            }
            data.append(item)
        print(f"📊 Loaded {len(data)} lines from text file")
    
    if not data:
        print("❌ Error: No data loaded")
        exit(1)
    
    # Cache embeddings
    print(f"\n🚀 Processing {len(data)} items...")
    
    if args.num_processes and args.num_processes > 1:
        print(f"🔄 Using {args.num_processes} processes for parallel processing")
        stats = cache_captions_parallel(
            captions=data,  # Note: This needs to be updated for the new data format
            model_name=args.model_name,
            cache_dir=str(cache_dir),
            num_processes=args.num_processes,
            batch_size=args.batch_size,
            max_length=args.max_length,
            normalize_embeddings=not args.no_normalize,
            categories=args.categories
        )
        print("✅ Parallel processing completed!")
        print(f"📈 Statistics: {stats}")
    else:
        print("🔄 Caching embeddings...")
        paths = embedder.cache_texts_batch(
            data, 
            show_progress=True, 
            force_regenerate=args.force_regenerate
        )
        valid_paths = [p for p in paths if p]
        print(f"✅ Cached {len(valid_paths)} embeddings")
    

    # make a new embedder with the same model and cache dir
    embedder = CachedTextEmbedding(
        model_name=args.model_name,
        cache_dir=str(cache_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    embedder.rebuild_metadata_from_cache()


    # get up to date metadata
    # embedder._load_metadata()
    # Show cache stats
    cache_stats = embedder.get_cache_stats()
    print("\n📊 Cache Statistics:")
    print("-" * 40)
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    # Upload to Hugging Face Hub if requested
    if args.upload_to_hub:
        if not args.repo_id:
            print("❌ Error: --repo-id is required when uploading to Hub")
            exit(1)
        
        try:
            # Import Hub utilities
            # from .hf_cache_utils import upload_cache_to_hub
            
            print(f"\n📤 Uploading to Hugging Face Hub: {args.repo_id}")
            
            # Create repo ID with dataset and split info
            # repo_id = f"{args.repo_id}-{args.dataset_name}-{split_name}"
            
            url = upload_cache_to_hub(
                cache_path=str(embedder.model_cache_dir),
                repo_id=args.repo_id,
                compress=True,  # Enable compression
                revision=args.revision,
                private=args.private,
                commit_message=args.commit_message or f"Add {args.dataset_name} {split_name} embeddings for {args.model_name}"
            )
            
            print(f"✅ Successfully uploaded compressed cache to: {url}")
            print(f"\n📖 Usage instructions:")
            print("```python")
            print("from miniclip.hf_cache_utils import create_embedder_from_hub")
            print(f"embedder = create_embedder_from_hub('{args.repo_id}')")
            print("embedding = embedder.load_embedding(your_data_item)")
            print("```")
            
        except ImportError as e:
            print(f"❌ Error: Hugging Face Hub not available: {e}")
            print("Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"❌ Error uploading to Hub: {e}")
            
    # Add compression argument to CLI
    parser.add_argument(
        "--compress-upload", 
        action="store_true",
        default=True,
        help="Compress cache before uploading to Hub (default: True)"
    )
    
    # Example of lazy loading
    print(f"\n🔍 Example: Loading first item embedding...")
    if data:
        embedding = embedder.load_embedding(data[0])
        if embedding is not None:
            # Get first category embedding
            first_category = list(embedding.keys())[0]
            first_embedding = embedding[first_category]
            print(f"  📐 Loaded embedding shape: {first_embedding.shape}")
            print(f"  📁 Embedding path: {embedder.get_embedding_path(data[0])}")
            print(f"  🏷️  Categories: {list(embedding.keys())}")
        else:
            print("  ❌ No embedding found for first item") 



