import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import MultiDatasetDataloader
from .distributed import is_master
from .precision import get_autocast
from .utils import AverageMeter, compute_weight_norms, compute_gradient_norms

try:
    import wandb
except ImportError:
    wandb = None


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a model from DDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module
    return model


def train_textdino_one_epoch(
    model: torch.nn.Module,
    data: Dict[str, Any],
    loss: torch.nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    scheduler: Any,
    args: Any,
):
    """
    Trains TextDino model for one epoch with gradient accumulation support.
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = torch.bfloat16 if 'bfloat16' in args.precision else torch.float32

    model.train()
    
    # Handle both standard and MultiDatasetDataloader
    if isinstance(data["train"], MultiDatasetDataloader):
        for dl in data["train"].dataloaders.values():
            if hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch)
        dataloader = data["train"]
        num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    else:
        dataloader = data["train"].dataloader
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        num_batches_per_epoch = len(dataloader) // args.accum_freq
    
    # Get the actual number of samples - for MultiDatasetDataloader this needs special handling
    if hasattr(dataloader, 'num_samples'):
        num_samples = dataloader.num_samples
    elif hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
        num_samples = len(dataloader.dataset)
    else:
        # For MultiDatasetDataloader, estimate from batch_size * num_batches
        num_samples = args.batch_size * len(dataloader) * args.world_size
    
    # For MultiDatasetDataloader, the actual number of batches is what matters
    # The dataloader will iterate exactly len(dataloader) times
    total_batches_in_epoch = len(dataloader)
    total_samples_in_epoch = total_batches_in_epoch * args.batch_size * args.world_size
    
    # Debug logging for first epoch
    if epoch == 0 and is_master(args):
        logging.info(f"Dataloader info: len(dataloader)={len(dataloader)}, "
                    f"dataset_size={num_samples}, total_samples_in_epoch={total_samples_in_epoch}, "
                    f"batch_size={args.batch_size}, world_size={args.world_size}, accum_freq={args.accum_freq}")
        if isinstance(data["train"], MultiDatasetDataloader):
            logging.info("Using MultiDatasetDataloader")
            for modal, dl_info in data["train"].dataloaders.items():
                logging.info(f"  Modal {modal}: num_samples={dl_info.dataloader.num_samples}, "
                           f"num_batches={dl_info.dataloader.num_batches}")
    
    sample_digits = math.ceil(math.log(total_samples_in_epoch + 1, 10))
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    teacher_entropy_m = AverageMeter()
    student_entropy_m = AverageMeter()
    
    optimizer.zero_grad()
    end = time.time()
    
    for i, multimodal_batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + (i // args.accum_freq)
        data_time_m.update(time.time() - end)

        # Determine if we should sync gradients this step
        is_sync_step = ((i + 1) % args.accum_freq) == 0 or (i + 1) == len(dataloader)
        
        # For TextDino, we process one modality at a time (not accumulating across modalities)
        # since we need paired image-text for distillation
        modalities = list(multimodal_batch.keys())
        
        # Debug logging for first batch
        if i == 0 and is_master(args):
            logging.info(f"TextDino dataloader batch keys: {modalities}")
        
        for modality in modalities:
            sync_context = model.no_sync if args.distributed and not is_sync_step else torch.enable_grad
            
            with sync_context():
                # Extract data for this modality
                batch = multimodal_batch[modality]
                
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["images"]
                    # For cached embeddings, look for common keys
                    texts = batch.get("texts", batch.get("tokens", batch.get("cached_embeddings", batch.get("text_embeddings"))))
                    
                    # Debug logging on first batch
                    if i == 0 and is_master(args):
                        logging.info(f"TextDino batch keys for {modality}: {list(batch.keys())}")
                        if texts is not None:
                            logging.info(f"Text tensor shape: {texts.shape if hasattr(texts, 'shape') else type(texts)}")
                elif isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        images, texts, info = batch
                    elif len(batch) == 2:
                        images, texts = batch
                    else:
                        raise ValueError(f"Unexpected batch format: {len(batch)} elements")
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")
                
                # Prepare inputs
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                
                # Create image dict for multimodal encoder (expects {modality: tensor})
                images_dict = {modality: images}
                
                # Handle text/cached embeddings
                if texts is None:
                    raise ValueError(f"No text data found in batch for {modality}. Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'non-dict batch'}")
                
                # Handle text embeddings - either single tensor or dict of tensors
                if isinstance(texts, dict):
                    # Multiple text embeddings - treat each as a separate teacher
                    texts_dict = {k: v.to(device=device, non_blocking=True) for k, v in texts.items()}
                    
                    # Debug logging
                    if i == 0 and modality == list(modalities)[0] and is_master(args):
                        logging.info(f"TextDino using multiple teachers: {list(texts_dict.keys())}")
                        for k, v in texts_dict.items():
                            logging.info(f"  {k}: shape {v.shape}")
                        logging.info(f"  Using equal weighting (simple average)")
                    
                    # Process each text embedding as a separate teacher
                    loss_components = {}
                    
                    for text_type, text_embedding in texts_dict.items():
                        with autocast():
                            # Forward pass with single text embedding
                            outputs = model(image=images_dict, text=text_embedding)
                        
                        # Extract outputs
                        teacher_logits = outputs["teacher_logits"]
                        student_logits = outputs["student_logits"]
                        center = outputs["center"]
                        teacher_temp = outputs["teacher_temp"]
                        student_temp = outputs["student_temp"]
                        
                        # Calculate loss for this teacher
                        loss_dict = loss(
                            teacher_logits=teacher_logits,
                            student_logits=student_logits,
                            center=center,
                            teacher_temp=teacher_temp,
                            student_temp=student_temp,
                            output_dict=True,
                        )
                        
                        loss_components[text_type] = loss_dict
                    
                    # Simple average of all teachers
                    total_loss_for_modality = sum(d["distillation_loss"] for d in loss_components.values()) / len(loss_components)
                    teacher_entropy = sum(d["teacher_entropy"] for d in loss_components.values()) / len(loss_components)
                    student_entropy = sum(d["student_entropy"] for d in loss_components.values()) / len(loss_components)
                    
                    # Use aggregated loss
                    total_loss = total_loss_for_modality / args.accum_freq
                    # Create aggregated loss dict
                    loss_dict = {
                        "distillation_loss": total_loss_for_modality,
                        "teacher_entropy": teacher_entropy,
                        "student_entropy": student_entropy,
                    }
                    
                elif hasattr(texts, 'to'):
                    # Single text embedding
                    texts = texts.to(device=device, non_blocking=True)
                    
                    with autocast():
                        # Forward pass
                        outputs = model(image=images_dict, text=texts)
                    
                    # Extract outputs
                    teacher_logits = outputs["teacher_logits"]
                    student_logits = outputs["student_logits"]
                    center = outputs["center"]
                    teacher_temp = outputs["teacher_temp"]
                    student_temp = outputs["student_temp"]
                    
                    # Calculate loss
                    loss_dict = loss(
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        center=center,
                        teacher_temp=teacher_temp,
                        student_temp=student_temp,
                        output_dict=True,
                    )
                    
                    total_loss = loss_dict["distillation_loss"] / args.accum_freq
                else:
                    raise ValueError(f"Unexpected text type: {type(texts)}")

            # Backward pass
            if args.precision == "amp":
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

        # Update metrics
        losses_m.update(loss_dict["distillation_loss"].item(), n=images.size(0))
        teacher_entropy_m.update(loss_dict["teacher_entropy"].item(), n=images.size(0))
        student_entropy_m.update(loss_dict["student_entropy"].item(), n=images.size(0))

        # Optimizer step only on sync steps
        if is_sync_step:
            # Track gradient norms BEFORE optimizer step
            grad_norm_info = None
            if args.track_gradient_norms:
                grad_norm_info = compute_gradient_norms(unwrap_model(model), args.norm_type)
            
            if args.precision == "amp":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Track weight norms after optimizer.step()
            weight_norm_info = None
            if args.track_weight_norms:
                weight_norm_info = compute_weight_norms(unwrap_model(model), args.norm_type)
                
            # Update learning rate
            if scheduler is not None:
                scheduler(step)

        batch_time_m.update(time.time() - end)
        end = time.time()

        # Logging
        if is_master(args) and ((i + 1) % args.log_every_n_steps == 0 or (i + 1) == len(dataloader)):
            samples_per_step = args.batch_size * args.world_size * args.accum_freq
            
            # Calculate actual samples processed
            samples_processed = (i + 1) * args.batch_size * args.world_size
            # Calculate percentage based on actual dataloader progress
            percent_complete = 100.0 * (i + 1) / len(dataloader)
            log_msg = (
                f"Train Epoch: {epoch} [{samples_processed:>{sample_digits}}"
                f"/{total_samples_in_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {losses_m.val:.4f} ({losses_m.avg:.4f}) "
                f"Teacher Entropy: {teacher_entropy_m.avg:.4f} "
                f"Student Entropy: {student_entropy_m.avg:.4f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_step / batch_time_m.val:.0f}/s "
                f"LR: {optimizer.param_groups[0]['lr']:.3e} "
            )
            
            # Add norm tracking info to log message if it was calculated this step
            if is_sync_step:
                if args.track_weight_norms and weight_norm_info:
                    log_msg += f"Weight Norm: {weight_norm_info['total_weight_norm']:.3f} "
                    
                if args.track_gradient_norms and grad_norm_info:
                    log_msg += f"Grad Norm: {grad_norm_info['total_grad_norm']:.3f} "
                    
            logging.info(log_msg)
            
            # Log to wandb if enabled
            if args.report_to == "wandb" and is_sync_step:
                log_dict = {
                    "train/loss": losses_m.avg,
                    "train/teacher_entropy": teacher_entropy_m.avg,
                    "train/student_entropy": student_entropy_m.avg,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/samples_per_second": samples_per_step / batch_time_m.val,
                    "train/steps_per_second": 1.0 / batch_time_m.val,
                    "train/data_time": data_time_m.avg,
                    "train/batch_time": batch_time_m.avg,
                }
                
                # Log individual teacher losses if available
                if 'loss_components' in locals() and loss_components:
                    for text_type, component in loss_components.items():
                        log_dict[f"train/teacher_{text_type}_loss"] = component["distillation_loss"].item()
                        log_dict[f"train/teacher_{text_type}_entropy"] = component["teacher_entropy"].item()
                
                # Add norm tracking to wandb
                if args.track_weight_norms and weight_norm_info:
                    log_dict["train/weight_norm"] = weight_norm_info["total_weight_norm"]
                    # Log fine-grained weight norms
                    for key, value in weight_norm_info.items():
                        if key.startswith("weight_norm/") and not key.startswith("weight_norm_raw/"):
                            log_dict[f"weight_norms/{key.replace('weight_norm/', '')}"] = value
                        
                if args.track_gradient_norms and grad_norm_info:
                    log_dict["train/gradient_norm"] = grad_norm_info["total_grad_norm"]
                    # Log fine-grained gradient norms
                    for key, value in grad_norm_info.items():
                        if key.startswith("grad_norm/") and not key.startswith("grad_norm_raw/"):
                            log_dict[f"gradient_norms/{key.replace('grad_norm/', '')}"] = value
                
                wandb.log(log_dict, step=step)


def evaluate_textdino(
    model: torch.nn.Module, 
    data: Dict[str, Any], 
    epoch: int, 
    args: Any, 
    tokenizer=None, 
    label=None
):
    """Evaluate TextDino on validation set."""
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = torch.bfloat16 if 'bfloat16' in args.precision else torch.float32

    model.eval()
    
    # Create loss function for evaluation
    from miniclip.factory import create_loss
    loss_fn = create_loss(args)

    # Handle both standard and MultiDatasetDataloader
    if isinstance(data["val"], MultiDatasetDataloader):
        dataloader = data["val"]
        # Log dataset info for MultiDatasetDataloader
        if is_master(args):
            logging.info("Validation using MultiDatasetDataloader:")
            for modal, dl_info in dataloader.dataloaders.items():
                logging.info(f"  Modal {modal}: {dl_info.dataloader.num_samples} samples, "
                           f"{dl_info.dataloader.num_batches} batches")
    else:
        dataloader = data["val"].dataloader
        if is_master(args):
            if hasattr(dataloader, 'num_samples'):
                logging.info(f"Validation dataset: {dataloader.num_samples} samples")
    
    losses_m = AverageMeter()
    teacher_entropy_m = AverageMeter()
    student_entropy_m = AverageMeter()
    batch_time_m = AverageMeter()
    
    # Get total number of batches for progress logging
    total_batches = len(dataloader)
    log_interval = max(1, total_batches // 10)  # Log 10 times during evaluation
    
    if is_master(args):
        logging.info(f"Starting evaluation with {total_batches} batches...")
    
    end = time.time()
    
    with torch.no_grad():
        for i, multimodal_batch in enumerate(dataloader):
            # Handle multimodal batches
            modalities = list(multimodal_batch.keys())
            
            for modality in modalities:
                # Extract data for this modality
                batch = multimodal_batch[modality]
                
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["images"]
                    # For cached embeddings, look for common keys
                    texts = batch.get("texts", batch.get("tokens", batch.get("cached_embeddings", batch.get("text_embeddings"))))
                    
                    # Debug logging on first batch
                    if i == 0 and is_master(args):
                        logging.info(f"TextDino batch keys for {modality}: {list(batch.keys())}")
                        if texts is not None:
                            logging.info(f"Text tensor shape: {texts.shape if hasattr(texts, 'shape') else type(texts)}")
                elif isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        images, texts, info = batch
                    elif len(batch) == 2:
                        images, texts = batch
                    else:
                        raise ValueError(f"Unexpected batch format: {len(batch)} elements")
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")
                
                # Prepare inputs
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                
                # Create image dict for multimodal encoder (expects {modality: tensor})
                images_dict = {modality: images}
                
                # Handle text/cached embeddings
                if texts is None:
                    raise ValueError(f"No text data found in batch for {modality}. Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'non-dict batch'}")
                
                # Similar to training, handle multiple teachers
                if isinstance(texts, dict):
                    # Multiple text embeddings - evaluate on each
                    texts_dict = {k: v.to(device=device, non_blocking=True) for k, v in texts.items()}
                    
                    # Process each text embedding as a separate teacher
                    loss_components = {}
                    
                    for text_type, text_embedding in texts_dict.items():
                        with autocast():
                            # Forward pass with single text embedding
                            outputs = model(image=images_dict, text=text_embedding)
                        
                        # Extract outputs
                        teacher_logits = outputs["teacher_logits"]
                        student_logits = outputs["student_logits"]
                        center = outputs["center"]
                        teacher_temp = outputs["teacher_temp"]
                        student_temp = outputs["student_temp"]
                        
                        # Calculate loss for this teacher
                        loss_dict = loss_fn(
                            teacher_logits=teacher_logits,
                            student_logits=student_logits,
                            center=center,
                            teacher_temp=teacher_temp,
                            student_temp=student_temp,
                            output_dict=True,
                        )
                        
                        loss_components[text_type] = loss_dict
                    
                    # Simple average of all teachers
                    total_loss = sum(d["distillation_loss"] for d in loss_components.values()) / len(loss_components)
                    teacher_entropy = sum(d["teacher_entropy"] for d in loss_components.values()) / len(loss_components)
                    student_entropy = sum(d["student_entropy"] for d in loss_components.values()) / len(loss_components)
                    
                elif hasattr(texts, 'to'):
                    # Single text embedding
                    texts = texts.to(device=device, non_blocking=True)
                    
                    with autocast():
                        # Forward pass
                        outputs = model(image=images_dict, text=texts)
                    
                    # Extract outputs
                    teacher_logits = outputs["teacher_logits"]
                    student_logits = outputs["student_logits"]
                    center = outputs["center"]
                    teacher_temp = outputs["teacher_temp"]
                    student_temp = outputs["student_temp"]
                    
                    # Calculate loss
                    loss_dict = loss_fn(
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        center=center,
                        teacher_temp=teacher_temp,
                        student_temp=student_temp,
                        output_dict=True,
                    )
                    
                    total_loss = loss_dict["distillation_loss"]
                    teacher_entropy = loss_dict["teacher_entropy"]
                    student_entropy = loss_dict["student_entropy"]
                else:
                    raise ValueError(f"Unexpected text type: {type(texts)}")

                # Update meters
                losses_m.update(total_loss.item(), n=images.size(0))
                teacher_entropy_m.update(teacher_entropy.item(), n=images.size(0))
                student_entropy_m.update(student_entropy.item(), n=images.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            # Progress logging
            if is_master(args) and ((i + 1) % log_interval == 0 or (i + 1) == total_batches):
                percent_complete = 100.0 * (i + 1) / total_batches
                logging.info(
                    f"Eval Progress: [{i+1}/{total_batches} ({percent_complete:.0f}%)] "
                    f"Loss: {losses_m.avg:.4f} "
                    f"Teacher Entropy: {teacher_entropy_m.avg:.4f} "
                    f"Student Entropy: {student_entropy_m.avg:.4f} "
                    f"Batch Time: {batch_time_m.avg:.3f}s"
                )

    metrics = {
        "textdino_val_loss": losses_m.avg,
        "textdino_teacher_entropy": teacher_entropy_m.avg,
        "textdino_student_entropy": student_entropy_m.avg,
    }

    if is_master(args):
        total_samples = losses_m.count  # Total samples processed
        logging.info(
            f"Eval Epoch: {epoch} Complete - "
            f"Processed {total_samples} samples in {total_batches} batches"
        )
        logging.info(
            f"Eval Results: "
            f"Loss: {metrics['textdino_val_loss']:.4f} "
            f"Teacher Entropy: {metrics['textdino_teacher_entropy']:.4f} "
            f"Student Entropy: {metrics['textdino_student_entropy']:.4f} "
            f"Avg Batch Time: {batch_time_m.avg:.3f}s"
        )
        
        # Log to wandb if enabled
        if args.report_to == "wandb":
            wandb.log({
                "val/loss": metrics["textdino_val_loss"],
                "val/teacher_entropy": metrics["textdino_teacher_entropy"],
                "val/student_entropy": metrics["textdino_student_entropy"],
                "epoch": epoch,
            })

    return metrics