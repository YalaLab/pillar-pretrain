# Filename: train_multimodal.py

"""
Trainer for multimodal models, specifically tailored for SigLIP and compatible
with a modality-keyed MultiDatasetDataloader.
"""

import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import MultiDatasetDataloader

try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .precision import get_autocast
from .utils import AverageMeter, compute_weight_norms, compute_gradient_norms

DEBUG = False

log = logging.getLogger(__name__)

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwraps a model from DDP or other wrappers."""
    return model.module if hasattr(model, "module") else model


def get_clip_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: Optional[float] = None,
) -> Dict[str, float]:
    """Computes CLIP-style retrieval metrics from given features."""
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.T)
    if logit_bias is not None:
        logits_per_image += logit_bias
    logits_per_image = logits_per_image.cpu()
    logits_per_text = logits_per_image.T
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1].numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def train_multimodal_siglip_one_epoch(
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
    Trains the model for one epoch, supporting MultiDatasetDataloader with
    memory-efficient, DDP-correct gradient accumulation.
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
    
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = defaultdict(AverageMeter)
    
    optimizer.zero_grad()
    end = time.time()
    
    for i, multimodal_batch in enumerate(dataloader):

        if DEBUG and i > 32:
            break

        step = num_batches_per_epoch * epoch + (i // args.accum_freq)
        data_time_m.update(time.time() - end)

        # Outer accumulation check (across data loader steps)
        is_sync_step = ((i + 1) % args.accum_freq) == 0 or (i + 1) == len(dataloader)
        
        modalities = list(multimodal_batch.keys())
        info = None
        
        # Loop through modalities and use model.no_sync() to prevent premature gradient sync
        for mod_idx, modality in enumerate(modalities):
            
            # Inner accumulation check (across modalities within a batch)
            is_last_modality = (mod_idx == len(modalities) - 1)
            
            # We only want to sync gradients on the VERY last backward pass of an accumulation cycle.
            should_sync_now = is_sync_step and is_last_modality
            sync_context = model.no_sync if args.distributed and not should_sync_now else torch.enable_grad

            with sync_context():
                batch = multimodal_batch[modality]
                if len(batch) == 3:
                    images, texts, info = batch
                elif len(batch) == 2:
                    images, texts = batch
                else:
                    images, texts = batch["images"], batch["texts"]
                
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)

                images_dict = {modality: images}
                if isinstance(texts, dict):
                    texts = {
                        k: v.to(device=device, non_blocking=True)
                        for k, v in texts.items()
                    }
                else:
                    texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = (
                        model(images_dict, texts, info=info)
                        if info is not None
                        else model(images_dict, texts)
                    )
                    
                    image_features = model_out.pop("image_features")
                    text_features = model_out.pop("text_features")
                    logits = model_out.pop("logits")
                    # breakpoint()
                    
                    # # For SigLIP, we compute the contrastive loss directly from features
                    # loss_inputs = {
                    #     "image_features": image_features,
                    #     "text_features": text_features,
                    #     "logit_scale": model_out.pop("logit_scale"),
                    # }
                    # if "logit_bias" in model_out:
                    #     loss_inputs["logit_bias"] = model_out.pop("logit_bias")
                    
                    total_loss = loss(logits, output_dict=True)
                    
                    # Normalize loss by both number of modalities and accumulation frequency
                    total_loss = total_loss['contrastive_loss'] / (len(modalities) * args.accum_freq)
                    losses_m[f"loss_{modality}"].update(total_loss.item())

                if scaler:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

        # Optimizer step is performed ONLY after a full accumulation cycle
        if is_sync_step:
            # Track gradient norms BEFORE optimizer.zero_grad()
            grad_norm_info = None
            if args.track_gradient_norms:
                grad_norm_info = compute_gradient_norms(unwrap_model(model), args.norm_type)
            
            if scaler:
                if args.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            if not args.skip_scheduler:
                scheduler(step)
            
            # Track weight norms after optimizer.step()
            weight_norm_info = None
            if args.track_weight_norms:
                with torch.no_grad():
                    weight_norm_info = compute_weight_norms(unwrap_model(model), args.norm_type)
        else:
            # Initialize to None for non-sync steps
            weight_norm_info = None
            grad_norm_info = None

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()

        if is_master(args) and is_sync_step and (i // args.accum_freq) % args.log_every_n_steps == 0:
            i_accum = i // args.accum_freq
            samples_so_far = (i_accum + 1) * args.batch_size * args.accum_freq * args.world_size
            loss_log = " | ".join([f"{name.replace('_', ' ').capitalize()}: {meter.avg:.4f}" for name, meter in losses_m.items()])
            
            log.info(
                f"Train Epoch: {epoch} [{samples_so_far:>{sample_digits}}/{dataloader.num_samples} "
                f"({100.0 * (i_accum + 1) / num_batches_per_epoch:.0f}%)] | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Scale: {unwrap_model(model).logit_scale.item():.3f} | {loss_log}"
            )

            if args.wandb:
                log_data = {
                    "epoch": epoch, "step": step,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/logit_scale": unwrap_model(model).logit_scale.item(),
                    **{f"train/{name}": meter.avg for name, meter in losses_m.items()}
                }
                
                # Add norm data to wandb logging
                if args.track_weight_norms and weight_norm_info:
                    log_data["train/weight_norm"] = weight_norm_info.get('total_weight_norm', weight_norm_info.get('total_norm', 0))
                    log_data["train/weight_norm_param_count"] = weight_norm_info.get('param_count', 0)
                    # Log fine-grained weight norms
                    for key, value in weight_norm_info.items():
                        if key.startswith('weight_norm/'):
                            log_data[f"train/{key}"] = value
                
                if args.track_gradient_norms and grad_norm_info:
                    log_data["train/grad_norm"] = grad_norm_info.get('total_grad_norm', grad_norm_info.get('total_norm', 0))
                    log_data["train/grad_norm_param_count"] = grad_norm_info.get('param_count', 0)
                    # Log fine-grained gradient norms
                    for key, value in grad_norm_info.items():
                        if key.startswith('grad_norm/'):
                            log_data[f"train/{key}"] = value
                
                wandb.log(log_data, step=step)


def evaluate_multimodal_siglip(model: torch.nn.Module, data: Dict[str, Any], epoch: int, args: Any, tokenizer=None, label=None):
    """
    Evaluates the model on the validation set, supporting MultiDatasetDataloader.
    """

    if 'val' not in data or not is_master(args):
        return {}

    log.info("Starting SigLIP validation with MultiDatasetDataloader support...")
    model.eval()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = torch.bfloat16 if 'bfloat16' in args.precision else torch.float32

    # Handle both standard and MultiDatasetDataloader for validation
    if isinstance(data["val"], MultiDatasetDataloader):
        dataloader = data["val"]
        samples_per_val = dataloader.modal_num_samples
    else:
        dataloader = data["val"].dataloader
        samples_per_val = dataloader.modal_num_samples if hasattr(dataloader, 'modal_num_samples') else {}

    # Store features and losses per modality
    all_image_features = defaultdict(list)
    all_text_features = defaultdict(list)
    cumulative_loss = defaultdict(float)
    num_samples = defaultdict(int)
    
    with torch.inference_mode():
        for i, multimodal_batch in enumerate(dataloader):
            info = None
            for modality, batch in multimodal_batch.items():
                # Handle different batch formats consistently with training
                if len(batch) == 3:
                    images, texts, info = batch
                elif len(batch) == 2:
                    images, texts = batch
                else:
                    images, texts = batch["images"], batch["texts"]
                
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                images_dict = {modality: images}
                
                if isinstance(texts, dict):
                    # For CustomTextCLIP
                    texts = {
                        k: v.to(device=device, non_blocking=True)
                        for k, v in texts.items()
                    }
                else:
                    texts = texts.to(device=device, non_blocking=True)
                
                with autocast():
                    model_out = (
                        model(images_dict, texts, info=info)
                        if info is not None
                        else model(images_dict, texts)
                    )
                
                img_feat = model_out.get("image_features")
                text_feat = model_out.get("text_features")
                logit_scale = model_out.get("logit_scale", 1.0)
                
                if img_feat is not None and text_feat is not None:
                    # Store features for retrieval metrics
                    all_image_features[modality].append(img_feat.cpu())
                    if isinstance(text_feat, dict):
                        all_text_features[modality].append({k: v.cpu() for k, v in text_feat.items()})
                    else:
                        all_text_features[modality].append(text_feat.cpu())
                    
                    # Compute validation loss manually (SigLIP-style)
                    batch_size = images.shape[0]
                    logits = model_out.get("logits")
                    if logits is not None:
                        # Handle case where logits is a dictionary (multiple text categories)
                        if isinstance(logits, dict):
                            total_loss = 0.0
                            for text_category, logit_tensor in logits.items():
                                labels = torch.eye(batch_size, device=device, dtype=logit_tensor.dtype)
                                category_loss = -torch.sum(F.logsigmoid(logit_tensor) * labels + F.logsigmoid(-logit_tensor) * (1 - labels)) / batch_size
                                total_loss += category_loss
                            total_loss = total_loss / len(logits)  # Average across categories
                        else:
                            # Single logits tensor
                            labels = torch.eye(batch_size, device=device, dtype=logits.dtype)
                            total_loss = -torch.sum(F.logsigmoid(logits) * labels + F.logsigmoid(-logits) * (1 - labels)) / batch_size
                        cumulative_loss[modality] += total_loss.item() * batch_size
                    num_samples[modality] += batch_size
                    
                    # Log progress
                    if is_master(args) and (i % 10) == 0:
                        avg_loss = cumulative_loss[modality] / num_samples[modality]
                        log.info(
                            f"Eval Epoch: {epoch} [{num_samples[modality]} / {samples_per_val.get(modality, 'N/A')}]\t"
                            f"{modality.upper()} SigLIP Loss: {avg_loss:.6f}"
                        )

    metrics = {}
    logit_scale_val = unwrap_model(model).logit_scale.item()
    logit_bias_val = unwrap_model(model).logit_bias.item() if hasattr(unwrap_model(model), 'logit_bias') else None

    # Calculate metrics for each modality found in the validation set
    for modality in all_image_features.keys():
        if not all_image_features[modality]: 
            continue

        log.info(f"Calculating metrics for modality: {modality}")
        img_feat_cat = torch.cat(all_image_features[modality], dim=0)
        
        # Handle dict of text features for CustomTextCLIP
        if isinstance(all_text_features[modality][0], dict):
            text_feat_cat = defaultdict(list)
            # This is a list of dicts. We want a dict of lists of tensors, then cat each list.
            for micro_batch_features in all_text_features[modality]:
                for key, text_tensor in micro_batch_features.items():
                    text_feat_cat[key].append(text_tensor)
            
            text_feat_cat = {key: torch.cat(tensor_list, dim=0) for key, tensor_list in text_feat_cat.items()}
        else:
            text_feat_cat = torch.cat(all_text_features[modality], dim=0)

        
        # Calculate per-modality validation loss
        if num_samples[modality] > 0:
            avg_loss = cumulative_loss[modality] / num_samples[modality]
            metrics[f"{modality}_siglip_val_loss"] = avg_loss
                
        # Add sample count for this modality
        metrics[f"{modality}_num_samples"] = num_samples[modality]

    if not metrics:
        log.warning("Validation complete, but no metrics could be calculated.")
        return {}

    # Add overall metrics
    metrics["epoch"] = epoch
    if all_image_features:
        total_samples = sum(num_samples.values())
        metrics["total_samples"] = total_samples

    log_str_list = [f"{k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)) and 'num_samples' not in k]
    log.info("Validation Metrics:\n" + "\n".join(log_str_list))

    if args.wandb:
        if "train" in data:
            if isinstance(data["train"], MultiDatasetDataloader):
                train_dataloader = data["train"]
            else:
                train_dataloader = data["train"].dataloader
            num_batches_per_epoch = train_dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        
        log_data = {"val/" + name: val for name, val in metrics.items()}
        log_data["epoch"] = epoch
        wandb.log(log_data, step=step)

    return metrics