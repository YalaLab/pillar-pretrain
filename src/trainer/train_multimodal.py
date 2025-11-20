import json
import logging
import math
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import rve

# Suppress warnings about non-writable tensors
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")

torch.autograd.set_detect_anomaly(True)
try:
    import wandb
except ImportError:
    wandb = None


from miniclip import get_input_dtype

from .data import MultiDatasetDataloader
from .distributed import is_master

from .precision import get_autocast
from .utils import AverageMeter, compute_weight_norms, compute_gradient_norms

DEBUG = False


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2],
    }


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def save_step_checkpoint(args, model, optimizer, scaler, epoch, step):
    """Save a step-level checkpoint."""
    save_frequency_step = getattr(args, "save_frequency_step", 0)
    if not args.save_logs or save_frequency_step <= 0:
        return
    
    if (step + 1) % save_frequency_step != 0:
        return
    
    checkpoint_dict = {
        "epoch": epoch,
        "step": step,
        "name": args.name,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()
    
    # Save step checkpoint
    step_path = os.path.join(args.checkpoint_path, f"step_{step}.pt")
    torch.save(checkpoint_dict, step_path)
    
    # Save as latest if requested
    if args.save_most_recent:
        latest_path = os.path.join(args.checkpoint_path, "step_latest.pt")
        torch.save(checkpoint_dict, latest_path)
    
    logging.info(f"Saved step checkpoint at step {step}")


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_multimodal_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    args,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    # breakpoint()

    model.train()

    if isinstance(data["train"], MultiDatasetDataloader):
        # If MultiDatasetDataloader, we need to set the epoch for each dataset
        for dl in data["train"].dataloaders.values():
            if hasattr(dl, "set_epoch"):
                dl.set_epoch(epoch)
        dataloader = data["train"]
        num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    else:
        # If not MultiDatasetDataloader, we can set the epoch directly
        if hasattr(data["train"], "set_epoch"):
            data["train"].set_epoch(epoch)
        dataloader = data["train"].dataloader
        num_batches_per_epoch = dataloader.num_batches // args.accum_freq

    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}
    modality_losses_m = defaultdict(lambda: AverageMeter())  # Track modality-specific losses
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # --- Accumulators using dictionaries to respect modality structure ---
    # For caching the original inputs needed for the gradient-enabled second pass
    accum_inputs = defaultdict(lambda: {"images": [], "texts": [], "info": []})
    # For caching the detached features from the first no_grad pass
    accum_features = defaultdict(lambda: {"image_features": [], "text_features": []})
    
    # Check if we're resuming from a step checkpoint
    resume_step = None
    step_in_epoch = 0  # Track position within current epoch
    if args.resume and os.path.exists(args.resume):
        try:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if "step" in checkpoint and checkpoint.get("epoch", -1) == epoch:
                resume_step = checkpoint["step"]  # This is the global step number
                
                # Calculate position within current epoch
                step_in_epoch = resume_step % num_batches_per_epoch
                
                logging.info(f"Resuming from global step {resume_step} (step {step_in_epoch} in epoch {epoch})")
                
                # Calculate remaining batches in this epoch
                if isinstance(dataloader, MultiDatasetDataloader):
                    # Each step processes args.accum_freq raw batches
                    completed_batches = step_in_epoch * args.accum_freq
                    remaining_batches = dataloader.num_batches - completed_batches
                    if remaining_batches > 0:
                        # Log modality status at resume point
                        modality_status = dataloader.get_modality_status_at_batch(completed_batches)
                        logging.info(f"Modality status at batch {completed_batches}:")
                        for modality, status in modality_status.items():
                            logging.info(f"  {modality}: {status}")
                        
                        dataloader.set_remaining_batches(remaining_batches)
                        logging.info(f"Set MultiDatasetDataloader to yield {remaining_batches} remaining batches")
                    else:
                        # Already completed this epoch
                        logging.info(f"Epoch {epoch} already completed (step_in_epoch={step_in_epoch}), skipping")
                        return {}  # Return empty metrics since epoch is complete
        except Exception:
            pass

    for i, multimodal_batch in enumerate(dataloader):
        if DEBUG and i > 17:
            break
        
        # Debug: Log batch iteration
        if i % 10 == 0:
            rank = args.rank if hasattr(args, 'rank') else 0
            logging.info(f"[Rank {rank}] Processing batch {i}, received {len(multimodal_batch)} modalities: {list(multimodal_batch.keys())}")
        
        data_time_m.update(time.time() - end)
        end = time.time()

        info = None
        optimizer.zero_grad()

        # --- Pass 1: Cache inputs and features into modality-keyed dictionaries ---
        # This pass runs with no_grad for efficiency.
        with torch.no_grad():
            with autocast():
                # breakpoint()
                for modality, batch in multimodal_batch.items():
                    # logging.info(f"Processing modality: {modality}")
                    # breakpoint()
                    if len(batch) == 3:
                        images, texts, info = batch
                    elif len(batch) == 2:
                        images, texts = batch
                    else:
                        images, texts = batch["images"], batch["texts"]
                    
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)

                    # Apply GPU-based 3D rotation augmentation if enabled
                    if getattr(args, 'enable_gpu_rotation', False) and model.training:
                        from trainer.gpu_augment import apply_3d_rotation_gpu
                        images = apply_3d_rotation_gpu(
                            images.to(torch.float32),
                            degrees=getattr(args, 'gpu_rotation_degrees', 10.0),
                            p=getattr(args, 'gpu_rotation_p', 0.5),
                            training=model.training
                        )
                    
                    images = images.to(
                        device=device, dtype=input_dtype, non_blocking=True
                    )
                    # Apply GPU transforms for Merlin CT data if configured
                    if modality in ['merlin_abd_ct', 'chest_ct', 'brain_ct', 'abdomen_ct'] and hasattr(args, 'window_type') and args.window_type is not None:
                        # from trainer.merlin_abd_ct import apply_merlin_gpu_transforms
                        # images = apply_merlin_gpu_transforms(
                        #     images, 
                        #     window_type=args.window_type,
                        #     modality='CT',
                        #     normalize_mean=getattr(args, 'image_mean', 0.289),
                        #     normalize_std=getattr(args, 'image_std', 0.198)
                        # )
                        images = rve.batch_apply_windowing_vectorized(
                            images,
                            windows=args.window_type,
                            modality='CT'
                        )
                    elif modality in ['breast_mr', 'prostate_mr']:
                        if modality == 'prostate_mr':
                            window_type = "znorm"
                        else:
                            window_type = "high_contrast"
                        from trainer.merlin_abd_ct import apply_merlin_gpu_transforms
                        images = apply_merlin_gpu_transforms(
                            images,
                            window_type=window_type,
                            modality='MR',
                            normalize_mean=getattr(args, 'image_mean', 0.289),
                            normalize_std=getattr(args, 'image_std', 0.198),
                            normalize=False
                        )

                    images_dict = {modality: images}
                    if isinstance(texts, dict):
                        # For CustomTextCLIP
                        texts = {
                            k: v.to(device=device, non_blocking=True)
                            for k, v in texts.items()
                        }
                    else:
                        texts = texts.to(device=device, non_blocking=True)

                    # logging.info("moving time: " + str(time.time() - end))   
                    # end = time.time()

                    # --- Cache inputs for the second pass ---
                    accum_inputs[modality]["images"].append(images_dict)
                    accum_inputs[modality]["texts"].append(texts)
                    if info is not None:
                        accum_inputs[modality]["info"].append(info)

                    
                    # --- Compute and cache features ---
                    model_out = (
                        model(images_dict, texts, info=info)
                        if info is not None
                        else model(images_dict, texts)
                    )
                    # logging.info("model time: " + str(time.time() - end))
                    # end = time.time()

                    # Store features as normal tensors (not inference tensors)
                    # to allow concatenation with grad-enabled tensors in pass 2
                    image_feats = model_out["image_features"]
                    text_feats = model_out["text_features"]
                    if isinstance(image_feats, dict):
                        image_feats = {k: v.detach().clone() for k, v in image_feats.items()}
                    else:
                        image_feats = image_feats.detach().clone()
                    if isinstance(text_feats, dict):
                        text_feats = {k: v.detach().clone() for k, v in text_feats.items()}
                    else:
                        text_feats = text_feats.detach().clone()

                    accum_features[modality]["image_features"].append(image_feats)
                    accum_features[modality]["text_features"].append(text_feats)

        # --- Accumulation Check ---
        # Continue accumulating batches until we reach the desired frequency.
        if ((i + 1) % args.accum_freq) != 0:
            continue

        # --- Pass 2: Re-compute all with gradients for a single, global loss ---
        # This block executes only once every `args.accum_freq` steps.
        i_accum = i // args.accum_freq
        
        # Calculate global step, accounting for resume
        if resume_step is not None:
            # When resuming, continue from the global step number
            # i_accum=0 corresponds to the next step after resume_step
            step = resume_step + 1 + i_accum
        else:
            step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        optimizer.zero_grad()

        # Re-run the forward pass for ALL cached inputs to build the computation graph
        with autocast():
            batch_sizes = {}
            for modality in accum_inputs:
                # Get the batch size for this modality
                batch_sizes[modality] = len(accum_inputs[modality]['images'][0][modality])

            # Iterate through each modality we've accumulated
            for modality, inputs_dict in accum_inputs.items():
                # Iterate through each micro-batch within that modality
                num_micro_batches = len(inputs_dict["images"])
                for j in range(num_micro_batches):
                    images = inputs_dict["images"][j]
                    texts = inputs_dict["texts"][j]
                    info = inputs_dict["info"][j] if inputs_dict["info"] else None

                    model_out = (
                        model(images, texts, info=info)
                        if info is not None
                        else model(images, texts)
                    )

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                        "logit_scale"
                    )
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features[modality].items():
                        accumulated = accum_features[modality][key]
                        if type(accumulated[0]) is dict:
                            # If the feature is a dict (e.g., for info), concatenate appropriately
                            captions_dict = {}
                            keylist = list(accumulated[0].keys())
                            for k in keylist:
                                accs = [acc[k] for acc in accumulated]
                                captions_dict[k] = torch.cat(
                                    accs[:j] + [model_out[key][k]] + accs[j + 1 :]
                                )
                            inputs[key] = captions_dict
                        else:
                            # Otherwise, concatenate the tensors directly
                            inputs[key] = torch.cat(
                                accumulated[:j]
                                + [model_out[key]]
                                + accumulated[j + 1 :]
                            )
                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                    
                    # Track modality-specific loss
                    modality_losses_m[modality].update(total_loss.item(), batch_sizes[modality])
                    
                    backward(total_loss, scaler)

        # logging.info("loss time: " + str(time.time() - end))   
        # end = time.time()

        # Track gradient norms BEFORE optimizer.zero_grad() or scaler operations
        grad_norm_info = None
        if args.track_gradient_norms:
            grad_norm_info = compute_gradient_norms(unwrap_model(model), args.norm_type)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # logging.info("optimizer time: " + str(time.time() - end))   
        # end = time.time()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_inputs = defaultdict(lambda: {"images": [], "texts": [], "info": []})
            # For caching the detached features from the first no_grad pass
            accum_features = defaultdict(
                lambda: {"image_features": [], "text_features": []}
            )

        # Track weight norms after optimizer.step()
        weight_norm_info = None
        if args.track_weight_norms:
            with torch.no_grad():
                weight_norm_info = compute_weight_norms(unwrap_model(model), args.norm_type)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        
        # Calculate batch count within current epoch
        if resume_step is not None:
            # When resuming, add the steps already completed in this epoch
            batch_count = step_in_epoch + 1 + i_accum
        else:
            batch_count = i_accum + 1
        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            # batch_size = batch_sizes[dataloader.shortest_modal] if hasattr(dataloader, 'shortest_modal') else len(next(iter(images.values())))
            batch_size = batch_sizes[dataloader.longest_modal] if hasattr(dataloader, 'longest_modal') else len(next(iter(images.values())))
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            
            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            
            # Add modality-specific losses to log
            if modality_losses_m:
                modality_log = " ".join(
                    [
                        f"{modality}: {loss_m.avg:#.5g}"
                        for modality, loss_m in modality_losses_m.items()
                    ]
                )
                loss_log += f" | Modality Losses: {modality_log}"
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})
            
            # Add modality-specific losses
            for modality, loss_m in modality_losses_m.items():
                log_data[f"{modality}_loss"] = loss_m.avg

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                
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

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        
        # Save step checkpoint
        save_step_checkpoint(args, model, optimizer, scaler, epoch, step)
    # end for


def _consolidate_features(feature_list):
    """Consolidate a list of features, handling both tensors and dicts of tensors."""
    if isinstance(feature_list[0], dict):
        # Handle a list of dictionaries of features
        text_features_dict = defaultdict(list)
        for features in feature_list:
            for key, val in features.items():
                text_features_dict[key].append(val)
        return {k: torch.cat(v, dim=0) for k, v in text_features_dict.items()}
    else:
        # Handle a list of tensors
        return torch.cat(feature_list, dim=0)


def _aggregate_metrics_ddp(cumulative_loss, num_samples, all_modalities, args, device):
    """Aggregate cumulative loss and sample counts across all GPUs."""
    if args.world_size <= 1:
        return cumulative_loss, num_samples, sorted(list(all_modalities))

    # Ensure all GPUs have the same master list of modalities to avoid deadlocks
    local_modalities = list(all_modalities)
    all_gathered_modalities = [None] * args.world_size
    dist.all_gather_object(all_gathered_modalities, local_modalities)
    master_modalities_list = sorted(list(set(m for sublist in all_gathered_modalities for m in sublist)))

    for modality in master_modalities_list:
        local_loss = cumulative_loss.get(modality, 0.0)
        local_samples = num_samples.get(modality, 0.0)
        
        # Create tensors for reduction
        local_loss_tensor = torch.tensor(local_loss, device=device)
        local_samples_tensor = torch.tensor(local_samples, device=device)

        # Sum across all GPUs
        dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_samples_tensor, op=dist.ReduceOp.SUM)

        # Update dictionaries on the master process
        if is_master(args):
            cumulative_loss[modality] = local_loss_tensor.item()
            num_samples[modality] = local_samples_tensor.item()

    return cumulative_loss, num_samples, master_modalities_list


def evaluate_multimodal(
    model, data, loss, epoch, args, tb_writer=None, tokenizer=None, label=None
):
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if "val" not in data or not (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        return metrics
    
    # Dataloader setup
    wandb_prefix = ""
    logging_prefix = ""
    if label is not None:
        logging_prefix = label.upper() + " "
        wandb_prefix = label + "_"
        val_source = data["val_labels"][label]
    else:
        val_source = data["val"]
    dataloader = val_source.dataloader if hasattr(val_source, 'dataloader') else val_source

    # Local accumulators for this GPU
    cumulative_loss = defaultdict(float)
    num_samples = defaultdict(float)
    all_modalities_in_run = set()

    # Configure loss for distributed evaluation
    loss.world_size = args.world_size
    loss.rank = args.rank
    
    with torch.no_grad():
        # Compute display batches considering optional cap
        total_batches = getattr(dataloader, 'num_batches', None)
        val_batches = total_batches
        if getattr(args, "max_val_batches", None) is not None and total_batches is not None:
            val_batches = min(total_batches, args.max_val_batches)
        if is_master(args):
            if val_batches is not None:
                logging.info(f"Eval Epoch: {epoch}, num batches: {val_batches}")
            else:
                logging.info(f"Eval Epoch: {epoch}")
        for i, multimodal_batch in enumerate(dataloader):
            
            # if DEBUG and i > 32:
            #     break

            with autocast():
                # Process each modality separately to get per-modality losses
                for modality, batch in multimodal_batch.items():
                    all_modalities_in_run.add(modality)
                    info = None
                    if len(batch) == 3:
                        images, texts, info = batch
                    else:
                        images, texts = batch

                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    if modality in ["merlin_abd_ct", "chest_ct", "brain_ct", "abdomen_ct"] and hasattr(args, "window_type") and args.window_type:
                        images = rve.batch_apply_windowing_vectorized(
                            images.to(torch.float32),
                            windows=args.window_type,
                            modality='CT'
                        )
                        # from trainer.merlin_abd_ct import apply_merlin_gpu_transforms
                        # images = apply_merlin_gpu_transforms(images, window_type=args.window_type, modality='CT', normalize_mean=getattr(args, 'image_mean', 0.289), normalize_std=getattr(args, 'image_std', 0.198))
                    elif modality in ['breast_mr', 'prostate_mr']:
                        from trainer.merlin_abd_ct import apply_merlin_gpu_transforms
                        if modality == 'prostate_mr':
                            window_type = "znorm"
                        else:
                            window_type = "high_contrast"
                        images = apply_merlin_gpu_transforms(
                            images,
                            window_type=window_type,
                            modality='MR',
                            normalize_mean=getattr(args, 'image_mean', 0.289),
                            normalize_std=getattr(args, 'image_std', 0.198),
                            normalize=False
                        )
                    
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = {k: v.to(device=device, non_blocking=True) for k, v in texts.items()} if isinstance(texts, dict) else texts.to(device=device, non_blocking=True)
                    model_out = model({modality: images}, texts, info=info)
                    
                    # Compute loss for this specific modality
                    losses = loss(
                        image_features=model_out["image_features"],
                        text_features=model_out["text_features"],
                        logit_scale=model_out["logit_scale"],
                        output_dict=True,
                    )
                    
                    # Update accumulators with modality-specific loss
                    total_loss = sum(losses.values())
                    bsz = images.shape[0]
                    cumulative_loss[modality] += total_loss.item() * bsz
                    num_samples[modality] += bsz

            if is_master(args) and (i + 1) % args.log_every_n_steps == 0:
                avg_loss = sum(cumulative_loss.values()) / (sum(num_samples.values()) + 1e-8)
                logging.info(f"Eval Epoch: {epoch} [Batch {i + 1}] Avg Loss: {avg_loss:.4f}")

            # Early stop evaluation if max-val-batches is set
            if getattr(args, "max_val_batches", None) is not None and (i + 1) >= args.max_val_batches:
                if is_master(args):
                    logging.info(f"Stopping eval early at {i + 1} batches due to --max-val-batches={args.max_val_batches}")
                break

    # Aggregate final metrics from all GPUs and get the master list of modalities
    cumulative_loss, num_samples, master_modalities_list = _aggregate_metrics_ddp(cumulative_loss, num_samples, all_modalities_in_run, args, device)

    # Master process calculates and logs final metrics
    if is_master(args):
        for modality in master_modalities_list:
            if num_samples.get(modality, 0) > 0:
                avg_loss = cumulative_loss[modality] / num_samples[modality]
                metrics[f"{wandb_prefix}{modality}_clip_val_loss"] = avg_loss
                metrics[f"{wandb_prefix}{modality}_num_samples"] = num_samples[modality]

        if not metrics:
            return metrics

        logging.info(f"Eval Epoch: {epoch} " + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

        log_data = {"val/" + name: val for name, val in metrics.items()}
        if args.save_logs:
            if tb_writer:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, epoch)
            with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, "Please install wandb."
            step = None
            if "train" in data:
                step = (data["train"].num_batches // args.accum_freq) * epoch
            log_data["epoch"] = epoch
            wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    if isinstance(text_features, dict):
        metrics = {}
        for key, text_feature_tensor in text_features.items():
            # Recursively call with each text feature tensor
            sub_metrics = get_clip_metrics(image_features, text_feature_tensor, logit_scale)
            for metric_name, val in sub_metrics.items():
                metrics[f"{key}_{metric_name}"] = val
        return metrics

    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)
 
    for name, logit in logits.items():
         ranking = torch.argsort(logit, descending=True)
         preds = torch.where(ranking == ground_truth)[1]
         preds = preds.detach().cpu().numpy()
         metrics[f"{name}_mean_rank"] = preds.mean() + 1
         metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
         for k in [1, 5, 10]:
             metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
