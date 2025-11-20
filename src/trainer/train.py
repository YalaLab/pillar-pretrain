import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
try:
    import wandb
except ImportError:
    wandb = None


from miniclip import CLIP, CustomTextCLIP, get_input_dtype

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


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    # dist_model,
    args,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_info, accum_features = [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        if DEBUG and i > 32:
            break
        # breakpoint()
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        info = None
        if len(batch) == 3:
            images, texts, info = batch
        elif len(batch) == 2:
            images, texts = batch
        else:
            images, texts = batch["images"], batch["texts"]
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)

        if isinstance(texts, dict):
            # For CustomTextCLIP
            texts = {
                k: v.to(device=device, non_blocking=True) for k, v in texts.items()
            }
        else:
            texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # breakpoint()
        if args.accum_freq == 1:
            with autocast():
                if info is not None:
                    model_out = model(images, texts, info=info)
                else:
                    model_out = model(images, texts)

                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    if info is not None:
                        model_out = model(images, texts, info=info)
                    else:
                        model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)
                if info is not None:
                    accum_info.append(info)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            # breakpoint()

            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    if len(accum_info) > 0:
                        info = accum_info[j]
                        model_out = model(images, texts, info=info)
                    else:
                        model_out = model(images, texts)
                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                        "logit_scale"
                    )
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        if type(accumulated[0]) is dict:
                            inputs_dict = {}
                            keylist = list(accumulated[0].keys())
                            for k in keylist:
                                accs = [acc[k] for acc in accumulated]
                                inputs_dict[k] = torch.cat(
                                    accs[:j] + [model_out[key][k]] + accs[j + 1 :]
                                )
                            inputs[key] = inputs_dict
                        else:
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

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
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

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_info, accum_features = [], [], [], {}

        # Track weight and gradient norms if enabled
        weight_norm_info = None
        grad_norm_info = None
        
        if args.track_weight_norms:
            with torch.no_grad():
                weight_norm_info = compute_weight_norms(unwrap_model(model), args.norm_type)
        
        if args.track_gradient_norms:
            grad_norm_info = compute_gradient_norms(unwrap_model(model), args.norm_type)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(images)
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

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                
                # Add norm data to wandb logging
                if args.track_weight_norms and weight_norm_info:
                    log_data["train/weight_norm"] = weight_norm_info['total_norm']
                    log_data["train/weight_norm_param_count"] = weight_norm_info['param_count']
                
                if args.track_gradient_norms and grad_norm_info:
                    log_data["train/grad_norm"] = grad_norm_info['total_norm']
                    log_data["train/grad_norm_param_count"] = grad_norm_info['param_count']
                
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None, label=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    # metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if "val" in data and (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        wandb_prefix = ""
        logging_prefix = ""
        if label != None:
            dataloader = data["val_labels"][label].dataloader
            logging_prefix = label.upper() + " "
            wandb_prefix = label + "_"
        else:
            dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        # all_image_features, all_text_features = [], []
        all_image_features = []
        all_text_features = None  # Can be a list or a dict of lists
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                info = None
                if len(batch) == 3:
                    images, texts, info = batch
                else:
                    images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                if isinstance(texts, dict):
                    # For CustomTextCLIP
                    texts = {
                        k: v.to(device=device, non_blocking=True)
                        for k, v in texts.items()
                    }
                else:
                    texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    if info is not None:
                        model_out = model(images, texts, info=info)
                    else:
                        model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    # all_text_features.append(text_features.cpu())
                    if isinstance(text_features, dict):
                        if all_text_features is None:
                            all_text_features = {k: [] for k in text_features.keys()}
                        for k, v in text_features.items():
                            all_text_features[k].append(v.cpu())
                    else:
                        if all_text_features is None:
                            all_text_features = []
                        all_text_features.append(text_features.cpu())

                    logit_scale = logit_scale.mean()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    # total_loss = (
                    #     F.cross_entropy(logits_per_image, labels)
                    #     + F.cross_entropy(logits_per_text, labels)
                    # ) / 2

                    if isinstance(text_features, dict):
                        total_loss = 0.0
                        # Average the loss across the different text features
                        for text_feat_tensor in text_features.values():
                            logits_per_image = (
                                logit_scale * image_features @ text_feat_tensor.t()
                            )
                            logits_per_text = logits_per_image.t()
                            total_loss += (
                                F.cross_entropy(logits_per_image, labels)
                                + F.cross_entropy(logits_per_text, labels)
                            ) / 2
                        total_loss /= len(text_features)
                    else:
                        logits_per_image = (
                            logit_scale * image_features @ text_features.t()
                        )
                        logits_per_text = logits_per_image.t()
                        total_loss = (
                            F.cross_entropy(logits_per_image, labels)
                            + F.cross_entropy(logits_per_text, labels)
                        ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"{logging_prefix}Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                    )

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t"
                        )

            image_features = torch.cat(all_image_features)
            if isinstance(all_text_features, dict):
                text_features = {k: torch.cat(v) for k, v in all_text_features.items()}
            else:
                text_features = torch.cat(all_text_features)

            val_metrics = get_clip_metrics(
                # image_features=torch.cat(all_image_features),
                # text_features=torch.cat(all_text_features),
                image_features=image_features,
                text_features=text_features,
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            val_metrics = {
                f"{wandb_prefix}{metric}": val_metrics[metric] for metric in val_metrics
            }
            metrics.update(
                {
                    **val_metrics,
                    f"{wandb_prefix}clip_val_loss": loss.item(),
                    "epoch": epoch,
                    "num_samples": num_samples,
                }
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, "Please install wandb."
        if "train" in data:
            dataloader = data["train"].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data["epoch"] = epoch
        wandb.log(log_data, step=step)
        print("logging", log_data)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    if isinstance(text_features, dict):
        metrics = {}
        for key, text_feature_tensor in text_features.items():
            # Recursively call with each text feature tensor
            sub_metrics = get_clip_metrics(
                image_features, text_feature_tensor, logit_scale
            )
            for metric_name, val in sub_metrics.items():
                metrics[f"{key}_{metric_name}"] = val
        return metrics

    # Base case: text_features is a single tensor
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
