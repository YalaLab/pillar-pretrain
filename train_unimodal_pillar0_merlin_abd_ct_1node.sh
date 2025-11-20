#!/bin/bash

# Set environment variables to prevent hanging
export OMP_NUM_THREADS=2
export NCCL_TIMEOUT=300  # 30 min timeout for distributed ops

# Timestamp used to uniquely tag this run name
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

uv run python -m torch.distributed.run \
    --master_port 4242 \
    --master_addr 127.0.0.1  \
    --nproc_per_node gpu \
    -m trainer.main \
    --dataset-type multimodal \
    --multimodal-config src/miniclip/data_configs/pillar0_merlin_abd_ct_384.yaml \
    --accum-freq 8 \
    --val-frequency 1 \
    --save-frequency 1 \
    --save-frequency-step 50 \
    --save-most-recent \
    --warmup 300 \
    --epochs 50 \
    --log-every-n-steps 2 \
    --lr 2.5e-4 \
    --wd 0.0 \
    --precision amp_bf16 \
    --image-mean "0.289" \
    --image-std "0.198" \
    --aug-cfg med_3d_aug=True \
    --image-interpolation bilinear \
    --model "Qwen3-Embedding-8B-Atlas_small_d2d2d8_multiscale__pillar0_multimodal_ucsf_abd_ct_multiwindow" \
    --name "PILLAR0-atlas_small_qwen8b_clip_merlin_abd_ct_r384_multiwindow_imagenetpretrained__rc1_1node_${HOSTNAME}_${RUN_TIMESTAMP}" \
    --report-to "wandb" \
    --wandb-project-name "atlasvlm" \
    --resume latest \
    --window-type "all" \
    --workers 2 \
    --dataloader-timeout 300 \
    --multimodal-rebuild-on-exhaust \
    --max-val-batches 32 \
    --enable-gpu-rotation \
    --gpu-rotation-degrees 20 \
    --gpu-rotation-p 0.5 \
    --lock-text \
    --lock-text-unlocked-layers 10 \
    --pretrained-model "YalaLab/multimodal_atlas_d2d2d8_small_r1024_008" \
    --revision "epoch_300" \
    --imagenet-pretrained
