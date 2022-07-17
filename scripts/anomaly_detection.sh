#!/bin/sh
root="./.."
CUDA_VISIBLE_DEVICES=7 python3 ${root}/anomaly_detection.py\
  --load_classifier ${root}/checkpoints/alexnet_inv/alexnet_ar.pt\
  --load_comparator ${root}/checkpoints/alexnet_inv/alexnet_ar.pt\
  --load_generator ${root}/checkpoints/alexnet_ad/alexnet_ar_cifar10_0.pt\
  --in_data_path ${root}/datasets/cifar\
  --out_dir ${root}/logs/anomaly_detection\
  --seed 16\
  --data ImageNet\
  --in_class_dataset cifar10\
  --upsample_mode conv5_tconv\
  --iterations 100\
  --sched_step 100\
  --step_size 0.1\
  --feature_loss_weight 0.01\
  --pixel_loss_weight 0.000002\
  --spectral_init
