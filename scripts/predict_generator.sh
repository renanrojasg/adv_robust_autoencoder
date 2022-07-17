#!/bin/sh
root="./.."
CUDA_VISIBLE_DEVICES=7 python3 ${root}/predict.py\
  --load_classifier ${root}/checkpoints/alexnet_inv/alexnet_ar.pt\
  --load_generator ${root}/checkpoints/alexnet_inv/alexnet_ar_px_ft_gan.pt\
  --data ${root}/datasets/ILSVRC2012/val\
  --out_dir ${root}/logs/predict_generator\
  --arch alexnet\
  --dataset imagenet\
  --upsample_mode conv5_tconv\
  --spectral_init\
  --output_layer conv5\
  --transform_test resize_norm\
  --num_workers 32\
  --batch_size 32\
  --compute_lpips\
  --compute_psnr\
  --compute_ssim\
  --lpips_net alex
