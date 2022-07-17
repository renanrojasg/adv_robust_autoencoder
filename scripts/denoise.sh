#!/bin/sh
root="./.."
CUDA_VISIBLE_DEVICES=7 python3 ${root}/predict.py\
  --dataset imagenet\
  --data ${root}/datasets/CBSD68\
  --load_classifier ${root}/checkpoints/alexnet_inv/alexnet_ar.pt\
  --load_generator ${root}/checkpoints/alexnet_den/alexnet_den_ar.pt\
  --out_dir ${root}/logs/denoise\
  --upsample_mode conv5_interp\
  --output_layer conv5\
  --spectral_init\
  --compute_psnr\
  --compute_ssim\
  --compute_lpips\
  --noise_level 0.196078431373\
  --pad_input\
  --wavelet_pooling\
  --transform_test skip\
  --seed 16
