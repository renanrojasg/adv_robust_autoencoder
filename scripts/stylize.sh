#!/bin/sh
root="./.."
CUDA_VISIBLE_DEVICES=7 python3 ${root}/stylize.py\
  --content_images ${root}/datasets/st/merge/content\
  --style_images ${root}/datasets/st/merge/style\
  --arch alexnet\
  --load_classifier ${root}/checkpoints/alexnet_inv/alexnet_ar.pt\
  --load_conv5_generator ${root}/checkpoints/alexnet_st/alexnet_ar_conv5.pt\
  --load_conv2_generator ${root}/checkpoints/alexnet_st/alexnet_ar_conv2.pt\
  --load_conv1_generator ${root}/checkpoints/alexnet_st/alexnet_ar_conv1.pt\
  --comparator_arch vgg19\
  --load_comparator ${root}/checkpoints/ext/tvis/vgg19.pth\
  --stylize_layers conv1 conv2 conv5\
  --compare_layers conv1 conv2 conv3 conv4 conv5\
  --out_dir ${root}/logs/stylize\
  --infer_export\
