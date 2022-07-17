#!/bin/sh
root="./.."
CUDA_VISIBLE_DEVICES=7 python3 -m robustness.main\
  --dataset imagenet\
  --data ${root}/datasets/ILSVRC2012\
  --eval-only 1\
  --out-dir ${root}/logs/predict_classifier\
  --resume ${root}/checkpoints/alexnet_inv/alexnet_ar.pt\
  --arch alexnet\
  --adv-eval 1\
  --constraint 2\
  --attack-lr 0.5\
  --attack-steps 7\
  --eps 3\
  --batch-size 256
