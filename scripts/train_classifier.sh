#!/bin/sh
root="./.."
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 ${root}/train_classifier.py\
  --data ${root}/datasets/ILSVRC2012\
  --out-dir ${root}/logs/train_classifier\
  --arch alexnet\
  --dataset imagenet\
  --adv-train 0\
  --constraint 2\
  --attack-lr 0.5\
  --attack-steps 7\
  --eps 1\
  --epochs 90\
  --lr 0.1\
  --batch-size 128\
  --step-lr 30\
  --seed 16\
  --save-ckpt-iters 5
