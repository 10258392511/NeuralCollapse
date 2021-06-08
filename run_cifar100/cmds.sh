#!/bin/bash

for K in {3..5}
do
  python3 train_K_gt_d.py --epoch 30 --batch-size 128 --classes $K --feature-dim 2 --lr 2e-2
  python3 train_K_gt_d.py --epoch 30 --batch-size 128 --classes $K --feature-dim 2 --feature-man --lr 2e-2
  python3 train_K_gt_d.py --epoch 30 --batch-size 128 --classes $K --feature-dim 2 --weight-man --lr 2e-2
  python3 train_K_gt_d.py --epoch 30 --batch-size 128 --classes $K --feature-dim 2 --feature-man --weight-man --lr 2e-2
done
