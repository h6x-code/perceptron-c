#!/usr/bin/env bash
for t in 1 2 4 8 16; do
  ./perceptron train \
    --dataset mnist \
    --mnist-images data/MNIST/raw/train-images-idx3-ubyte \
    --mnist-labels data/MNIST/raw/train-labels-idx1-ubyte \
    --limit 10000 --val 0.1 \
    --layers 2 --units 256,64 \
    --epochs 10 --batch 128 --lr 0.05 --momentum 0.9 \
    --lr-decay 0.95 --lr-step 2 \
    --seed 1337 --threads $t \
    | tee logs/thread${t}.log
done
