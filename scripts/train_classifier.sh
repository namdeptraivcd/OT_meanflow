#!/usr/bin/env bash
set -euo pipefail

python train_mnist_classifier.py \
  --data-dir ./data \
  --out ./outputs/mnist_classifier.pt \
  --epochs 3 \
  --batch-size 256

