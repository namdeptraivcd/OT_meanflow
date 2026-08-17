#!/usr/bin/env bash
set -euo pipefail

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BASE_CH="${BASE_CH:-32}"

torchrun --standalone --nproc_per_node=2 train_mnist_meanflow.py \
  --method ot_gmf_hard \
  --data-dir ./data \
  --out-dir ./outputs \
  --classifier ./outputs/mnist_classifier.pt \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --base-ch "$BASE_CH" \
  --save-every 1000 \
  --eval-every 2000

