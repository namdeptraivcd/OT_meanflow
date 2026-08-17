#!/usr/bin/env bash
set -euo pipefail

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BASE_CH="${BASE_CH:-32}"
SINKHORN_EPS="${SINKHORN_EPS:-0.05}"
SINKHORN_ITERS="${SINKHORN_ITERS:-80}"

torchrun --standalone --nproc_per_node=2 train_mnist_meanflow.py \
  --method ot_gmf_sinkhorn_hard \
  --data-dir ./data \
  --out-dir ./outputs \
  --classifier ./outputs/mnist_classifier.pt \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --base-ch "$BASE_CH" \
  --sinkhorn-eps "$SINKHORN_EPS" \
  --sinkhorn-iters "$SINKHORN_ITERS" \
  --save-every 1000 \
  --eval-every 2000

