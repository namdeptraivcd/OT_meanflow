#!/usr/bin/env bash
set -euo pipefail

METHOD="${METHOD:?Set METHOD to meanflow, ot_gmf_hard, or ot_gmf_sinkhorn_hard}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BASE_CH="${BASE_CH:-32}"

torchrun --standalone --nproc_per_node=2 train_mnist_meanflow.py \
  --method "$METHOD" \
  --data-dir ./data \
  --out-dir ./outputs \
  --classifier ./outputs/mnist_classifier.pt \
  --resume "./outputs/${METHOD}/checkpoints/latest.pt" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --base-ch "$BASE_CH" \
  --save-every 1000 \
  --eval-every 2000

