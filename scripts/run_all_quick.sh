#!/usr/bin/env bash
set -euo pipefail

export STEPS="${STEPS:-20000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export BASE_CH="${BASE_CH:-32}"

bash scripts/train_classifier.sh
bash scripts/run_meanflow.sh
bash scripts/run_ot_hard.sh
bash scripts/run_ot_sinkhorn.sh

