#!/usr/bin/env bash
set -euo pipefail

python compare_results.py \
  --out-dir ./outputs \
  "${@}"

