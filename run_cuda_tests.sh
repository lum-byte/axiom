#!/usr/bin/env sh
set -eu
NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc missing: skipping CUDA tests"
  exit 0
fi
"$NVCC" -DAXIOM_OFFLINE_TEST offline/gpu_encoder.cu offline/gradient_accumulator.cu offline/weight_updater.cu offline/batch_scheduler.c offline/test_offline.cu -o offline/test_offline
./offline/test_offline
