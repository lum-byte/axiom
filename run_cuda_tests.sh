#!/usr/bin/env sh
set -eu
NVCC="${NVCC:-nvcc}"
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
TEST_BIN_DIR="${TEST_BIN_DIR:-tests/.bin}"
AXIOM_CUDA_HOME="${AXIOM_CUDA_HOME:-}"
mkdir -p "$TEST_BIN_DIR"
if [ -z "$AXIOM_CUDA_HOME" ]; then
  for candidate in .venv/lib/python*/site-packages/nvidia/cu13 .venv/lib/python*/site-packages/nvidia/cu12; do
    if [ -x "$candidate/bin/nvcc" ]; then
      AXIOM_CUDA_HOME="$candidate"
      break
    fi
  done
fi
CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS:-}"
CUDA_LDFLAGS="${CUDA_LDFLAGS:-}"
if [ -n "$AXIOM_CUDA_HOME" ]; then
  if [ "$NVCC" = "nvcc" ] && [ -x "$AXIOM_CUDA_HOME/bin/nvcc" ]; then
    NVCC="$AXIOM_CUDA_HOME/bin/nvcc"
  fi
  CUDA_NVCC_FLAGS="-I$AXIOM_CUDA_HOME/include $CUDA_NVCC_FLAGS"
  CUDA_LDFLAGS="-L$AXIOM_CUDA_HOME/lib $CUDA_LDFLAGS"
  export LD_LIBRARY_PATH="$AXIOM_CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if command -v "$NVCC" >/dev/null 2>&1; then
  if "$NVCC" $CUDA_NVCC_FLAGS -DAXIOM_OFFLINE_TEST offline/gpu_encoder.cu offline/gradient_accumulator.cu offline/weight_updater.cu offline/batch_scheduler.c offline/test_offline.cu $CUDA_LDFLAGS -o "$TEST_BIN_DIR/test_offline" >/tmp/axiom_nvcc_build.log 2>&1; then
    "$TEST_BIN_DIR/test_offline"
    exit 0
  fi
  echo "nvcc build failed; falling back to CPU C++ offline tests"
  cat /tmp/axiom_nvcc_build.log
fi

if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "g++ missing: skipping CPU fallback offline tests"
  exit 0
fi

"$CC" -DAXIOM_OFFLINE_TEST -std=c11 -O2 -Wall -Wextra -c offline/batch_scheduler.c -o "$TEST_BIN_DIR/batch_scheduler.o"
"$CXX" -DAXIOM_OFFLINE_TEST -x c++ -std=c++17 -O2 -Wall -Wextra \
  offline/gpu_encoder.cu \
  offline/gradient_accumulator.cu \
  offline/weight_updater.cu \
  offline/test_offline.cu \
  -x none \
  "$TEST_BIN_DIR/batch_scheduler.o" \
  -o "$TEST_BIN_DIR/test_offline_cpu"
"$TEST_BIN_DIR/test_offline_cpu"
