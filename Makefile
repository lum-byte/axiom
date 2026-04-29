.PHONY: all check-prereqs test test-python test-go test-c test-cuda test-rust build build-runtime clean

CC ?= gcc
PYTHON ?= python
CPPFLAGS ?=
CFLAGS ?= -std=c11 -O2 -Wall -Wextra
AXIOM_POSIX_CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L
RELEASE_ROOT ?= Releases-x64
LINUX_BIN_DIR ?= $(RELEASE_ROOT)/compiled/binaries/Linux64
WIN_BIN_DIR ?= $(RELEASE_ROOT)/compiled/binaries/Winx64
CUDA_HOME_CANDIDATES := $(wildcard .venv/lib/python*/site-packages/nvidia/cu13 .venv/lib/python*/site-packages/nvidia/cu12)
AXIOM_CUDA_HOME ?= $(firstword $(CUDA_HOME_CANDIDATES))
NVCC ?= $(if $(AXIOM_CUDA_HOME),$(AXIOM_CUDA_HOME)/bin/nvcc,nvcc)
CUDA_CPPFLAGS ?= $(if $(AXIOM_CUDA_HOME),-I$(AXIOM_CUDA_HOME)/include,)
CUDA_LDFLAGS ?= $(if $(AXIOM_CUDA_HOME),-L$(AXIOM_CUDA_HOME)/lib,)
CUDA_LIBRARY_PATH ?= $(if $(AXIOM_CUDA_HOME),$(AXIOM_CUDA_HOME)/lib,)

PCRE2_PROBE := $(shell printf '%s\n' '#define PCRE2_CODE_UNIT_WIDTH 8' '#include <pcre2.h>' 'int main(void){return 0;}' | $(CC) $(CPPFLAGS) -x c - -E >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(PCRE2_PROBE),yes)
  STRIP_PCRE2_FLAGS :=
  STRIP_PCRE2_LIBS := -lpcre2-8
else
  STRIP_PCRE2_FLAGS := -DAXIOM_NO_PCRE2
  STRIP_PCRE2_LIBS :=
endif

all: check-prereqs build

check-prereqs:
	@$(PYTHON) --version
	@$(CC) --version >/dev/null
	@$(NVCC) --version >/dev/null || true
	@go version >/dev/null || echo "go missing: preparser tests skipped until installed"
	@cargo --version >/dev/null || echo "cargo missing: Rust TUI tests skipped until installed"

build: build-c build-cuda build-rust build-runtime

build-c:
	mkdir -p $(LINUX_BIN_DIR)
	$(CC) $(AXIOM_POSIX_CPPFLAGS) $(CPPFLAGS) $(CFLAGS) $(STRIP_PCRE2_FLAGS) alpine_strip/strip_engine.c alpine_strip/tool_strip_accelerator.c alpine_strip/batch_runner.c -o $(LINUX_BIN_DIR)/batch_runner $(STRIP_PCRE2_LIBS)
	$(CC) $(AXIOM_POSIX_CPPFLAGS) $(CPPFLAGS) $(CFLAGS) daemons/phase_daemon.c -o $(LINUX_BIN_DIR)/phase_daemon
	$(CC) $(AXIOM_POSIX_CPPFLAGS) $(CPPFLAGS) $(CFLAGS) -DAXIOM_PHASE_DAEMON_NO_MAIN daemons/phase_daemon.c daemons/store_sentinel.c -o $(LINUX_BIN_DIR)/store_sentinel

build-cuda:
	mkdir -p $(LINUX_BIN_DIR)
	LD_LIBRARY_PATH="$(CUDA_LIBRARY_PATH):$${LD_LIBRARY_PATH}" $(NVCC) $(CUDA_CPPFLAGS) offline/gpu_encoder.cu offline/gradient_accumulator.cu offline/weight_updater.cu offline/batch_scheduler.c $(CUDA_LDFLAGS) -o $(LINUX_BIN_DIR)/offline_batch_scheduler || true

build-rust:
	mkdir -p $(LINUX_BIN_DIR)
	cd axiom_tui && cargo build --release || true
	cp axiom_tui/target/release/axiom $(LINUX_BIN_DIR)/axiom || true

build-runtime:
	mkdir -p $(RELEASE_ROOT) $(LINUX_BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared axiom_runtime/axiom_runtime.c -o $(LINUX_BIN_DIR)/axirt.so
	cp $(LINUX_BIN_DIR)/axirt.so $(RELEASE_ROOT)/axi.so
	rm -f $(RELEASE_ROOT)/axirt.so $(RELEASE_ROOT)/axirt.dll $(LINUX_BIN_DIR)/axi.so

test: test-python test-go test-c test-cuda test-rust

test-python:
	$(PYTHON) -m pytest tests/test_contract_boundary.py tests/test_axiom_spec_surface.py tests/test_axiom_sdk_surface.py tests/test_tools_bridge.py -q
	$(PYTHON) -m pytest tests/test_crawler_bus.py -q
	$(PYTHON) -m pytest tests/test_runtime_surface.py -q

test-go:
	go test ./preparser/...

test-c:
	sh ./run_c_tests.sh

test-cuda:
	sh ./run_cuda_tests.sh

test-rust:
	cd axiom_tui && cargo test

clean:
	rm -rf $(RELEASE_ROOT) tests/.bin
	rm -f alpine_strip/test_strip alpine_strip/test_strip.exe alpine_strip/test_strip.o alpine_strip/batch_runner daemons/test_daemons daemons/test_daemons.exe daemons/phase_daemon daemons/store_sentinel offline/test_offline offline/test_offline_cpu offline/test_offline_nvcc_probe offline/batch_scheduler offline/batch_scheduler.o axiom_runtime/libaxiom_runtime.so axiom_runtime/axiom_runtime.dll axiom_runtime/test_axiom_runtime
