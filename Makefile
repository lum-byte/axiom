.PHONY: all check-prereqs test test-python test-go test-c test-cuda test-rust build clean

CC ?= gcc
NVCC ?= nvcc
PYTHON ?= python

all: check-prereqs build

check-prereqs:
	@$(PYTHON) --version
	@$(CC) --version >/dev/null
	@$(NVCC) --version >/dev/null || true
	@go version >/dev/null || echo "go missing: preparser tests skipped until installed"
	@cargo --version >/dev/null || echo "cargo missing: Rust TUI tests skipped until installed"

build: build-c build-cuda build-rust

build-c:
	$(CC) -std=c11 -O2 -Wall -Wextra alpine_strip/strip_engine.c alpine_strip/batch_runner.c -o alpine_strip/batch_runner
	$(CC) -std=c11 -O2 -Wall -Wextra daemons/phase_daemon.c -o daemons/phase_daemon
	$(CC) -std=c11 -O2 -Wall -Wextra -DAXIOM_PHASE_DAEMON_NO_MAIN daemons/phase_daemon.c daemons/store_sentinel.c -o daemons/store_sentinel

build-cuda:
	$(NVCC) offline/gpu_encoder.cu offline/gradient_accumulator.cu offline/weight_updater.cu offline/batch_scheduler.c -o offline/batch_scheduler || true

build-rust:
	cd axiom_tui && cargo build --release || true

test: test-python test-go test-c test-cuda test-rust

test-python:
	$(PYTHON) -m pytest tests/test_contract_boundary.py tests/test_crawler_bus.py tests/test_runtime_surface.py tests/test_axiom_spec_surface.py -q

test-go:
	go test ./preparser/...

test-c:
	sh ./run_c_tests.sh

test-cuda:
	sh ./run_cuda_tests.sh

test-rust:
	cd axiom_tui && cargo test

clean:
	rm -f alpine_strip/test_strip alpine_strip/batch_runner daemons/test_daemons daemons/phase_daemon daemons/store_sentinel offline/test_offline offline/batch_scheduler
