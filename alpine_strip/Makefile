# alpine_strip Makefile
# Build: make all (or make test_strip / make batch_runner)
# Test:  make test

CC       ?= gcc
CFLAGS   := -std=c11 -Wall -Wextra -Wpedantic -O2
LDFLAGS  :=

PCRE2_PROBE := $(shell printf '%s\n' '#define PCRE2_CODE_UNIT_WIDTH 8' '#include <pcre2.h>' 'int main(void){return 0;}' | $(CC) -x c - -E >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(PCRE2_PROBE),yes)
  LDFLAGS += -lpcre2-8
else
  CFLAGS  += -DAXIOM_NO_PCRE2
endif

# Detect pthreads on non-Windows
UNAME := $(shell uname -s 2>/dev/null || echo Windows)
ifneq ($(findstring MINGW,$(UNAME)),)
  LDFLAGS += -lws2_32
else ifneq ($(findstring Windows,$(UNAME)),)
  LDFLAGS += -lws2_32
else
  CFLAGS  += -pthread
  LDFLAGS += -pthread
endif

# PCRE2 is used automatically when pcre2.h is available. If it is missing,
# the deterministic in-tree regex fallback is enabled with AXIOM_NO_PCRE2.

STRIP_SRCS := strip_engine.c tool_strip_accelerator.c
BATCH_SRCS := batch_runner.c
TEST_SRCS  := test_strip.c

STRIP_OBJS := $(STRIP_SRCS:.c=.o)
BATCH_OBJS := $(BATCH_SRCS:.c=.o)
TEST_OBJS  := $(TEST_SRCS:.c=.o)

.PHONY: all clean test

all: test_strip batch_runner

test_strip: $(STRIP_OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

batch_runner: $(STRIP_OBJS) $(BATCH_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c strip_engine.h strip_engine_internal.h batch_runner.h tool_strip_accelerator.h
	$(CC) $(CFLAGS) -c -o $@ $<

test: test_strip
	./test_strip

clean:
	rm -f *.o test_strip batch_runner test_strip.exe batch_runner.exe
