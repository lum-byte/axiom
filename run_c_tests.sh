#!/usr/bin/env sh
set -eu
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L}"
TEST_BIN_DIR="${TEST_BIN_DIR:-tests/.bin}"
STRIP_CFLAGS=""
STRIP_LDFLAGS=""
mkdir -p "$TEST_BIN_DIR"
if printf '%s\n' '#define PCRE2_CODE_UNIT_WIDTH 8' '#include <pcre2.h>' 'int main(void){return 0;}' | "$CC" -x c - -E >/dev/null 2>&1; then
  STRIP_LDFLAGS="-lpcre2-8"
else
  STRIP_CFLAGS="-DAXIOM_NO_PCRE2"
fi
"$CC" $CFLAGS $STRIP_CFLAGS alpine_strip/strip_engine.c alpine_strip/tool_strip_accelerator.c alpine_strip/test_strip.c -o "$TEST_BIN_DIR/test_strip" $STRIP_LDFLAGS
"$TEST_BIN_DIR/test_strip"
"$CC" $CFLAGS -DAXIOM_DAEMON_TEST daemons/phase_daemon.c daemons/store_sentinel.c daemons/test_daemons.c -o "$TEST_BIN_DIR/test_daemons"
"$TEST_BIN_DIR/test_daemons"
