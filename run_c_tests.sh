#!/usr/bin/env sh
set -eu
CC="${CC:-gcc}"
"$CC" -std=c11 -O2 -Wall -Wextra alpine_strip/strip_engine.c alpine_strip/test_strip.c -o alpine_strip/test_strip
./alpine_strip/test_strip
"$CC" -std=c11 -O2 -Wall -Wextra -DAXIOM_DAEMON_TEST daemons/phase_daemon.c daemons/store_sentinel.c daemons/test_daemons.c -o daemons/test_daemons
./daemons/test_daemons

