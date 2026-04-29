#!/bin/sh
# Canonical AXIOM NEWS_ARTICLE recipe entrypoint.
# The implementation file in this checkout is news_articles.sh; keep this
# wrapper so registry.py, tests, and internal docs agree on news_article.sh.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
exec sh "$SCRIPT_DIR/news_articles.sh"
