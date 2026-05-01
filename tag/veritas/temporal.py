"""Temporal helpers for VERITAS legacy classification."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Optional


YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")
DATE_RE = re.compile(r"\b(20[0-9]{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])\b")


def newest_year(texts: Iterable[str]) -> Optional[int]:
    years = []
    for text in texts:
        years.extend(int(match.group(1)) for match in YEAR_RE.finditer(str(text or "")))
    return max(years) if years else None


def newest_date(texts: Iterable[str]) -> Optional[datetime]:
    dates = []
    for text in texts:
        for match in DATE_RE.finditer(str(text or "")):
            try:
                dates.append(datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))))
            except ValueError:
                continue
    return max(dates) if dates else None


def temporal_delta_label(candidate_text: str, anchor_texts: Iterable[str]) -> str:
    candidate_year = newest_year([candidate_text])
    anchor_year = newest_year(anchor_texts)
    if candidate_year is None or anchor_year is None:
        return "unknown"
    if anchor_year > candidate_year:
        return "anchor_newer"
    if candidate_year > anchor_year:
        return "candidate_newer"
    return "same_year"
