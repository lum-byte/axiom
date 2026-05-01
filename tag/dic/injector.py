"""Context injection formatter for DIC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from tag.dic.assembler import DICContext


@dataclass(frozen=True)
class InjectedContext:
    text: str
    json_payload: Dict[str, Any]


class DirectContextInjector:
    def format(self, context: DICContext) -> InjectedContext:
        payload = context.to_dict()
        sections = [
            "[ANCHOR SOURCES]",
            json.dumps(payload["anchor"], ensure_ascii=False, indent=2),
            "[SUPPORTING CONTEXT]",
            json.dumps(payload["supporting"], ensure_ascii=False, indent=2),
            "[CONTESTED / VERIFY]",
            json.dumps(payload["contested"], ensure_ascii=False, indent=2),
            "[QUERY TRACE]",
            json.dumps(payload["query_trace"], ensure_ascii=False, indent=2),
            "[TAG-DIC ANSWER]",
            payload["answer"],
        ]
        return InjectedContext(text="\n\n".join(sections), json_payload=payload)
