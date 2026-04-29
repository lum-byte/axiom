"""Manual signal-kernel probe against a live SaaS docs page.

This stays outside the collected pytest file pattern because it performs a live
network fetch and is meant for operator verification.
"""

from __future__ import annotations

import urllib.request

from signal_kernel.contracts import (
    IntentVectorHash,
    KernelInput,
    SourceURL,
    TopologyClassStr,
    new_run_id,
)
from signal_kernel.pipeline import execute_sync
from signal_kernel.recipes import registry, validator


def main() -> None:
    url = "https://docs.stripe.com/api"
    with urllib.request.urlopen(url) as response:
        raw = response.read().decode("utf-8", errors="replace")

    kernel_input = KernelInput(
        raw_content=raw,
        topology_class=TopologyClassStr("SAAS_DOCS"),
        intent_vector_hash=IntentVectorHash("a" * 64),
        content_type="html",
        source_url=SourceURL(url),
        run_id=new_run_id(),
    )
    result = execute_sync(kernel_input, registry=registry, validator_check=validator.check)

    print("=== STATS ===")
    print(f"raw bytes:    {result.raw_byte_count:,}")
    print(f"clean bytes:  {result.clean_byte_count:,}")
    print(f"reduction:    {result.token_reduction_pct:.1%}")
    print(f"empty:        {result.extraction_empty}")
    print(f"latency:      {result.latency_ms:.1f}ms")
    print("\n=== CLEAN SIGNAL ===\n")
    print(result.clean_signal)


if __name__ == "__main__":
    main()
