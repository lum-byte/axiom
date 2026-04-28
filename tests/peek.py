import urllib.request
from signal_kernel.contracts import KernelInput, TopologyClassStr, IntentVectorHash, SourceURL, new_run_id
from signal_kernel.pipeline import execute_sync
from signal_kernel.recipes import registry, validator

url = "https://docs.stripe.com/api"

with urllib.request.urlopen(url) as r:
    raw = r.read().decode("utf-8", errors="replace")

ki = KernelInput(
    raw_content=raw,
    topology_class=TopologyClassStr("SAAS_DOCS"),
    intent_vector_hash=IntentVectorHash("a" * 64),
    content_type="html",
    source_url=SourceURL(url),
    run_id=new_run_id(),
)

result = execute_sync(ki, registry=registry, validator_check=validator.check)

print(f"=== STATS ===")
print(f"raw bytes:    {result.raw_byte_count:,}")
print(f"clean bytes:  {result.clean_byte_count:,}")
print(f"reduction:    {result.token_reduction_pct:.1%}")
print(f"empty:        {result.extraction_empty}")
print(f"latency:      {result.latency_ms:.1f}ms")
print(f"\n=== CLEAN SIGNAL ===\n")
print(result.clean_signal)