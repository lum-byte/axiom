"""Diagnostic: trace exactly where the search pipeline loses documents."""
import asyncio
import time
from signal_kernel.contracts import new_run_id
from tag.interface import AxiomInterface, InterfaceRequest

__test__ = False

async def main():
    interface = AxiomInterface()
    orch = interface.orchestrator

    # 1. Generate candidates
    query = "what is google"
    candidates = orch._candidate_sources(query)
    print(f"\n[1] CANDIDATES: {len(candidates)} generated")
    for c in candidates[:5]:
        print(f"    {c['domain']:20s} {c['reason']:30s} {c['url'][:80]}")

    # 2. Fetch ONE document directly (bypassing swarm)
    await interface.runtime.ensure_crawl_stack()
    run_id = str(new_run_id())
    test_url = "https://en.wikipedia.org/wiki/Google"
    print(f"\n[2] DIRECT FETCH: {test_url}")
    
    # Check bloom filter state
    fetcher = interface.runtime.fetcher
    if fetcher and hasattr(fetcher, '_bloom') and fetcher._bloom:
        in_bloom = await fetcher._bloom.contains(test_url)
        print(f"    Bloom filter contains URL: {in_bloom}")
    else:
        print(f"    No bloom filter active")
    
    doc = await orch._fetch_document(test_url, run_id, reason="test")
    if doc:
        print(f"    SUCCESS: blocks={len(doc.blocks)}, clean_text={len(doc.clean_text)}")
    else:
        print(f"    FAILED: _fetch_document returned None")

    # 3. Run full collect_documents through swarm
    print(f"\n[3] SWARM COLLECT:")
    run_id2 = str(new_run_id())
    documents = await orch._collect_documents(query, run_id2, candidates)
    print(f"    Documents returned: {len(documents)}")
    for d in documents:
        print(f"    {d.domain:20s} blocks={len(d.blocks):3d} status={d.status_code} clean_text_len={len(d.clean_text)}")

    # 4. Rank
    ranked = orch._rank_documents(query, documents)
    print(f"\n[4] RANKED BLOCKS: {len(ranked)}")
    for r in ranked[:3]:
        print(f"    score={r['score']:6.2f} {r['domain']:20s} text={r['text'][:60]}...")

    # 5. Check the event queue for errors
    print(f"\n[5] EVENT QUEUE (fetch failures/exceptions):")
    while interface.runtime.queued_work:
        evt = interface.runtime.queued_work.popleft()
        if evt.get("type") in ("fetch_failed", "fetch_exception", "crawl_swarm_complete"):
            print(f"    {evt}")

if __name__ == "__main__":
    asyncio.run(main())
