"""Quick diagnostic: why are 0 blocks extracted?"""
import asyncio, os, sys
os.environ.setdefault("AXIOM_CONFIG_TOML", "config.toml")

async def main():
    # Force debug logging to a file so we can see what's happening
    import logging
    logging.basicConfig(
        filename="diag_search.log", level=logging.DEBUG, force=True,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from tag.interface import AxiomInterface

    iface = AxiomInterface()
    resp = await iface.handle_line("search | swarm -10 | depth -2 | what is google")
    d = resp.data or {}
    blocks = d.get("blocks", [])
    sources = d.get("sources", [])
    cached = iface.runtime.document_cache

    out = []
    out.append(f"STATUS: {resp.status}")
    out.append(f"MESSAGE: {resp.message}")
    out.append(f"SOURCES: {len(sources)}")
    out.append(f"CACHED_DOCS: {len(cached)}")
    out.append(f"RANKED_BLOCKS: {len(blocks)}")

    for url, doc in list(cached.items()):
        out.append(
            f"  DOC domain={doc.domain} status={doc.status_code} "
            f"blocks={len(doc.blocks)} clean={len(doc.clean_text)} "
            f"kernel={len(doc.kernel_signal)} fetch={doc.fetch_mode}"
        )
        if doc.blocks:
            out.append(f"    BLOCK0: {doc.blocks[0][:150]}")

    report = "\n".join(out)
    with open("diag_search_result.txt", "w") as f:
        f.write(report + "\n")
    print(report)

    await iface.runtime.close()

asyncio.run(main())
