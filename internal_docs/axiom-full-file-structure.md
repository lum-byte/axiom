# AXIOM — Complete File Structure
**Classification: AXIOM INTERNAL // DO NOT SURFACE**
**Status: signal_kernel/ COMPLETE. Everything else: not started.**

---

```
axiom/
│
├── signal_kernel/                          ✅ COMPLETE
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── docker-compose.yml
│   ├── contracts.py
│   ├── exceptions.py
│   ├── pipeline.py
│   ├── feedback.py
│   ├── checkpoint/
│   │   ├── mft_checkpoint.sh
│   │   ├── restore.sh
│   │   └── checkpoint_monitor.py
│   └── recipes/
│       ├── registry.py
│       ├── validator.py
│       ├── hardcoded/
│       │   ├── news_article.sh
│       │   ├── saas_docs.sh
│       │   ├── rest_api_json.sh
│       │   ├── json_ld.sh
│       │   └── ecommerce.sh
│       ├── compiler_generated/             ← written by topology/parser.py at runtime
│       └── test_fixtures/                  ← per topology class, used by validator dry-run
│           ├── NEWS_ARTICLE/
│           ├── SAAS_DOCS/
│           ├── REST_API_JSON/
│           ├── JSON_LD_STRUCTURED/
│           └── ECOMMERCE_PRODUCT/
│
│
├── tag/
│   │
│   ├── interface.py                        ← AXIOM graph → TAG boundary. public surface.
│   ├── index_daemon.py                     ← RL loop. never terminates. gradient steps.
│   ├── crawler_bus.py                      ← typed event bus. subscribe/emit. zero logic.
│   ├── preparse_daemon.py                  ← watches triggers. dispatches preparse cycles.
│   ├── cold_start.py                       ← initialization orchestrator. runs before queries.
│   ├── store_watchdog.py                   ← inotify watcher. zero polling. debounced reload.
│   │
│   ├── crawler/                            ← LAYER 1: acquisition only. the vacuum.
│   │   ├── __init__.py
│   │   ├── fetcher.py                      ← httpx + Playwright. receives CrawlManifest.
│   │   │                                      emits RawFetchEvent. nothing else.
│   │   ├── rate_limiter.py                 ← per-domain pacing from DomainMap.
│   │   │                                      never discovers limits reactively.
│   │   ├── frontier.py                     ← resumable crawl frontier. SQLite backend.
│   │   │                                      survives crashes. interrupt-safe.
│   │   ├── crawl_cursor.py                 ← atomic position checkpoint every N URLs.
│   │   │                                      resume from exact position after any crash.
│   │   └── bloom_filter.py                 ← seen URL dedup. mmap. 400M URLs in 500MB.
│   │
│   ├── alpine_strip/                       ← LAYER 2: same signal_kernel, offline context.
│   │   ├── __init__.py
│   │   ├── offline_pipeline.py             ← wraps signal_kernel/pipeline.py for batch use.
│   │   │                                      subscribes to RawFetchEvent. emits CleanSignalEvent.
│   │   └── batch_executor.py               ← GPU-parallel batch invocation.
│   │                                          80K URLs/sec on 5080.
│   │                                          same warm Alpine container as critical path.
│   │
│   ├── preparser/                          ← LAYER 3: structural intelligence. the human.
│   │   ├── __init__.py
│   │   ├── robots_parser.py                ← robots.txt → structural signals. not compliance.
│   │   ├── sitemap_parser.py               ← XML sitemap → URL frontier + path topology.
│   │   ├── path_classifier.py              ← URL patterns → topology classes.
│   │   ├── domain_analyzer.py              ← OPUS. synthesizes all signals → DomainMap.
│   │   │                                      emits DomainTopologyEvent + CrawlManifestReadyEvent.
│   │   └── crawl_planner.py                ← OPUS. DomainMap → optimal CrawlManifest.
│   │                                          path ordering. priority. rate pacing.
│   │
│   ├── world_model/                        ← LAYERS 4A + 4B: parallel. never coordinate.
│   │   ├── __init__.py
│   │   ├── latent_model.py                 ← OPUS. WLM. satellite view.
│   │   │                                      subscribes to DomainTopologyEvent.
│   │   │                                      outputs TraversalPolicy + FrictionForecast.
│   │   │                                      feeds Phantom at query time.
│   │   └── latent_parser.py                ← OPUS. WLP. street level.
│   │                                          subscribes to CleanSignalEvent.
│   │                                          outputs ZoneMap per topology class.
│   │                                          emits ZoneMapUpdatedEvent → parser recompiles.
│   │
│   ├── topology/                           ← classification + compilation + surprise
│   │   ├── __init__.py
│   │   ├── classifier.py                   ← OPUS (model path) + SONNET (signal paths).
│   │   │                                      URL + headers + 4KB window → topology class string.
│   │   │                                      never fetches a page. classification before fetch.
│   │   ├── parser.py                       ← OPUS. recipe compiler.
│   │   │                                      subscribes to ZoneMapUpdatedEvent.
│   │   │                                      WLP zone map → compiled grep recipe.
│   │   │                                      writes to signal_kernel/recipes/compiler_generated/.
│   │   ├── sanitizer.py                    ← SONNET. last mile strip.
│   │   │                                      HTML entities. unicode. GDPR fragments.
│   │   │                                      JS artifacts. whitespace. code fragments.
│   │   └── surprise_detector.py            ← OPUS (score+dissolve) + SONNET (history).
│   │                                          emits SurpriseEvent to bus.
│   │                                          index_daemon subscribes.
│   │
│   ├── phantom/                            ← live traversal. critical path only.
│   │   ├── __init__.py
│   │   ├── phantom.py                      ← SONNET. httpx + Playwright.
│   │   │                                      receives TraversalPolicy from WLM.
│   │   │                                      returns PhantomResult.
│   │   └── render_policy.py                ← SONNET. static vs headless decision.
│   │                                          lookup into TraversalPolicy.render_mode.
│   │
│   ├── offline/                            ← LAYER 6: ingestion + encoding. GPU work.
│   │   ├── __init__.py
│   │   ├── preparse.py                     ← cycle orchestrator. sources → encoders.
│   │   ├── manifest.py                     ← /store/manifest.json read/write.
│   │   │                                      source states. timing. article counts.
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── wikipedia.py                ← XML dump parser + incremental diff via API.
│   │   │   │                                  monthly full. daily incremental (~50K articles).
│   │   │   ├── arxiv.py                    ← OAI-PMH harvest. abstract corpus.
│   │   │   ├── docs_crawler.py             ← top 10K documentation sites. sitemap-driven.
│   │   │   └── source_registry.py          ← registered sources + schedules + priority.
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   ├── signal_batcher.py           ← GPU batch kernel invocation.
│   │   │   │                                  clean signal in. topology features out.
│   │   │   ├── link_graph_builder.py       ← weighted link graph per source corpus.
│   │   │   ├── intent_clusterer.py         ← OPUS. intent vector clustering.
│   │   │   │                                  articles → intent cluster membership.
│   │   │   └── topology_encoder.py         ← OPUS. topology graph → structural_layer.pt.
│   │   │                                      gradient descent. weights as index.
│   │   └── validation/
│   │       ├── __init__.py
│   │       ├── integrity_checker.py        ← SHA256 verify staging file before swap.
│   │       └── quality_sampler.py          ← spot-check N random articles post-encode.
│   │
│   └── store/                              ← four files. everything TAG knows.
│       ├── topology_router.pt              ← tiny MLP weights. IS the MFT index.
│       │                                      query embedding → forward pass → routing vector.
│       │                                      fine-tuned by RL loop. gradient step = index update.
│       ├── recipe_registry.mmap            ← memory-mapped recipe lookup. OS handles paging.
│       │                                      survives process death. on restart: immediately available.
│       ├── phase_states.mmap               ← per topology class phase tracking (I / II / III).
│       │                                      updated by index_daemon on every surprise evaluation.
│       ├── structural_layer.pt             ← hivemind shared weights. invariant primitives.
│       │                                      written by offline/encoders/topology_encoder.py.
│       │                                      federated via rsync across fleet instances.
│       ├── manifest.json                   ← preparse history. source states. timing. hashes.
│       ├── triggers/
│       │   ├── cold_start                  ← written by entrypoint.sh on container start.
│       │   ├── preparse                    ← written by crond daily.
│       │   └── reload_structural_layer     ← written by preparse_daemon after atomic swap.
│       ├── staging/
│       │   ├── structural_layer.pt.staging ← preparse writes here. never directly to .pt.
│       │   └── structural_layer.pt.sha256  ← hash of staging for integrity check.
│       └── checkpoints/                    ← 48 rotating archives from crond. 12hr history.
│           ├── mft_20260305_120000.tar.gz
│           └── ...
│
│
└── axiom/                                  ← AXIOM graph layer. not started. built last.
    ├── controller.py                       ← AxiomState orchestration
    ├── graph/
    │   ├── __init__.py
    │   ├── semantic_intent_graph.py        ← custom graph engine. not LangGraph.
    │   ├── nodes/
    │   │   ├── tag_node.py                 ← calls interface.py
    │   │   ├── haiku_node.py               ← Semantic Extractor. prose + code split.
    │   │   ├── se_separator.py             ← splits clean signal → prose + code
    │   │   └── ...
    │   └── state/
    │       └── axiom_state.py              ← AxiomState dataclass
    └── hivemind/
        ├── __init__.py
        └── rsync_federation.py             ← structural_layer.pt → fleet instances
```

---

## Counts

```
signal_kernel/          ~15K LOC production  ~10K tests  =  25K total  ✅ DONE
tag/crawler/            ~1K   LOC             ~1K  tests  =  2K
tag/alpine_strip/       ~500  LOC             ~500 tests  =  1K        (wraps existing)
tag/preparser/          ~3K   LOC             ~3K  tests  =  6K
tag/world_model/        ~8K   LOC             ~8K  tests  =  16K
tag/topology/           ~5K   LOC             ~6K  tests  =  11K
tag/phantom/            ~1K   LOC             ~1K  tests  =  2K
tag/offline/            ~4K   LOC             ~3K  tests  =  7K
tag/daemons+init        ~3K   LOC             ~2K  tests  =  5K
axiom/graph/            ~6K   LOC             ~5K  tests  =  11K
axiom/hivemind/         ~1K   LOC             ~500 tests  =  1.5K
                       ────────────────────────────────────────────
TOTAL ESTIMATE          ~47K  production      ~40K tests  =  ~87K
+ signal_kernel                                           =  ~112K LOC
```

---

## What is done and what is next

```
DONE        signal_kernel/          kernel tested. 350 tests pass.
                                    Stripe: 1.1MB → 8.4KB in 7.4ms
                                    Twilio: 1.6MB → 1.5KB in 9ms

NEXT        tag/topology/classifier.py
                first file of the intelligence layer
                Sonnet builds signal paths first
                Opus builds model path after
                depends on: contracts.py additions
                            exceptions.py additions
                            store/topology_router.pt initialized
```

---

*AXIOM Internal // Do Not Surface*
*TAG not RAG. Weights as index. Loop never terminates.*
