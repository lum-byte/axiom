# AXIOM Swarm Bridge

The imported swarm tree is treated as a generic task/message language, not as a
terminal adapter.  AXIOM owns the bridge in `swarm/axiom/bridge.ts`.

Bridge watermark: `axiom.swarm.webwide.v1`

The bridge accepts loose envelopes such as queued messages, teammate tasks,
agent tasks, prompts, message arrays, and direct text.  It emits a concrete
`AxiomCrawlPlan` with query text, seed domains, source URLs, requested worker
count, and compute controls.  Python TAG code then clamps workers against the
runtime ceiling and executes with one crawler per site.
