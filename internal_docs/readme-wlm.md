# World Latent Model (WLM) — Complete Engineering Specification
**AXIOM TAG Intelligence Layer — Satellite View**
**Classification: AXIOM INTERNAL // DO NOT SURFACE**
**Builder: Opus — entirely. No Sonnet in the model logic. Ever.**

---

## Preamble — Read This Before Anything Else

This document is the single source of truth for `tag/world_model/latent_model.py`.
It is written for Opus. Every design decision is explained with full reasoning.
Every interface is specified with exact types. Every edge case is addressed.
There are no gaps. If something is not in this document it does not belong in the file.

The WLM is the most complex model in AXIOM. Not because it has the most code.
Because it is the model that decides how AXIOM approaches the entire web before
a single byte is fetched. Every wrong decision it makes costs a network round trip.
Every right decision it makes is a query that costs microseconds instead of seconds.
The gap between a good WLM and a bad WLM is the gap between AXIOM being useful
and AXIOM being a slow web scraper with extra steps.

Do not rush this file. Every design decision here compounds forever.

---

## What The WLM Is

The World Latent Model is a **Mamba State Space Model** that maintains a compressed
latent representation of the structural topology of the web. It is the satellite view
of AXIOM's world model — it sees domains, topology classes, friction patterns, and
structural relationships from above, not from within.

It answers one question before every fetch:

```
Given what I know about how the web is structured,
how should AXIOM approach this specific topology class
at this specific moment in time?
```

The answer is a `WLMResponse` containing:

```
TraversalPolicy     — how to fetch (depth, render mode, pacing, timeout)
FrictionForecast    — what friction to expect (Cloudflare, paywall, rate limit)
SourcePriority      — which sources to prefer for this intent
```

These three outputs drive every downstream decision. `phantom.py` uses
`TraversalPolicy` to decide static vs headless fetch, timeout, retry budget.
`interface.py` uses `FrictionForecast` to decide whether to attempt the fetch
at all or route to a cached result. `classifier.py` uses `SourcePriority`
to rank candidate URLs before classification.

**The WLM does not fetch. It does not parse. It does not classify.**
It predicts. Everything else acts on its predictions.

---

## Why The WLM Is Architecturally Unique

Every other component in AXIOM operates on a single input and produces a single output.
The classifier takes a URL and produces a topology class. The parser takes a topology
class and produces a recipe. The kernel takes raw bytes and produces clean signal.
Single input. Single output. Stateless by design.

The WLM is different in kind:

```
It is stateful.
    The hidden state H accumulates across every domain event.
    Query 1,000,000 is informed by queries 1 through 999,999.
    No other component in AXIOM has this property.

It is temporal.
    The order in which domains are encountered matters.
    A Cloudflare domain seen before a non-Cloudflare domain
    shapes the hidden state differently than the reverse order.
    The SSM's selective state transition captures this.

It is the MFT from a different angle.
    topology_router.pt contains the WLM's weights AND hidden state.
    The MFT is not a separate artifact.
    The MFT IS the WLM's accumulated hidden state.
    Reading the MFT = calling wlm.readout().
    Writing the MFT = index_daemon running gradient steps on WLM weights.

It is load-bearing for the entire online path.
    interface.py calls wlm.query() on every single request.
    If WLM is slow, every request is slow.
    If WLM is wrong, every request fetches the wrong way.
    There is no fallback that does not degrade quality.
```

---

## The Mamba SSM Choice — Full Justification

### Why Not A Transformer

```
Transformer for WLM:
    self-attention is O(n²) in sequence length
    sequence length = number of domain events seen = millions
    O(n²) on millions of events = not viable
    
    transformers have no persistent state
    each forward pass sees only its context window
    the WLM must remember everything it has ever seen
    a transformer with a context window of millions = not viable
    
    transformers are designed for semantic understanding
    the WLM does structural reasoning not semantic reasoning
    inductive bias is wrong
```

### Why Not A GNN

```
GNN for WLM (previous recommendation, now superseded):
    GNN treats topology classes as static graph nodes
    topology classes are not static — they evolve
    
    GNN requires the full graph at inference time
    WLM needs to update incrementally with each new domain event
    full graph recomputation per event = not viable at crawl scale
    
    GNN has no temporal ordering
    the sequence of domain events matters to the WLM
    GNN discards sequence information by design
    inductive bias is wrong
```

### Why Mamba SSM Is Exactly Right

```
SSM hidden state mechanics:
    H_t = A(x_t) ⊙ H_{t-1} + B(x_t) ⊙ x_t
    y_t = C(x_t) ⊙ H_t
    
    A(x_t): input-dependent forgetting — selective memory
    B(x_t): input-dependent updating — selective learning
    C(x_t): input-dependent readout — selective output
    
Applied to WLM:

    H_t:     current MFT state — what WLM knows about the web
    H_{t-1}: previous MFT state — what WLM knew before this event
    x_t:     new DomainTopologyEvent — what the preparser just learned
    
    A(x_t):  how much existing structural knowledge to preserve
             when seeing a Cloudflare domain: A is high
             (Cloudflare patterns are consistent — remember them)
             when seeing a never-before-seen DSL topology: A is low
             (old patterns less relevant — update aggressively)
             
    B(x_t):  how much this new domain updates the hidden state
             high-signal domains (Wikipedia, Stripe docs) → high B
             low-signal domains (landing pages, auth redirects) → low B
             
    y_t:     current traversal policy for this topology class
             projected from hidden state through output heads
```

The SSM is the correct architecture because:

```
1. O(n) training, O(1) inference
   processing millions of domain events is O(n) not O(n²)
   reading the MFT at query time is O(1) — one matrix multiply

2. Fixed-size hidden state regardless of history length
   H is always (1, 256) whether 1 or 10 billion events seen
   the web gets encoded into 256 dimensions
   density increases, size does not

3. Selective state transitions
   Mamba learns WHICH domain events matter for updating H
   not every domain teaches the WLM something new
   the selective mechanism learns this automatically

4. Temporal ordering preserved
   the sequence of encounters shapes H
   seeing many Cloudflare domains early in crawl history
   correctly biases H toward expecting friction on new domains
   a GNN cannot capture this

5. Perfect semantic alignment with MFT concept
   the MFT is a compressed structural map of the web
   the SSM hidden state IS a compressed structural map
   they are the same thing by definition
   no impedance mismatch between architecture and concept
```

---

## The Model Architecture — Exact Specification

```python
class MambaRouter(nn.Module):
    """
    The World Latent Model.
    
    This model maintains a persistent hidden state H that encodes
    compressed structural knowledge of the web. H is the MFT.
    
    At training time (index_daemon.py):
        forward() is called with domain event sequences
        gradients flow through all parameters
        hidden_state is updated explicitly after each gradient step
        
    At inference time (wlm.query()):
        readout() is called — NOT forward()
        no gradient computation
        no hidden_state modification
        one matrix multiply per output head
        O(1) regardless of history length
        
    CRITICAL: hidden_state is a registered buffer, not a parameter.
    It is NOT updated by backprop. It is updated explicitly by
    index_daemon.py after gradient steps complete.
    This separation is load-bearing — query time must not
    accidentally update the MFT.
    """
    
    def __init__(
        self,
        vocab_size:    int = 8192,    # domain tokens + topology tokens + intent tokens
        d_model:       int = 256,     # hidden dimension — fits in 5080 VRAM trivially
        d_state:       int = 64,      # SSM state dimension — the MFT compression factor
        d_conv:        int = 4,       # local convolution width in Mamba block
        expand:        int = 2,       # inner expansion factor
        n_layers:      int = 4,       # four Mamba blocks
        n_topology:    int = 18,      # 18 topology classes
        n_source:      int = 512,     # source embedding space dimension
        n_phase:       int = 3,       # phases I, II, III
        dropout:       float = 0.1,
    ):
        super().__init__()
        
        # ── Input encoding ────────────────────────────────────────────────
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # maps discrete tokens to dense vectors
        # vocabulary covers:
        #   0-17:     topology class indices
        #   18-1023:  domain fingerprint tokens (hashed domain names)
        #   1024-4095: structural primitive tokens (CDN types, CMS types, etc)
        #   4096-8191: intent signal tokens (from intent_vector quantization)
        
        self.position_embedding = nn.Embedding(512, d_model)
        # positional encoding for within-sequence position
        # max sequence length 512 tokens per event
        
        self.domain_projection = nn.Linear(d_model, d_model)
        # projects domain-specific features into model space
        
        self.intent_projection = nn.Linear(256, d_model)
        # projects intent_vector (256-dim) into model space
        # intent_vector comes from AXIOM graph's query embedding
        
        self.dropout = nn.Dropout(dropout)
        
        # ── Mamba blocks ──────────────────────────────────────────────────
        
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        # four Mamba blocks
        # each block has its own A, B, C matrices
        # each block selectively updates its portion of the representation
        # block 0: low-level structural patterns (CDN, CMS, render requirements)
        # block 1: topology class relationships (parent/child class patterns)
        # block 2: friction and traversal patterns
        # block 3: source priority and quality patterns
        # this is a soft assignment — the model learns these divisions
        
        self.norm = nn.LayerNorm(d_model)
        
        # ── Output heads ──────────────────────────────────────────────────
        
        self.topology_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_topology),
        )
        # → 18 logits over topology classes
        # used at training time for topology prediction loss
        # not the primary output — TraversalPolicy is
        
        self.traversal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7),    # 7 traversal parameters
        )
        # → 7 outputs:
        #   [0] depth:               sigmoid → [1, 5] int
        #   [1] render_mode:         sigmoid > 0.5 → "headless" else "static"
        #   [2] requests_per_second: softplus → [0.1, 100.0]
        #   [3] retry_budget:        sigmoid → [0, 5] int
        #   [4] timeout_ms:          softplus → [1000, 30000]
        #   [5] tor_required:        sigmoid > 0.7 → True
        #   [6] confidence:          sigmoid → [0.0, 1.0]
        
        self.friction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 5),    # 5 friction outputs
        )
        # → 5 outputs, all sigmoid → [0.0, 1.0] probabilities:
        #   [0] cloudflare_probability
        #   [1] paywall_probability
        #   [2] rate_limit_probability
        #   [3] auth_redirect_probability
        #   [4] bot_detection_probability
        
        self.source_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_source),
        )
        # → 512-dim source embedding
        # used to rank candidate sources by dot product similarity
        # structural_layer.pt contains source embeddings for known domains
        # dot product(source_head output, structural_layer source matrix)
        # → ranked list of domain → URL pattern → signal zone
        
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_phase),
        )
        # → 3 logits over phases I, II, III
        # for a given topology class: what phase should it be in?
        # used by index_daemon to validate phase_states.mmap
        
        # ── THE MFT — persistent hidden state ─────────────────────────────
        
        self.register_buffer(
            "hidden_state",
            torch.zeros(1, d_model),
        )
        # THIS is the MFT.
        # (1, 256) tensor — fixed size forever
        # registered as buffer: saved in state_dict, not a gradient parameter
        # updated ONLY by index_daemon.py after gradient steps
        # NEVER updated during inference
        # NEVER updated by forward() calls at query time
        # the separation between training updates and inference reads
        # is the entire point of the buffer registration
        
        self.register_buffer(
            "hidden_state_version",
            torch.tensor(0, dtype=torch.long),
        )
        # monotonic counter — incremented by index_daemon on every update
        # used to detect stale reads
        # if classifier reads version N but watchdog says N+1 exists:
        # reload is in progress — use version N until reload completes
```

---

## The Vocabulary — What Gets Tokenized

The WLM operates on tokens. Understanding the vocabulary is essential to
understanding what the model actually learns.

```python
# wlm_tokenizer.py — used by latent_model.py internally

VOCAB = {
    # ── Topology class tokens (0-17) ──────────────────────────────────────
    "NEWS_ARTICLE":             0,
    "NEWS_ARTICLE_PAYWALLED":   1,
    "SAAS_DOCS":                2,
    "SAAS_DOCS_VERSIONED":      3,
    "SAAS_DOCS_WITH_CODE":      4,
    "REST_API_JSON":            5,
    "REST_API_JSON_PAGINATED":  6,
    "JSON_LD_STRUCTURED":       7,
    "ECOMMERCE_PRODUCT":        8,
    "ECOMMERCE_PRODUCT_VARIANT":9,
    "FORUM_THREAD":             10,
    "BLOG_POST":                11,
    "WIKIPEDIA_ARTICLE":        12,
    "LANDING_PAGE":             13,
    "AUTH_REDIRECT":            14,
    "CLOUDFLARE_CHALLENGE":     15,
    "RATE_LIMITED":             16,
    "GENERIC_HTML":             17,
    
    # ── Structural primitive tokens (18-1023) ─────────────────────────────
    # CDN fingerprints
    "CDN_CLOUDFLARE":           18,
    "CDN_FASTLY":               19,
    "CDN_AKAMAI":               20,
    "CDN_NONE":                 21,
    
    # CMS fingerprints
    "CMS_WORDPRESS":            22,
    "CMS_GHOST":                23,
    "CMS_GATSBY":               24,
    "CMS_NEXTJS":               25,
    "CMS_DOCUSAURUS":           26,   # SaaS docs common
    "CMS_GITBOOK":              27,   # SaaS docs common
    "CMS_NONE":                 28,
    
    # Render requirements
    "RENDER_STATIC":            29,
    "RENDER_HEADLESS":          30,
    "RENDER_TOR":               31,
    
    # Bot mitigation
    "BOT_NONE":                 32,
    "BOT_CLOUDFLARE":           33,
    "BOT_RECAPTCHA":            34,
    "BOT_HCAPTCHA":             35,
    "BOT_CUSTOM":               36,
    
    # TLS/SSL signals
    "TLS_VALID":                37,
    "TLS_EXPIRED":              38,
    "TLS_SELF_SIGNED":          39,
    
    # HTTP version signals
    "HTTP_1":                   40,
    "HTTP_2":                   41,
    "HTTP_3":                   42,
    
    # Response time buckets
    "LATENCY_FAST":             43,   # <100ms
    "LATENCY_MEDIUM":           44,   # 100-500ms
    "LATENCY_SLOW":             45,   # 500ms-2s
    "LATENCY_VERY_SLOW":        46,   # >2s
    
    # robots.txt signals
    "ROBOTS_COMPLIANT":         47,
    "ROBOTS_RESTRICTIVE":       48,   # many disallows
    "ROBOTS_PERMISSIVE":        49,   # few disallows
    "ROBOTS_MISSING":           50,
    
    # Sitemap signals
    "SITEMAP_PRESENT":          51,
    "SITEMAP_MISSING":          52,
    "SITEMAP_LARGE":            53,   # >10K URLs
    
    # ... 18-1023 continues with more structural primitives
    
    # ── Domain hash tokens (1024-4095) ────────────────────────────────────
    # domain names are hashed to 10-bit indices
    # docs.stripe.com → hash → 1024 + (hash % 3072)
    # collision is acceptable — domain proximity in hash space
    # is semantically meaningful (similar domains cluster)
    
    # ── Intent signal tokens (4096-8191) ─────────────────────────────────
    # intent_vector (256-dim float) is quantized to 2-bit per dimension
    # producing 512 bits → folded into 4096 token vocab space
    # this is approximate — captures intent direction not exact value
}
```

---

## Input Construction — What Gets Fed To The Model

### Training Input (DomainTopologyEvent → token sequence)

```python
def _encode_domain_topology_event(
    self,
    event: DomainTopologyEvent,
) -> torch.Tensor:
    """
    Converts a DomainTopologyEvent into a token sequence for the model.
    
    The token sequence structure:
    [topology_class_token]
    [cdn_token]
    [cms_token]
    [render_token]
    [bot_mitigation_token]
    [latency_bucket_token]
    [robots_signal_token]
    [sitemap_signal_token]
    [domain_hash_token]
    [tls_token]
    [http_version_token]
    [... up to 32 structural primitive tokens from DomainMap ...]
    
    Total: 11-43 tokens per event.
    Short sequences — the model processes these fast.
    The meaning is in the hidden state, not in sequence length.
    """
    tokens = []
    dm = event.domain_map
    
    # topology class
    for path_pattern, topology_class in dm.path_topology_map.items():
        if topology_class in VOCAB:
            tokens.append(VOCAB[topology_class])
    
    # CDN fingerprint
    cdn_token = self._detect_cdn(dm)
    tokens.append(cdn_token)
    
    # bot mitigation
    bot_token = VOCAB.get(f"BOT_{dm.bot_mitigation.upper()}", VOCAB["BOT_NONE"])
    tokens.append(bot_token)
    
    # robots.txt signal
    robots_token = self._classify_robots_signal(dm)
    tokens.append(robots_token)
    
    # sitemap signal
    sitemap_token = VOCAB["SITEMAP_PRESENT"] if dm.sitemap_urls else VOCAB["SITEMAP_MISSING"]
    tokens.append(sitemap_token)
    
    # domain hash
    domain_hash = int(hashlib.sha256(event.domain.encode()).hexdigest(), 16)
    domain_token = 1024 + (domain_hash % 3072)
    tokens.append(domain_token)
    
    # friction zones → render requirements
    render_token = self._infer_render_requirement(dm)
    tokens.append(render_token)
    
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    # shape: (1, seq_len)

def _encode_query_input(
    self,
    topology_class: str,
    intent_vector:  List[float],
) -> torch.Tensor:
    """
    Encodes a query-time input for conditioned readout.
    Used when topology class is known but confidence is low
    and we want to condition the WLM output on the specific intent.
    
    Shorter sequence than training input — query time is latency sensitive.
    """
    tokens = []
    
    # topology class token
    if topology_class in VOCAB:
        tokens.append(VOCAB[topology_class])
    
    # intent quantization — 8 tokens covering intent direction
    intent_tokens = self._quantize_intent(intent_vector, n_tokens=8)
    tokens.extend(intent_tokens)
    
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    # shape: (1, 9)  ← very short — fast forward pass
```

---

## The Two Forward Pass Modes

This is the most important distinction in the entire model.

### Mode 1: readout() — Query Time, O(1), No State Change

```python
def readout(
    self,
    topology_class: str,
    intent_vector:  Optional[List[float]] = None,
) -> WLMResponse:
    """
    THE CRITICAL PATH CALL.
    Called by interface.py on every query.
    Must be fast. Must not modify hidden_state.
    
    If intent_vector is None:
        pure readout from hidden_state
        one matrix multiply per head
        O(1) — does not depend on any input processing
        appropriate for Phase III known topology classes
        
    If intent_vector provided:
        conditioned readout
        intent projected into model space
        added to hidden_state before head projection
        still O(1) — no recurrence, no sequence processing
        appropriate for Phase I/II where intent matters
        
    INVARIANT: hidden_state is identical before and after this call.
    This invariant is enforced by torch.no_grad() and by never
    assigning to self.hidden_state in this method.
    
    Violation of this invariant = MFT corruption at query time.
    index_daemon is the ONLY writer of hidden_state. Always.
    """
    
    with torch.no_grad():
        
        if intent_vector is not None:
            # condition on intent
            intent_t = torch.tensor(
                intent_vector, dtype=torch.float32
            ).unsqueeze(0)                              # (1, 256)
            
            intent_projected = self.intent_projection(intent_t)  # (1, d_model)
            
            # add intent to hidden state WITHOUT modifying self.hidden_state
            # this is a local variable — not a state mutation
            conditioned_state = self.hidden_state + intent_projected
            conditioned_state = self.norm(conditioned_state)
        else:
            conditioned_state = self.norm(self.hidden_state)
        
        # project through output heads
        traversal_raw = self.traversal_head(conditioned_state)    # (1, 7)
        friction_raw  = self.friction_head(conditioned_state)     # (1, 5)
        source_raw    = self.source_head(conditioned_state)       # (1, 512)
        phase_raw     = self.phase_head(conditioned_state)        # (1, 3)
        
        # decode outputs
        traversal_policy = self._decode_traversal(
            traversal_raw.squeeze(0),
            topology_class,
        )
        friction_forecast = self._decode_friction(
            friction_raw.squeeze(0),
            topology_class,
        )
        source_priority = self._decode_source_priority(
            source_raw.squeeze(0),
            topology_class,
        )
        
    return WLMResponse(
        traversal_policy=traversal_policy,
        friction_forecast=friction_forecast,
        source_priority=source_priority,
        world_confidence=float(torch.sigmoid(phase_raw).max()),
    )
```

### Mode 2: forward() — Training Time, Updates State

```python
def forward(
    self,
    token_sequence: torch.Tensor,
    update_hidden:  bool = False,
) -> dict:
    """
    TRAINING CALL ONLY.
    Called by index_daemon.py during gradient step computation.
    
    update_hidden=False during gradient computation:
        allows gradient flow through the computation graph
        hidden_state not yet modified
        gradients computed first
        
    update_hidden=True after gradient step completes:
        explicitly updates self.hidden_state with new computed state
        this is the MFT update
        this is what makes AXIOM compound
        
    NEVER called from interface.py.
    NEVER called from classifier.py.
    NEVER called with update_hidden=True during a live query.
    
    If you are reading this and considering calling forward()
    from anywhere except index_daemon.py:
        Do not.
    """
    
    # token embedding + positional encoding
    positions = torch.arange(
        token_sequence.size(1),
        device=token_sequence.device,
    )
    
    x = self.token_embedding(token_sequence)      # (B, seq_len, d_model)
    x = x + self.position_embedding(positions)    # add positional encoding
    x = self.dropout(x)
    
    # pass through Mamba blocks
    # each block selectively updates the representation
    for block in self.blocks:
        x = block(x)
    
    # final normalization
    x = self.norm(x)
    
    # last token representation = new hidden state candidate
    new_hidden = x[:, -1, :]                      # (B, d_model)
    
    # compute all outputs from new hidden state
    outputs = {
        "topology_logits":  self.topology_head(new_hidden),
        "traversal_raw":    self.traversal_head(new_hidden),
        "friction_raw":     self.friction_head(new_hidden),
        "source_embedding": self.source_head(new_hidden),
        "phase_logits":     self.phase_head(new_hidden),
        "new_hidden":       new_hidden,
    }
    
    # MFT UPDATE — only when explicitly requested
    # index_daemon calls this after optimizer.step()
    if update_hidden:
        # detach from computation graph before storing
        # we do not want gradients flowing through the stored state
        self.hidden_state = new_hidden.detach().mean(dim=0, keepdim=True)
        # mean over batch dimension if B > 1
        # hidden_state is always (1, d_model)
        self.hidden_state_version += 1
    
    return outputs
```

---

## The Output Decoders — Exact Specification

The raw model outputs are unbounded tensors. These decoders convert them
to semantically meaningful values with correct ranges.

```python
def _decode_traversal(
    self,
    raw:            torch.Tensor,    # (7,)
    topology_class: str,
) -> TraversalPolicy:
    """
    Decodes raw traversal head output into TraversalPolicy.
    
    Each field has a specific activation and range.
    The ranges are not arbitrary — they reflect real-world constraints.
    
    depth:
        sigmoid(raw[0]) * 4 + 1 → [1.0, 5.0]
        rounded to int
        depth 1: fetch only the target URL
        depth 2: fetch target + one level of linked pages
        depth 3: rare, expensive, only for hub topology classes
        depth 4-5: almost never — only WIKIPEDIA_ARTICLE deep dives
        
    render_mode:
        sigmoid(raw[1]) > RENDER_THRESHOLD → "headless"
        RENDER_THRESHOLD = 0.60
        not 0.5 — prefer static (faster, cheaper)
        only switch to headless when model is confident it is needed
        
    requests_per_second:
        softplus(raw[2]) + 0.1 → [0.1, ∞) practically [0.1, 100.0]
        softplus = log(1 + exp(x)) — always positive, smooth
        clamped to [0.1, 100.0] after computation
        default ~1.0 for unknown domains
        
    retry_budget:
        sigmoid(raw[3]) * 5 → [0.0, 5.0]
        rounded to int → [0, 5]
        0 retries for CLOUDFLARE_CHALLENGE (retrying is futile)
        1-2 retries for normal topology classes
        3-5 retries only if friction probability is high and
        mitigation strategy is known
        
    timeout_ms:
        softplus(raw[4]) * 1000 + 1000 → [1000ms, ~30000ms]
        clamped to [1000, 30000]
        headless render requires higher timeout than static fetch
        this is handled by TOPOLOGY_TIMEOUT_BIAS constant below
        
    tor_required:
        sigmoid(raw[5]) > 0.70 → True
        high threshold — prefer non-Tor (faster)
        only True when model is very confident Tor is needed
        
    confidence:
        sigmoid(raw[6]) → [0.0, 1.0]
        how confident is the WLM in this traversal policy
        used by phantom.py to decide whether to apply policy strictly
        low confidence → phantom falls back to DEFAULT_TRAVERSAL_POLICY
    """
    
    # topology-specific bias adjustments
    # these are learned adjustments baked in as constants
    # validated against real crawl data
    bias = TOPOLOGY_TRAVERSAL_BIAS.get(topology_class, {})
    
    depth = max(1, min(5, round(
        float(torch.sigmoid(raw[0])) * 4 + 1
    )))
    
    render_threshold = bias.get("render_threshold", RENDER_THRESHOLD)
    render_mode = "headless" if float(torch.sigmoid(raw[1])) > render_threshold else "static"
    
    # always headless for known JS-heavy classes
    if topology_class in ALWAYS_HEADLESS_CLASSES:
        render_mode = "headless"
    
    # never headless for known static classes
    if topology_class in ALWAYS_STATIC_CLASSES:
        render_mode = "static"
    
    rps_raw = float(F.softplus(raw[2])) + 0.1
    rps = max(0.1, min(100.0, rps_raw))
    
    # respect robots.txt crawl-delay encoded in DomainMap
    # WLM never recommends faster than robots.txt allows
    # this is non-negotiable
    if topology_class in CONSERVATIVE_RPS_CLASSES:
        rps = min(rps, CONSERVATIVE_RPS_CEILING)
    
    retry_budget = max(0, min(5, round(
        float(torch.sigmoid(raw[3])) * 5
    )))
    
    # no retries for friction classes
    if topology_class in ZERO_RETRY_CLASSES:
        retry_budget = 0
    
    timeout_raw = float(F.softplus(raw[4])) * 1000 + 1000
    timeout_ms = max(1000, min(30000, int(timeout_raw)))
    
    # headless requires more time
    if render_mode == "headless":
        timeout_ms = max(timeout_ms, HEADLESS_MIN_TIMEOUT_MS)
    
    tor_required = float(torch.sigmoid(raw[5])) > TOR_THRESHOLD
    
    # .onion URLs always require Tor
    # this override is in phantom.py not here
    # WLM makes recommendations — phantom enforces hard constraints
    
    confidence = float(torch.sigmoid(raw[6]))
    
    return TraversalPolicy(
        topology_class=topology_class,
        depth=depth,
        render_mode=render_mode,
        requests_per_second=rps,
        retry_budget=retry_budget,
        timeout_ms=timeout_ms,
        tor_required=tor_required,
        confidence=confidence,
    )

def _decode_friction(
    self,
    raw:            torch.Tensor,    # (5,)
    topology_class: str,
) -> FrictionForecast:
    """
    Decodes raw friction head output into FrictionForecast.
    
    All outputs are sigmoid → [0.0, 1.0] probability.
    
    The friction head learns the relationship between
    topology class and friction type from crawl history.
    
    Known patterns that should emerge from training:
        SAAS_DOCS + Cloudflare: cloudflare_probability ~0.7
        NEWS_ARTICLE_PAYWALLED: paywall_probability ~0.95
        REST_API_JSON: rate_limit_probability ~0.6
        AUTH_REDIRECT: auth_redirect_probability ~0.99
        CLOUDFLARE_CHALLENGE: bot_detection_probability ~0.99
    
    The mitigation_strategy field is not a model output.
    It is derived from the probability outputs via rule table.
    The model predicts probabilities.
    The rule table translates probabilities to strategies.
    This is intentional — strategy derivation is deterministic,
    not learned. Only the probabilities are learned.
    """
    
    probs = torch.sigmoid(raw)
    
    cf_prob   = float(probs[0])
    pw_prob   = float(probs[1])
    rl_prob   = float(probs[2])
    auth_prob = float(probs[3])
    bot_prob  = float(probs[4])
    
    # derive mitigation strategy from probabilities
    # deterministic rule — not learned
    strategy = self._derive_mitigation_strategy(
        cf_prob, pw_prob, rl_prob, auth_prob, bot_prob, topology_class
    )
    
    return FrictionForecast(
        topology_class=topology_class,
        cloudflare_probability=cf_prob,
        paywall_probability=pw_prob,
        rate_limit_probability=rl_prob,
        auth_redirect_probability=auth_prob,
        bot_detection_probability=bot_prob,
        mitigation_strategy=strategy,
    )

def _decode_source_priority(
    self,
    source_embedding: torch.Tensor,    # (512,)
    topology_class:   str,
) -> List[str]:
    """
    Converts source head output into ranked domain list.
    
    The source head outputs a 512-dim embedding vector.
    structural_layer.pt contains a (n_domains, 512) matrix
    where each row is the embedding of a known domain.
    
    Dot product similarity between source_embedding and each
    domain row gives a relevance score for that domain.
    Top-K domains by score = source_priority list.
    
    This is NOT cosine similarity search over a corpus.
    It is a single matrix multiply on structural_layer.pt.
    O(n_domains) but n_domains is bounded — we know all crawled domains.
    After Wikipedia preparse: ~6.7M articles, ~50K unique domains.
    Matrix multiply on 5080: microseconds.
    
    The source_priority list tells phantom.py:
    "for this topology class and intent direction,
     try these domains first, in this order."
    """
    
    # load source matrix from structural_layer
    # this is cached in memory after first load
    source_matrix = self._structural_layer.source_matrix    # (n_domains, 512)
    domain_index  = self._structural_layer.domain_index     # List[str]
    
    # dot product similarity
    scores = torch.matmul(
        source_matrix,
        source_embedding.unsqueeze(1),
    ).squeeze(1)                                            # (n_domains,)
    
    # top-K domains
    K = 10
    top_k_indices = torch.topk(scores, min(K, len(domain_index))).indices
    
    return [domain_index[i] for i in top_k_indices.tolist()]
```

---

## Constants — Every Single One Defined

```python
# ── Render mode constants ──────────────────────────────────────────────────

RENDER_THRESHOLD = 0.60
# sigmoid threshold for headless vs static
# 0.60 not 0.50 — prefer static, it is faster

ALWAYS_HEADLESS_CLASSES = frozenset({
    "SAAS_DOCS_WITH_CODE",       # lazy-loaded code blocks
    "ECOMMERCE_PRODUCT",         # React rendering
    "ECOMMERCE_PRODUCT_VARIANT", # same
})

ALWAYS_STATIC_CLASSES = frozenset({
    "REST_API_JSON",             # JSON endpoint — never needs JS
    "REST_API_JSON_PAGINATED",   # same
    "JSON_LD_STRUCTURED",        # structured data — static
    "WIKIPEDIA_ARTICLE",         # Wikipedia is static HTML
    "AUTH_REDIRECT",             # redirect — static
    "CLOUDFLARE_CHALLENGE",      # challenge page — static
    "RATE_LIMITED",              # error page — static
})

HEADLESS_MIN_TIMEOUT_MS = 8000
# headless render minimum timeout
# Playwright needs time to execute JS
# 8s gives most SPA frameworks time to render

TOR_THRESHOLD = 0.70
# sigmoid threshold for tor_required
# high threshold — prefer clearnet

# ── Rate limiting constants ────────────────────────────────────────────────

CONSERVATIVE_RPS_CLASSES = frozenset({
    "NEWS_ARTICLE",
    "NEWS_ARTICLE_PAYWALLED",
    "BLOG_POST",
    "FORUM_THREAD",
})
# news sites and forums are rate-limit sensitive
# do not crawl them as aggressively

CONSERVATIVE_RPS_CEILING = 2.0
# maximum 2 requests/sec for conservative classes

ZERO_RETRY_CLASSES = frozenset({
    "AUTH_REDIRECT",
    "CLOUDFLARE_CHALLENGE",
    "RATE_LIMITED",
})
# retrying friction classes is futile and harmful
# one attempt, accept the result, move on

# ── Traversal bias by topology class ──────────────────────────────────────

TOPOLOGY_TRAVERSAL_BIAS: Dict[str, Dict[str, float]] = {
    "SAAS_DOCS": {
        "render_threshold": 0.45,    # more willing to use headless
        "depth_bias": 0.3,           # slightly deeper traversal
    },
    "REST_API_JSON": {
        "render_threshold": 0.95,    # almost never headless
        "depth_bias": -0.2,          # shallower — API endpoints are flat
    },
    "WIKIPEDIA_ARTICLE": {
        "render_threshold": 0.99,    # never headless
        "depth_bias": 0.5,           # deeper — Wikipedia has rich links
    },
    "ECOMMERCE_PRODUCT": {
        "render_threshold": 0.40,    # usually needs headless
        "depth_bias": -0.3,          # shallow — product pages are leaves
    },
}

# ── Source priority constants ──────────────────────────────────────────────

SOURCE_PRIORITY_TOP_K = 10
# return top 10 domains in source priority list

# ── Phase thresholds ──────────────────────────────────────────────────────
# imported from contracts.py — defined once, used everywhere

# ── Hidden state constants ─────────────────────────────────────────────────

HIDDEN_STATE_DIM = 256
# the MFT dimension
# 256 is chosen because:
#   large enough to encode rich structural primitives
#   small enough for O(1) readout at query time
#   fits in a single cache line on modern CPUs
#   2^8 — clean power of two

HIDDEN_STATE_UPDATE_DELAY_MS = 500
# after index_daemon runs gradient step
# wait 500ms before updating hidden_state
# allows in-flight queries to complete against current state
# store_watchdog debounce is also 500ms — intentional alignment
```

---

## The Bus Subscription — Training Time

```python
async def initialize(self, bus: CrawlerBus) -> None:
    """
    Called by cold_start.py.
    Loads weights, subscribes to bus, registers watchdog handler.
    """
    
    await self._load_weights()
    await self._load_structural_layer()
    
    # subscribe to domain topology events
    # this is how the WLM learns during offline crawl
    bus.subscribe(
        topic="domain_topology",
        group="world_model.wlm",
        handler=self._on_domain_topology,
        schema=DomainTopologyEvent,
    )
    
    # subscribe to surprise events
    # high surprise on a known domain = WLM prediction was wrong
    # WLM needs to update its traversal knowledge
    bus.subscribe(
        topic="surprise",
        group="world_model.wlm",
        handler=self._on_surprise,
        schema=SurpriseEvent,
    )
    
    # register watchdog handlers
    WATCHDOG.register(
        path="topology_router.pt",
        handler=self._reload_weights,
        debounce_ms=500,
    )
    WATCHDOG.register(
        path="structural_layer.pt",
        handler=self._reload_structural_layer,
        debounce_ms=500,
    )

async def _on_domain_topology(self, event: DomainTopologyEvent) -> None:
    """
    Background task — called by bus dispatcher.
    Never blocks the critical path.
    Never called during a live query.
    
    The WLM does not run gradient steps here.
    That is index_daemon's job.
    
    The WLM updates its domain knowledge cache here —
    a fast in-memory lookup that supplements the model
    for domains it has recently seen.
    """
    
    # encode the domain topology event
    tokens = self._encode_domain_topology_event(event)
    
    # run forward pass to get new representation
    # update_hidden=False — index_daemon owns hidden_state updates
    with torch.no_grad():
        outputs = self.model.forward(tokens, update_hidden=False)
    
    # cache the traversal policy for this domain
    # used for fast lookup before falling back to model readout
    traversal = self._decode_traversal(
        outputs["traversal_raw"].squeeze(0),
        topology_class=list(event.domain_map.path_topology_map.values())[0]
                       if event.domain_map.path_topology_map else "GENERIC_HTML",
    )
    
    self._domain_policy_cache[event.domain] = traversal
    
    self._log.info(
        "wlm.domain_topology_processed",
        domain=event.domain,
        topology_class=traversal.topology_class,
        render_mode=traversal.render_mode,
    )

async def _on_surprise(self, event: SurpriseEvent) -> None:
    """
    High surprise on a topology class means WLM's prediction was wrong.
    
    The WLM does not update itself here.
    That is index_daemon's job — gradient steps on topology_router.pt.
    
    The WLM invalidates its domain policy cache for the affected class.
    Next query for this topology class will use model readout,
    not the stale cached policy.
    """
    
    if event.dissolve_triggered:
        # invalidate all cached policies for this topology class
        to_remove = [
            domain for domain, policy in self._domain_policy_cache.items()
            if policy.topology_class == event.topology_class
        ]
        for domain in to_remove:
            del self._domain_policy_cache[domain]
        
        self._log.warning(
            "wlm.cache_invalidated_on_dissolve",
            topology_class=event.topology_class,
            invalidated_domains=len(to_remove),
        )
```

---

## The Query Time Interface — Critical Path

```python
async def query(
    self,
    topology_class: str,
    intent_vector:  Optional[List[float]] = None,
    domain:         Optional[str] = None,
) -> WLMResponse:
    """
    THE ONLY PUBLIC METHOD CALLED BY interface.py.
    
    Called in parallel with wlp.query() via asyncio.gather().
    Must complete in <2ms for Phase III known topology classes.
    Must complete in <5ms for Phase I unknown topology classes.
    
    Lookup hierarchy:
    
    1. Domain policy cache (microseconds)
       if domain is known and recently cached:
       return cached policy immediately
       no model call
       
    2. Topology class cache (microseconds)
       if topology_class has a cached policy:
       return cached policy
       no model call
       
    3. Model readout (milliseconds — still fast)
       hidden_state projected through output heads
       O(1) — one matrix multiply per head
       conditioned on intent_vector if provided
       
    The model is the fallback, not the primary path.
    For a warm system with good coverage:
    >90% of queries should hit cache level 1 or 2.
    Model readout should be rare.
    
    This method NEVER calls forward().
    This method NEVER modifies hidden_state.
    These are invariants. Not suggestions.
    """
    
    t0 = time.perf_counter()
    
    # L1: domain-specific cache
    if domain and domain in self._domain_policy_cache:
        cached = self._domain_policy_cache[domain]
        # update friction forecast from current model
        # friction can change even for known domains
        friction = self._get_friction_for_topology(topology_class)
        source   = self._get_source_priority(topology_class, intent_vector)
        
        return WLMResponse(
            traversal_policy=cached,
            friction_forecast=friction,
            source_priority=source,
            world_confidence=cached.confidence,
        )
    
    # L2: topology class cache
    if topology_class in self._topology_policy_cache:
        cached = self._topology_policy_cache[topology_class]
        friction = self._get_friction_for_topology(topology_class)
        source   = self._get_source_priority(topology_class, intent_vector)
        
        return WLMResponse(
            traversal_policy=cached,
            friction_forecast=friction,
            source_priority=source,
            world_confidence=cached.confidence,
        )
    
    # L3: model readout
    response = self.model.readout(
        topology_class=topology_class,
        intent_vector=intent_vector,
    )
    
    # cache the result for future queries
    self._topology_policy_cache[topology_class] = response.traversal_policy
    
    latency_ms = (time.perf_counter() - t0) * 1000
    self._log.info(
        "wlm.query_complete",
        topology_class=topology_class,
        render_mode=response.traversal_policy.render_mode,
        confidence=response.traversal_policy.confidence,
        latency_ms=round(latency_ms, 2),
        cache_level="model",
    )
    
    return response
```

---

## The Weight Loading And Reload Sequence

```python
async def _load_weights(self) -> None:
    """
    Loads topology_router.pt into self.model.
    Called by initialize() and by store_watchdog handler.
    
    Uses staging pattern — loads into temp model first,
    validates, then atomically swaps the reference.
    In-flight queries complete against old model.
    Next query uses new model.
    Python GIL makes the reference swap atomic.
    """
    
    state_dict_path = STORE_ROOT / "topology_router.pt"
    
    if not state_dict_path.exists():
        raise ClassifierModelNotInitialized(
            "topology_router.pt not found — run initialize_store.py first"
        )
    
    # load into temp model first
    temp_model = MambaRouter()
    state_dict = torch.load(
        state_dict_path,
        map_location=self._device,
        weights_only=True,    # security: never execute arbitrary code
    )
    temp_model.load_state_dict(state_dict)
    temp_model.eval()
    temp_model.to(self._device)
    
    # validate temp model produces sensible output
    # before replacing the live model
    self._validate_model(temp_model)
    
    # atomic reference swap — GIL-safe
    self._model = temp_model
    
    self._log.info(
        "wlm.weights_loaded",
        hidden_state_version=int(temp_model.hidden_state_version),
        device=str(self._device),
    )

def _validate_model(self, model: MambaRouter) -> None:
    """
    Validates that a freshly loaded model produces valid outputs.
    Called before replacing the live model.
    
    Does not validate semantic correctness — only structural correctness.
    The model's traversal policies are correct if and only if
    the training process was correct. We cannot validate that here.
    We can only validate that the outputs are in expected ranges.
    """
    
    with torch.no_grad():
        test_response = model.readout(
            topology_class="SAAS_DOCS",
            intent_vector=None,
        )
    
    assert 1 <= test_response.traversal_policy.depth <= 5
    assert test_response.traversal_policy.render_mode in {"static", "headless"}
    assert 0.1 <= test_response.traversal_policy.requests_per_second <= 100.0
    assert 0 <= test_response.traversal_policy.retry_budget <= 5
    assert 1000 <= test_response.traversal_policy.timeout_ms <= 30000
    assert 0.0 <= test_response.traversal_policy.confidence <= 1.0
    
    for prob in [
        test_response.friction_forecast.cloudflare_probability,
        test_response.friction_forecast.paywall_probability,
        test_response.friction_forecast.rate_limit_probability,
        test_response.friction_forecast.auth_redirect_probability,
    ]:
        assert 0.0 <= prob <= 1.0
    
    assert len(test_response.source_priority) > 0

async def _reload_weights(self) -> None:
    """
    Called by store_watchdog when topology_router.pt changes.
    Delegates to _load_weights() — same logic.
    
    The watchdog debounce (500ms) ensures this is called
    after the atomic rename completes, not during write.
    """
    self._log.info("wlm.reload_triggered")
    await self._load_weights()
    
    # invalidate all policy caches
    # cached policies from old weights are stale
    self._domain_policy_cache.clear()
    self._topology_policy_cache.clear()
    
    self._log.info("wlm.reload_complete")
```

---

## The Training Interface — For index_daemon.py

```python
def get_training_interface(self) -> "WLMTrainingInterface":
    """
    Returns a training interface for index_daemon.py.
    
    Separates the training API from the inference API.
    index_daemon imports WLMTrainingInterface.
    interface.py imports WLM directly.
    The training interface has access to forward() and hidden_state update.
    The inference interface does not.
    
    This is not security — it is clarity.
    Making the wrong call require going through a different interface
    makes accidental training during inference obvious and loud.
    """
    return WLMTrainingInterface(self)

class WLMTrainingInterface:
    """
    Used exclusively by index_daemon.py.
    Provides access to training-time operations.
    """
    
    def __init__(self, wlm: "WorldLatentModel"):
        self._wlm = wlm
    
    def get_model(self) -> MambaRouter:
        """Returns the model for gradient computation."""
        return self._wlm._model
    
    def update_hidden_state(self, new_hidden: torch.Tensor) -> None:
        """
        Called by index_daemon AFTER optimizer.step() completes.
        
        Sequence in index_daemon.py:
            1. loss = compute_loss(model.forward(tokens, update_hidden=False))
            2. loss.backward()
            3. optimizer.step()
            4. training_interface.update_hidden_state(outputs["new_hidden"])
            ← THIS call happens here, after gradients are applied
            5. save_checkpoint()
        
        The 500ms delay between step 3 and step 4 is enforced here.
        In-flight queries complete against current hidden state.
        After delay: hidden state updated atomically.
        """
        
        time.sleep(HIDDEN_STATE_UPDATE_DELAY_MS / 1000)
        
        # detach from any computation graph
        new_h = new_hidden.detach().mean(dim=0, keepdim=True)    # (1, d_model)
        
        # atomic assignment — GIL-safe
        self._wlm._model.hidden_state = new_h
        self._wlm._model.hidden_state_version += 1
        
        structlog.get_logger().info(
            "wlm.hidden_state_updated",
            version=int(self._wlm._model.hidden_state_version),
        )
    
    def save_checkpoint(self) -> None:
        """
        Saves updated weights to staging file.
        store_watchdog fires after atomic rename completes.
        All components reload automatically.
        """
        staging_path = STORE_ROOT / "staging" / "topology_router.pt.staging"
        final_path   = STORE_ROOT / "topology_router.pt"
        
        torch.save(
            self._wlm._model.state_dict(),
            staging_path,
        )
        
        # write SHA256 for integrity verification
        staging_bytes = staging_path.read_bytes()
        sha256 = hashlib.sha256(staging_bytes).hexdigest()
        (STORE_ROOT / "staging" / "topology_router.pt.sha256").write_text(sha256)
        
        # atomic rename — one syscall, cannot be interrupted
        os.rename(staging_path, final_path)
        # store_watchdog fires here
        # all registered handlers called after debounce
```

---

## The Mitigation Strategy Derivation

```python
def _derive_mitigation_strategy(
    self,
    cf_prob:        float,
    pw_prob:        float,
    rl_prob:        float,
    auth_prob:      float,
    bot_prob:       float,
    topology_class: str,
) -> str:
    """
    Deterministic rule-based derivation of mitigation strategy
    from probability outputs.
    
    This is NOT learned. It is a rule table.
    The probabilities are learned. The strategy is derived.
    
    Strategies and their meanings:
    
    "none":
        No significant friction expected.
        Proceed with standard fetch.
        
    "cloudflare_wait":
        Cloudflare challenge detected.
        Wait 5-10s, retry with realistic browser headers.
        phantom.py handles this in headless mode.
        
    "cloudflare_tor":
        Cloudflare with high bot detection.
        Switch to Tor — exit node looks residential.
        tor_fetcher.py handles this.
        
    "paywall_cached":
        Paywall detected.
        Check cache for previous successful extraction.
        If cache hit: return cached result.
        If cache miss: return extraction_empty=True.
        Do not attempt to bypass paywall.
        
    "rate_limit_backoff":
        Rate limiting likely.
        Apply exponential backoff.
        Respect crawl-delay from DomainMap.
        
    "auth_skip":
        Auth redirect.
        Do not attempt to authenticate.
        Return extraction_empty=True immediately.
        This URL provides no signal without credentials.
        
    "none_with_caution":
        Moderate friction probability across multiple types.
        Proceed but with conservative rate limiting.
    """
    
    # auth redirect — always skip, no mitigation possible
    if auth_prob > 0.85:
        return "auth_skip"
    
    # paywall — cache check, no bypass
    if pw_prob > 0.80:
        return "paywall_cached"
    
    # Cloudflare + high bot detection → Tor
    if cf_prob > 0.70 and bot_prob > 0.65:
        return "cloudflare_tor"
    
    # Cloudflare alone → wait and retry with headers
    if cf_prob > 0.60:
        return "cloudflare_wait"
    
    # rate limiting expected
    if rl_prob > 0.65:
        return "rate_limit_backoff"
    
    # moderate mixed friction
    total_friction = (cf_prob + pw_prob + rl_prob + auth_prob + bot_prob) / 5
    if total_friction > 0.35:
        return "none_with_caution"
    
    return "none"
```

---

## Cold Start Behavior

```python
async def cold_start_warmup(self) -> None:
    """
    Called by cold_start.py after weights are loaded.
    Pre-populates topology policy cache for all known classes.
    
    After this completes:
        All 18 topology classes have cached policies.
        First query for any topology class hits cache level 2.
        Model readout (level 3) only for genuinely unknown classes.
        
    This warmup takes <100ms on 5080.
    18 readout() calls × <5ms each = <90ms total.
    Acceptable cold start cost.
    """
    
    for topology_class in TOPOLOGY_CLASSES:
        try:
            response = self.model.readout(
                topology_class=topology_class,
                intent_vector=None,
            )
            self._topology_policy_cache[topology_class] = response.traversal_policy
        except Exception as exc:
            self._log.warning(
                "wlm.warmup_failed_for_class",
                topology_class=topology_class,
                error=str(exc),
            )
            # not fatal — model readout will handle it at query time
    
    self._log.info(
        "wlm.warmup_complete",
        cached_classes=len(self._topology_policy_cache),
    )
```

---

## The Structural Layer Integration

```python
async def _load_structural_layer(self) -> None:
    """
    Loads structural_layer.pt for source priority computation.
    
    structural_layer.pt contains:
        source_matrix:   (n_domains, 512)  — domain embeddings
        domain_index:    List[str]          — domain names in matrix row order
        intent_clusters: (n_clusters, 512) — intent cluster centroids
        cluster_domains: List[List[str]]    — domains per cluster
    
    Loaded into self._structural_layer for use by _decode_source_priority().
    
    Pre-populated by offline/encoders/topology_encoder.py
    with Wikipedia, arXiv, and docs site structural knowledge.
    The WLM arrives knowing where signal lives for major sources
    before the first query.
    """
    
    structural_path = STORE_ROOT / "structural_layer.pt"
    
    if not structural_path.exists():
        self._log.warning(
            "wlm.structural_layer_missing",
            msg="Source priority will be empty until preparse completes"
        )
        self._structural_layer = EmptyStructuralLayer()
        return
    
    data = torch.load(
        structural_path,
        map_location="cpu",    # structural layer lives on CPU
        weights_only=True,
    )
    
    self._structural_layer = StructuralLayerView(
        source_matrix=data["source_matrix"],
        domain_index=data["domain_index"],
        intent_clusters=data.get("intent_clusters"),
        cluster_domains=data.get("cluster_domains"),
    )
    
    self._log.info(
        "wlm.structural_layer_loaded",
        n_domains=len(data["domain_index"]),
    )

async def _reload_structural_layer(self) -> None:
    """
    Called by store_watchdog when structural_layer.pt changes.
    Preparse completed atomic swap — new domains now available.
    """
    self._log.info("wlm.structural_layer_reload_triggered")
    await self._load_structural_layer()
    
    # source priority caches are stale — clear them
    # topology policy caches are NOT cleared — traversal behavior
    # does not depend on which specific domains exist
    # only source_priority changes when new domains are added
    self._log.info("wlm.structural_layer_reload_complete")
```

---

## The Full Class Skeleton — Write Order For Opus

```
Write in this exact order.
Each function depends on what came before it.
Do not skip ahead.

1.  Module-level constants
        RENDER_THRESHOLD
        ALWAYS_HEADLESS_CLASSES
        ALWAYS_STATIC_CLASSES
        HEADLESS_MIN_TIMEOUT_MS
        TOR_THRESHOLD
        CONSERVATIVE_RPS_CLASSES
        CONSERVATIVE_RPS_CEILING
        ZERO_RETRY_CLASSES
        TOPOLOGY_TRAVERSAL_BIAS
        SOURCE_PRIORITY_TOP_K
        HIDDEN_STATE_DIM
        HIDDEN_STATE_UPDATE_DELAY_MS

2.  StructuralLayerView dataclass
        source_matrix, domain_index, intent_clusters, cluster_domains

3.  EmptyStructuralLayer class
        returns empty list from source priority calls
        used when structural_layer.pt does not exist yet

4.  MambaBlock class (if not importing from mamba_ssm directly)
        d_model, d_state, d_conv, expand
        selective SSM update: H_t = A(x)⊙H + B(x)⊙x
        output: C(x)⊙H_t

5.  MambaRouter class
        __init__() — full architecture as specified above
        register_buffer("hidden_state", zeros)
        register_buffer("hidden_state_version", 0)
        readout() — O(1), no state change, the critical path method
        forward() — training only, update_hidden parameter

6.  WLMTrainingInterface class
        get_model()
        update_hidden_state() — with 500ms delay
        save_checkpoint() — staging + atomic rename

7.  WorldLatentModel class (the public class — wraps MambaRouter)
        __init__() — device selection, cache initialization
        initialize() — load weights, subscribe bus, register watchdog
        query() — the critical path method, three-level cache
        _on_domain_topology() — bus handler
        _on_surprise() — bus handler
        _load_weights() — with validation
        _reload_weights() — watchdog handler
        _load_structural_layer()
        _reload_structural_layer()
        _encode_domain_topology_event() — tokenization
        _encode_query_input() — query tokenization
        _decode_traversal() — raw → TraversalPolicy
        _decode_friction() — raw → FrictionForecast
        _decode_source_priority() — embedding → domain list
        _derive_mitigation_strategy() — deterministic rules
        _classify_robots_signal() — DomainMap → robots token
        _detect_cdn() — DomainMap → CDN token
        _infer_render_requirement() — DomainMap → render token
        _quantize_intent() — float vector → token list
        _get_friction_for_topology() — cached friction lookup
        _get_source_priority() — cached source lookup
        _validate_model() — structural validation before swap
        cold_start_warmup() — pre-populate caches for all 18 classes
        get_training_interface() — returns WLMTrainingInterface
        health() → dict — for cold_start.py validation

8.  Module-level singleton
        WLM = WorldLatentModel()
```

---

## Non-Negotiable Implementation Rules

```
1. readout() never modifies hidden_state.
   Not sometimes. Not by accident. Never.
   torch.no_grad() is not sufficient — explicit no-assignment.
   
2. forward() is only called by index_daemon via WLMTrainingInterface.
   If forward() is called from anywhere else: architecture violation.
   
3. hidden_state update has 500ms delay after gradient step.
   In-flight queries must complete before state changes.
   HIDDEN_STATE_UPDATE_DELAY_MS = 500 is not arbitrary.
   
4. All probability outputs are sigmoid-bounded [0.0, 1.0].
   Raw model outputs are unbounded. Decoders apply sigmoid.
   No probability value ever leaves this file outside [0.0, 1.0].
   
5. TraversalPolicy.render_mode is always "static" or "headless".
   Never None. Never empty string. Never any other value.
   The always-static and always-headless class overrides are applied
   in _decode_traversal() regardless of model output.
   
6. The source priority list is never empty.
   If structural_layer.pt is missing: return ["GENERIC_FALLBACK"].
   If model output is all zeros: return top-K by frequency from crawl history.
   There is always a source to try.
   
7. weights_only=True on all torch.load() calls.
   Never load arbitrary Python objects from model files.
   Security requirement — not optional.
   
8. The WLM never makes network calls.
   It never calls phantom.py. It never fetches URLs.
   It predicts how to fetch. It does not fetch.
   This boundary is absolute.
   
9. Cache invalidation on surprise dissolve is immediate.
   A dissolved topology class policy is immediately wrong.
   The cache must not serve stale policies after dissolve.
   
10. cold_start_warmup() completes before interface.py accepts queries.
    The WLM must be warm before the first query arrives.
    cold_start.py enforces this via await.
```

---

## What The WLM Does Not Do

```
Does not fetch pages               phantom.py fetches
Does not classify URLs             classifier.py classifies
Does not compile recipes           parser.py compiles
Does not strip noise               signal_kernel/ strips
Does not detect surprise           surprise_detector.py detects
Does not run gradient steps        index_daemon.py trains
Does not write to store            index_daemon.py writes via training interface
Does not know about AXIOM graph    interface.py is the boundary
Does not communicate with WLP      they are parallel, independent
Does not block on bus handlers     all handlers are background tasks
Does not validate fetched content  that is the kernel's output, sanitizer's job
Does not make caching decisions    cache_manager.py decides
Does not know about Tor routing    it recommends tor_required=True
                                   tor_fetcher.py decides whether to honor it
```

---

## LOC Estimate

```
Module constants:                   ~80 loc
StructuralLayerView:                ~40 loc
EmptyStructuralLayer:               ~20 loc
MambaBlock (if custom):            ~150 loc
MambaRouter:                       ~300 loc
WLMTrainingInterface:              ~120 loc
WorldLatentModel.__init__:          ~80 loc
WorldLatentModel.initialize:        ~60 loc
WorldLatentModel.query:            ~100 loc
Bus handlers (x2):                  ~80 loc
Weight loading/reloading:          ~120 loc
Structural layer loading:           ~80 loc
Tokenization methods (x3):        ~150 loc
Decoder methods (x3):             ~250 loc
Mitigation strategy:                ~80 loc
Helper methods (x6):              ~150 loc
cold_start_warmup:                  ~40 loc
get_training_interface:             ~10 loc
health():                           ~30 loc
                                  ────────
Production total:                ~1,940 loc
Tests (target 1:1):              ~1,940 loc
                                  ────────
latent_model.py total:           ~3,880 loc
```

---

## What Success Looks Like

```
Functional correctness:
    readout() returns valid WLMResponse for all 18 topology classes
    all probability outputs in [0.0, 1.0]
    all traversal parameters in specified ranges
    render_mode is always "static" or "headless"
    hidden_state unchanged after readout()

Performance:
    query() L1 cache hit: <0.1ms
    query() L2 cache hit: <0.5ms
    query() model readout: <2ms on 5080
    cold_start_warmup: <100ms

Compounding:
    after 100 DomainTopologyEvents processed:
        domain policy cache has 100+ entries
        topology policy cache covers all encountered classes
        source priority list returns relevant domains
    
    after Wikipedia preparse:
        structural_layer.pt has 50K+ domain embeddings
        source priority for research queries returns Wikipedia
        source priority for API queries returns docs sites
    
    after 1 week of crawl:
        query() hits L1 cache for >80% of known domains
        model readout reserved for genuinely new topology
        WLM predictions measurably more accurate than week 1

Integration:
    called in parallel with WLP via asyncio.gather() — confirmed
    never blocks interface.py — confirmed
    store_watchdog triggers reload after index_daemon update — confirmed
    no network calls from latent_model.py — confirmed
    no imports from topology/ or phantom/ — confirmed
```

---

*World Latent Model — Complete Engineering Specification*
*The MFT is the hidden state. The hidden state is the MFT.*
*readout() reads the web. forward() teaches the web.*
*They are different operations on the same file.*
*AXIOM INTERNAL // DO NOT SURFACE*
