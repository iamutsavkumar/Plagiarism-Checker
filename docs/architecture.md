# Architecture Decision Record

## ADR-001 — Multi-Strategy Similarity Blending

**Status:** Accepted  
**Date:** 2025

### Context

Single-strategy plagiarism detectors fail in predictable ways:
- Jaccard shingling misses synonym substitution and sentence reordering.
- TF-IDF cosine fails when a paraphraser uses entirely different vocabulary.
- Semantic embeddings alone give false positives on topically related but
  independently written content (e.g., two separate summaries of the same
  Wikipedia article).

### Decision

Blend three strategies with tunable weights:

| Strategy | Strength | Weight |
|---|---|---|
| Jaccard | Catches exact/near-exact copies fast | 25% |
| TF-IDF cosine | Shared-vocabulary paraphrasing | 40% |
| Semantic embeddings | Deep semantic equivalence | 35% |

Weights auto-adjust when `sentence-transformers` is not installed
(Jaccard 38%, TF-IDF 62%).

### Consequences

**Positive**
- Catches all common plagiarism vectors.
- Gracefully degrades on systems without GPU / heavy dependencies.
- Transparent: all three sub-scores are returned to the client.

**Negative**
- First request after server start may take ~5 s while the MiniLM model
  loads into memory (subsequent requests are fast).
- Docker image is large (~2.5 GB) due to PyTorch.

---

## ADR-002 — Frontend: Vanilla JS, No Framework

### Context

The UI is primarily a form + results view with no complex client-side routing,
no real-time subscriptions, and no shared component tree requiring a virtual DOM.

### Decision

Pure HTML + CSS + vanilla JS. Zero build step, zero bundler, zero framework.

### Consequences

**Positive**
- Deployable as static files served by FastAPI's `StaticFiles`.
- No npm, no webpack, no version churn.
- Easy to fork and understand without framework knowledge.

**Negative**
- Manual DOM manipulation is more verbose than JSX for complex UIs.
- No component reuse primitives (mitigated by keeping the UI scope small).

---

## ADR-003 — Sentence Alignment Algorithm

### Context

Document-level similarity alone doesn't tell the user *which* passages
are suspicious. Sentence-level alignment is required for the highlight view.

### Decision

Use a greedy one-to-one matching approach:
1. Build a TF-IDF matrix over all sentences from both documents.
2. For each sentence in Doc A, find the highest-scoring unmatched sentence
   in Doc B above threshold (0.55).
3. For borderline pairs (0.30–0.55), run the semantic model as a fallback.
4. Each sentence in Doc B can only be matched once (prevents many-to-one).

### Consequences

Greedy matching is O(n × m) in TF-IDF distance and avoids the O(2^n)
combinatorial cost of optimal global alignment, which is acceptable for
typical document lengths (< 500 sentences).

---

## ADR-004 — File Size and Text Length Limits

| Limit | Value | Reason |
|---|---|---|
| Max file size | 5 MB | Prevents memory exhaustion on the server |
| Max text length | 50,000 chars | ~10k words; TF-IDF matrix stays manageable |
| Min text length | 20 chars | Prevents trivially empty comparisons |

These limits are enforced at both the Pydantic validation layer (text endpoint)
and the route handler (file endpoint).
