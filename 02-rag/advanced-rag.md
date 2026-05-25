# Advanced RAG

> The refinement layer. Naive RAG (previous file) retrieves *something* relevant; these patterns make it retrieve *the right thing* in messy, real-world documents like filings.

## The one-sentence truth

Naive top-k semantic search has three predictable failure modes — it misses exact terms, its cut is blunt, and it inherits messy queries. Re-ranking, hybrid search, and query rewriting each fix one.

## The problem we're solving

From the end of the last file: pure semantic search matches *meaning*, which is powerful but has gaps. Three concrete failures on the Filing Analyst:

1. A query for the literal phrase **"substantial doubt"** (the going-concern red flag) doesn't reliably surface the chunk containing it, because embeddings blur exact strings into meaning.
2. The genuinely best chunk ranks **#7**, just outside your `k=5` window — so the answer was retrievable but you never showed it to the model.
3. A user asks something **vague** ("how's Tesla doing?") and gets vague chunks back.

Each advanced pattern targets one of these.

---

## Pattern 1: Re-ranking (fixes the blunt cut)

The tension: you want high *recall* (don't miss the right chunk) but also high *precision* (don't flood the context with noise). Naive RAG forces a tradeoff via k. Re-ranking breaks the tradeoff with a **two-stage retrieve**:

```
Stage 1: RETRIEVE WIDE          Stage 2: RE-RANK NARROW
(fast, approximate)             (slow, accurate)

query ──▶ vector search ──▶ top 25 candidates ──▶ cross-encoder ──▶ top 4
          (bi-encoder,         (high recall,        (scores each       (high
           cheap, HNSW)         some noise)          query+chunk        precision,
                                                     pair directly)     fed to LLM)
```

### Bi-encoder vs cross-encoder (the key distinction)

- A **bi-encoder** (your embedding model) encodes the query and each chunk *separately*, then compares vectors. Fast — chunks are pre-embedded — but it never sees query and chunk *together*, so it misses fine interactions.
- A **cross-encoder** takes `(query, chunk)` as a *single* input and outputs one relevance score. It can model exactly how well *this* chunk answers *this* query — much more accurate. But it can't pre-compute anything (the score depends on the pair), so it's far too slow to run over your whole corpus.

The combo is the insight: use the cheap bi-encoder to get 25 candidates fast, then the expensive cross-encoder to *re-order* just those 25. You pay the slow model on 25 items, not 50,000. The chunk that ranked #7 in stage 1 can jump to #2 after re-ranking — now it's in the prompt.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

candidates = coll.query(query_embeddings=[q_vec], n_results=25)["documents"][0]
scored = reranker.predict([(question, chunk) for chunk in candidates])
top4 = [c for _, c in sorted(zip(scored, candidates), reverse=True)][:4]
# top4 → the chunks you actually put in the prompt
```

Re-ranking is usually the **single highest-ROI upgrade** to a naive RAG system. If you add one advanced pattern, add this.

---

## Pattern 2: Hybrid search (fixes the missed exact terms)

Semantic search and keyword search fail in *opposite* ways, which is exactly why combining them works:

- **Dense (semantic / embeddings)** — great at meaning and synonyms ("parts sourcing" ↔ "supply chain"), bad at exact strings, codes, names, rare jargon.
- **Sparse (keyword / BM25)** — great at exact terms ("substantial doubt", "Item 1A", a specific product SKU), bad at synonyms and paraphrase.

```
          DENSE (embeddings)              SPARSE (BM25)
          ✓ synonyms, paraphrase          ✓ exact phrases, codes, names
          ✗ exact strings                 ✗ synonyms

                        ╲           ╱
                         ╲         ╱
                          HYBRID
                   run both, fuse the rankings
                   (e.g. Reciprocal Rank Fusion)
```

**BM25** is the classic keyword-ranking algorithm — it scores a chunk by how often the query's terms appear in it, dampened by how common those terms are across the corpus (rare words count more). It's decades old, fast, and still excellent at literal matching.

You run both retrievers and **fuse** their result lists. The common fusion method is **Reciprocal Rank Fusion (RRF)** — it combines rankings by position rather than raw scores (so you don't have to normalize a cosine score against a BM25 score, which live on different scales):

```python
def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)   # fused ranking
```

For the Filing Analyst this matters a lot: questions about filings mix the semantic ("what are the risks") with the literal (specific dollar figures, exact legal phrases, section names). Hybrid catches both. Many production vector DBs (Qdrant, Weaviate, pgvector + extensions) offer hybrid search natively so you don't hand-roll the fusion.

---

## Pattern 3: Query rewriting (fixes the messy query)

The retriever can only be as good as the query it's given. Real user questions are often bad retrieval queries — vague, conversational, multi-part, or context-dependent. So you transform the query *before* retrieving. Three flavors:

**Rewriting / clarification** — turn a conversational question into a clean retrieval query. `"how's Tesla doing?"` → `"Tesla 2025 financial performance revenue profit risk factors"`.

**Decomposition** — split a multi-part question into sub-queries, retrieve for each, combine. `"Compare Tesla and Ford's EV risks"` → `["Tesla EV risk factors 2025", "Ford EV risk factors 2025"]`. (This is the seed of multi-document RAG and starts to feel agentic — it carries straight into Module 3.)

**HyDE (Hypothetical Document Embeddings)** — a clever trick: ask the LLM to *write a fake answer* to the question, then embed *that* and retrieve with it. The hypothetical answer is written in the same style/vocabulary as the real document, so it lands closer in vector space than the terse question would.

```
question ──▶ LLM writes a plausible (possibly wrong) answer ──▶ embed the answer
                                                                      │
                                                                      ▼
                                                    retrieve chunks near it
            (the fake answer "looks like" a real filing passage, so it matches better)
```

The "context-dependent" case is the one you'll hit constantly in a chat copilot: a follow-up like `"what about their supply chain?"` is meaningless to a retriever without the prior turn. You rewrite it using conversation history into a standalone query: `"Tesla supply chain risk factors 2025"`.

---

## How these stack

A production-grade RAG query path uses all three, in order:

```
user question
      │
      ▼
[ query rewriting ]   ── clean, standalone, maybe decomposed
      │
      ▼
[ hybrid search ]     ── dense + sparse, fused → ~25 candidates (high recall)
      │
      ▼
[ re-ranking ]        ── cross-encoder → top 4 (high precision)
      │
      ▼
[ augment + generate ]── context-engineered prompt → grounded, cited answer
```

You don't build all of this at once. Build naive RAG first (previous file), confirm it works, then add re-ranking (biggest win), then hybrid (for exact-term questions), then query rewriting (for the chat experience).

---

## The unavoidable question: how do you *know* it improved?

Every pattern here is a claimed improvement — but "I added re-ranking and it feels better" is not engineering. The honest answer is you measure it: build a small set of question→correct-chunk pairs (a **golden dataset**), and check whether each change actually retrieves the right chunks more often (recall@k, precision@k) and whether the final answers are more faithful to the sources.

That measurement discipline — golden datasets, faithfulness scoring, LLM-as-a-judge — is **Module 5**. For now, just plant the flag: *don't trust a RAG improvement you haven't measured.* When we build, we'll set up a tiny eval harness alongside the copilot so every upgrade is justified by numbers, not vibes.

## Mental model to carry forward

- Naive RAG fails three ways: **missed exact terms, blunt cut, messy queries.**
- **Re-ranking** = retrieve wide (bi-encoder) then narrow (cross-encoder). Highest ROI single upgrade.
- **Hybrid search** = dense + sparse (BM25), fused with RRF. Catches meaning *and* exact strings.
- **Query rewriting** (rewrite / decompose / HyDE) fixes bad queries before retrieval.
- Stack order: **rewrite → hybrid → re-rank → generate.**
- **Measure every change** — that's Module 5, and it's not optional.
