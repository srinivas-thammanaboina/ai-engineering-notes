# Hybrid Retrieval (Dense + Sparse, fused)

> One-line takeaway: dense (embeddings) and sparse (BM25) retrieval fail in *opposite* ways — dense misses exact/rare terms, sparse misses meaning — so running both and fusing their rankings (usually with Reciprocal Rank Fusion) catches what either alone would drop.

This file is the deeper companion to the "Pattern 2: Hybrid search" section of [advanced-rag.md](advanced-rag.md). That section gives the one-paragraph version; this one works the mechanics with numbers and is honest about when hybrid *doesn't* help. It's tied to the Module 02 project, where the `lexical` golden-set category exists precisely to measure a hybrid win (see `module-02-rag-app/notes/advanced/eval-notes.md`).

---

## 0. Overview — why two retrievers beat one

**Simple terms:** there are two completely different ways to ask "which chunks match this query?"

- **Dense / semantic** — embed the query and every chunk into vectors and compare by cosine similarity. Matches on *meaning*. "parts sourcing" finds a chunk about "supply chain" even with zero shared words.
- **Sparse / keyword (BM25)** — score a chunk by how many of the query's literal words appear in it, weighting rare words more. Matches on *exact strings*. "FDDEI" finds the chunk containing the literal token "FDDEI."

Their weaknesses are mirror images:

```
        DENSE (embeddings)                 SPARSE (BM25)
        ✓ synonyms, paraphrase             ✓ exact terms, acronyms, codes, names
        ✗ rare/opaque tokens               ✗ synonyms, paraphrase
        ✗ novel terms (post-training)      ✗ "meaning" — only sees surface words

                         ╲           ╱
                          ╲         ╱
                           HYBRID
                    run both, fuse the rankings
```

Because they fail on disjoint cases, the union of their strengths covers far more than either alone. That's the whole idea — not that hybrid is "smarter," but that it's *complementary*.

**How it's used:** retrieve a candidate list from each retriever independently (say top-50 each), then **fuse** the two ranked lists into one, and take the top-k of the fused list to hand the generator.

---

## 1. One plain example (the case that motivates it)

Query: **"What does NVIDIA disclose about FDDEI?"** (a tax acronym — foreign-derived deduction eligible income).

- **Dense whiffs.** "FDDEI" wasn't a meaningful token in the embedding model's training data, so its vector is near-random. Cosine similarity to the real answer chunk is no better than to dozens of unrelated tax chunks. The answer doesn't appear in dense's top-10. (This is a real result from the Module 02 corpus — `lexical` category, Q22.)
- **BM25 nails it.** Only two chunks in 678 contain the literal string "FDDEI." BM25 gives "FDDEI" a huge weight *because* it's rare (see IDF below), so those two chunks rocket to ranks 1–2 of the sparse list.
- **Hybrid wins.** Fuse the two lists and the answer chunk — top-ranked in the sparse lane, absent in the dense lane — lands inside the top-5 of the fused list. The generator finally sees it.

Flip the query to **"What macroeconomic factors affect Apple's results?"** and it reverses: no single rare keyword, the answer is phrased in paraphrase ("global economic conditions," "FX headwinds"), dense nails it and BM25 flails. Hybrid covers both queries with one pipeline.

---

## 2. BM25 — how the sparse score actually works

**Overview:** BM25 ("Best Match 25") is a decades-old keyword ranking function. It scores a (query, chunk) pair by summing, over each query term, three intuitions:

1. **Term frequency (TF)** — the more times the term appears in the chunk, the higher the score… but with *diminishing returns* (10 occurrences isn't 10× better than 1).
2. **Inverse document frequency (IDF)** — a term that appears in *few* chunks is more discriminating, so it's weighted more. "FDDEI" (in 2 of 678 chunks) weighs enormously; "the" (in all 678) weighs ~nothing.
3. **Length normalization** — a term appearing in a long chunk counts a little less than in a short one (long chunks match everything by sheer size).

**The formula** (for query term $q$ in chunk $d$):

$$\text{score}(q, d) = \text{IDF}(q)\cdot\frac{f(q,d)\,(k_1+1)}{f(q,d) + k_1\big(1 - b + b\,\frac{|d|}{\text{avgdl}}\big)}$$

- $f(q,d)$ = how many times $q$ appears in $d$ (raw term frequency).
- $|d|$ = chunk length in words; $\text{avgdl}$ = average chunk length in the corpus.
- $k_1$ (≈1.2–1.5) controls **TF saturation** — how fast extra occurrences stop helping. Higher $k_1$ = TF matters more.
- $b$ (≈0.75) controls **length normalization** — $b=0$ ignores length, $b=1$ fully normalizes.

The total BM25 score for a chunk is the sum of this over all query terms.

**Worked intuition:** query "FDDEI tax". Corpus of 678 chunks.
- "FDDEI" appears in 2 chunks → IDF is large (rare = discriminating).
- "tax" appears in ~120 chunks → IDF is small (common = weak signal).
- A chunk containing "FDDEI" twice and "tax" three times scores high *mostly because of the FDDEI term* — the rare word dominates. That's exactly the behavior you want: the discriminating token drives the ranking.

**Key consideration — tokenization decides everything.** BM25 sees only surface tokens, so how you split text matters. "FDDEI", "GDPR", "Section 232" must survive tokenization as findable tokens (lowercase + split on non-alphanumerics is a safe default). Get tokenization wrong — strip digits, over-stem, split on hyphens badly — and the exact-match advantage evaporates.

---

## 3. Fusing the two lists — Reciprocal Rank Fusion (RRF)

**The problem:** dense returns cosine similarities (0–1); BM25 returns unbounded scores on a totally different scale (could be 4.2, could be 38). You can't just add them — a BM25 score of 12 doesn't "mean" the same as a cosine of 0.12.

**Two ways to combine:**

### 3a. Reciprocal Rank Fusion (the default)

**Overview:** throw the scores away entirely and use only each document's *rank* in each list. A document's fused score is the sum, across lists, of $1/(k + \text{rank})$ with a constant $k \approx 60$.

```python
def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(dense_ids):    # rank = 0, 1, 2, ...
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

**Why it works:** because it's rank-only, the cosine-vs-BM25 scale mismatch *disappears* — there's nothing to normalize. A document ranked highly in *either* list gets a good score; a document ranked highly in *both* gets boosted (its contributions add). The constant $k$ softens the dominance of the very top ranks (a larger $k$ makes ranks 1 vs 2 vs 3 matter less relative to merely being present).

**Worked example** (k=60, ranks shown 1-indexed for readability). Query = the FDDEI question.

| doc | dense rank | sparse rank | RRF score | |
|---|---|---|---|---|
| **ANSWER** (FDDEI chunk) | — (absent) | 1 | 1/61 = **0.0164** | surfaces! |
| D1 (a tax chunk) | 1 | 3 | 1/61 + 1/63 = **0.0323** | in both → boosted |
| D2 | 2 | — | 1/62 = 0.0161 | |
| S2 (another keyword hit) | — | 2 | 1/62 = 0.0161 | |

Fused order: **D1, ANSWER, D2 ≈ S2, …** The answer chunk — *invisible to dense, #1 in sparse* — lands at rank 2 of the fused list, inside the generator's top-5 window. That's the hybrid win, made concrete. Note D1 ranks top *because it appears in both lanes* — RRF rewards cross-lane agreement.

**Key consideration:** RRF discards score *magnitude*. A document BM25 ranked #1 by a mile and one it ranked #1 by a hair look identical to RRF. You trade confidence information for robustness and zero tuning. In practice that trade is usually worth it — which is why RRF is the production default (Elastic, Qdrant, Weaviate ship it).

### 3b. Weighted score sum (the alternative)

**Overview:** normalize each list's scores to a common range (e.g. min-max to [0,1]), then take $\alpha\cdot\text{dense} + (1-\alpha)\cdot\text{sparse}$.

**Tradeoff:** it *keeps* magnitude (a runaway-confident match stays dominant), but you now own two headaches — choosing a normalization that's stable across queries, and tuning $\alpha$. And $\alpha$ is corpus- and query-mix-dependent, so it's a knob that needs a golden set to set honestly. Reach for this only when RRF leaves measurable quality on the table; start with RRF.

---

### 3c. When RRF fails — the one-lane cap (a real result from the project)

RRF's whole premise is that the right answer shows up in *both* lanes (dense ranks it low, sparse ranks it high, agreement surfaces it). That premise quietly breaks when **one retriever is *blind* to the query** — not ranking the answer low, but missing it entirely. That's exactly the opaque-token case hybrid exists for, so it's worth seeing the failure concretely.

**The setup (real numbers from the Filing Analyst).** Query *"What does NVIDIA disclose about FDDEI?"* The answer chunk:

- BM25 (after stripping stopwords from the query — see §2) ranks it **#1**.
- Dense ranks it **#87 of 278** — effectively blind; "FDDEI" is an opaque token with no useful learned vector. It's nowhere near the candidate pool, so it appears in the dense lane *not at all*.

Now fuse (k=60, ranks 0-indexed):

```
ANSWER chunk:    BM25 rank 1, dense absent
   RRF = 1/(60+0)                  = 0.01667        ← one lane only

generic chunk:   dense rank 1  +  BM25 rank 5
   RRF = 1/(60+0) + 1/(60+4)       = 0.0323         ← two lanes
```

The answer got the **best score a single-lane document can possibly earn** (1/60). It still loses — because any two-lane document adds a *second* term on top. A two-lane doc's floor (say dense #1 + BM25 #50) is `1/60 + 1/110 ≈ 0.0259`, which already beats the one-lane ceiling of `0.0167`. So:

> **RRF rewards lane-*count* far more than rank-*position*. Being in two lanes beats being #1 in one lane.** And the large `k` flattens position further — within a lane, rank 1 (0.0167) vs rank 50 (0.0091) barely differ, so "how many lanes" dominates "how high."

That is fatal when the answer can *only ever* be in one lane: dense is structurally blind to the token, so the chunk can never become a two-lane document, so RRF caps it below the generic two-lane chunks forever. Cleaning the BM25 query (§2) is *necessary* — it gets the answer to BM25 rank 1 — but *not sufficient*: rank 1 in one lane still can't clear the cap.

**Subtlety:** lowering `k` doesn't save you. A smaller `k` sharpens position *within* a lane, but a two-lane chunk still carries the one-lane chunk's contribution *plus its own second term* — it dominates at every `k`. The cap is structural, not a tuning artifact.

**The fix — guaranteed-slot fusion (round-robin interleave).** Instead of scoring by consensus, take the top picks from each lane *alternately*: dense#1, sparse#1, dense#2, sparse#2, … (dedupe). This **guarantees each lane a slot regardless of the other lane's opinion**, so a one-lane answer survives — in the example above the answer lands at position 2. It's the same merge used for query decomposition (give each sub-query guaranteed representation).

The tradeoff is the mirror image of RRF's:

| fusion | dense-*blind* answer (opaque token) | categories dense already wins |
|---|---|---|
| **RRF** (consensus) | ✗ one-lane cap buries it | ✓ agreement protects them |
| **Round-robin** (guaranteed slots) | ✓ forced slot surfaces it | ✗ forces BM25's picks in → can inject noise |

Round-robin rescues the case RRF can't, but on a *semantic* question — where BM25 is mostly noise — it forces BM25's top picks into the output and may displace a good dense hit. Neither dominates; **which one wins is an empirical question you settle on a golden set, per category**, not by argument. (This is why the project added a `--fusion {rrf,interleave}` knob and measured both.)

**The general lesson:** RRF is the right default when both retrievers are *competent* on the query and you want their consensus. It is the *wrong* tool when one retriever is *blind* to a class of queries and you specifically need the other lane's lonely-but-correct pick to survive — there you want guaranteed lane representation, not consensus. Knowing *why* your two retrievers disagree (low-rank vs blind) tells you which fusion to reach for.

---

## 4. The wrinkle nobody mentions — metadata filtering

Dense stores (Chroma, Qdrant) filter by metadata natively (`where ticker = "NVDA"`). BM25 indexes usually don't — they're just term→document maps. So in a system that filters retrieval by company/date/section, the two lanes must apply the *same* filter or the comparison is dishonest and the fused list leaks off-target chunks.

Practical options: (a) build a separate BM25 index per filter value (fine for a handful of tickers); or (b) retrieve a large BM25 pool and post-filter by chunk metadata before fusing — pull *extra* (e.g. 200) so enough survives the filter. Whichever you pick, **both lanes must respect the filter identically.**

---

## 5. Limitations — when hybrid doesn't help (or hurts)

Honest list, because "add hybrid" is cargo-culted constantly:

- **BM25 is literal — it adds no understanding.** If the answer is pure paraphrase with no shared rare token, sparse contributes nothing useful and may inject keyword-matched-but-off-topic noise. Hybrid helps the *exact-term* failure, not the *semantic* one.
- **It can drag down what dense already aced.** Injecting a sparse lane risks pushing a keyword-matching-but-irrelevant chunk into the window on questions dense already answered perfectly. RRF keeps dense voting so the damage is usually small — but you must **measure the collateral**, not assume it's zero. (In the Module 02 eval this is exactly why we report per-category: a `lexical` win that costs `semantic` recall is not a free win.)
- **It does not fix enumeration / multi-aspect questions.** "What end markets does NVIDIA serve, and what does each cover?" needs *several* chunks, one per market — that's a coverage problem (decomposition / multi-query), not a keyword problem. Hybrid won't move it.
- **Tokenization fragility** (§2) — the entire BM25 advantage rides on the rare token surviving tokenization. Acronyms, hyphenation, casing, and digits are where it silently breaks.
- **RRF throws away confidence** (§3a) — you lose the "how sure" signal. Usually fine, occasionally matters.
- **RRF can't rescue a *blind*-lane answer** (§3c) — when one retriever misses the answer entirely (not just ranks it low), RRF's one-lane cap buries it under generic two-lane chunks. Reach for guaranteed-slot fusion (round-robin) for that case.
- **Operational cost** — two indexes to build, keep in sync on every corpus update, and query per request. More moving parts than naive dense.
- **It is not guaranteed to win on your corpus — you must measure.** On a corpus where every question is semantic, hybrid is dead weight. The only way to know is a golden set with *exact-term/opaque-token* questions in it. A set where dense already hits everything (hit@5=1.00 everywhere) is structurally blind to a hybrid win — see the v2 golden-set story in the project notes.

---

## 6. Where it sits in the advanced-RAG stack

The usual order is **query rewrite → hybrid retrieve → re-rank → generate**:

- **Hybrid** widens *recall* — it gets the right chunk into the candidate pool when dense alone would miss it (a recall fix).
- **Re-ranking** then fixes *precision/order* — a cross-encoder re-scores the fused pool and lifts the best chunk to the top (a ranking fix).

They're complementary: hybrid can't reorder within a lane well, and re-ranking can't rescue a chunk that was never retrieved. Hybrid raises the ceiling; re-ranking reaches it.

---

## Quick reference

| | Dense (embeddings) | Sparse (BM25) | Hybrid (RRF) |
|---|---|---|---|
| Matches on | meaning / synonyms | exact tokens | both |
| Wins on | paraphrase, famous terms | acronyms, codes, rare/novel terms | the union |
| Loses on | rare/opaque tokens | synonyms, paraphrase | (covers each other) |
| Score scale | cosine 0–1 | unbounded | rank-only (RRF) |
| Tuning needed | — | tokenizer | $k$ (mild), or $\alpha$ if weighted-sum |

**Workflow tip:** add hybrid only after a baseline dense eval, and only if your golden set contains exact-term/opaque-token questions that can *see* the win. Fuse with RRF first (no tuning); reach for weighted-sum only if measurement shows RRF leaving quality on the table.
