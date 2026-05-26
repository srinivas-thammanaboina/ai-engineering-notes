# Module 02 — RAG & Context Engineering: Personal Notes & Q&A

All questions, clarifications, and learnings from working through this module go here.

---

## Embeddings

### Q: What is cosine similarity, and how does it determine whether two chunks of text are semantically related?

**The core idea:**

After Stage 2 (Embeddings), every chunk of text is a vector — a list of numbers representing its meaning. To find which chunks are most relevant to a query, you need a way to measure how similar two vectors are.

You could measure straight-line distance (Euclidean distance), but that has a problem: a long document and a short document about the same topic produce vectors of very different magnitudes, so distance would wrongly call them dissimilar. Cosine similarity solves this by **ignoring magnitude entirely** — it only measures the angle between the two vectors.

```
Direction = meaning
Magnitude = how much text / how strong the signal

Cosine similarity cares only about direction.
```

---

**The geometry:**

```
        chunk: "supply chain disruption"
              ●
             /
            /  small angle → high similarity (0.95)
           /
origin ●──────────────● query: "problems sourcing parts"


        chunk: "quarterly dividend policy"
              ●
             /
            /  large angle → low similarity (0.12)
           /
origin ●──────────────● query: "problems sourcing parts"
```

Same query vector. The supply chain chunk points nearly the same direction → similar. The dividend chunk points a very different direction → dissimilar.

---

**The math:**

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

- `A · B` = dot product — multiply matching positions, then sum them all up
- `|A|` = magnitude of A — square root of the sum of squares
- Dividing by magnitudes cancels out length, leaving only the angle

Range: **1.0** = identical direction (same meaning), **0** = perpendicular (unrelated), **-1** = opposite meaning.

---

**Concrete example with real numbers:**

Real models use 384–1536 dimensions; 3 dimensions is enough to show the math:

```python
import numpy as np

supply_chain_chunk = np.array([0.9, 0.1, 0.8])   # "Tesla supply chain risk"
sourcing_query     = np.array([0.8, 0.2, 0.9])   # "problems sourcing parts"
dividend_chunk     = np.array([0.1, 0.9, 0.2])   # "quarterly dividend policy"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(sourcing_query, supply_chain_chunk))  # → 0.98
print(cosine_similarity(sourcing_query, dividend_chunk))      # → 0.27
```

**Why 0.98?** The supply chain chunk and the query have high values in the same positions (dims 0 and 2). They point the same direction — same topic.

**Why 0.27?** The dividend chunk has its high value in dim 1, while the query is strong in dims 0 and 2. Different directions — different topic.

---

**Why not Euclidean distance?**

```
Short doc: "supply chain risk"              → vector magnitude: 0.5
Long doc:  "supply chain risk ... [5 para]" → vector magnitude: 3.2

Euclidean distance:   LARGE  (wrongly says they're dissimilar)
Cosine similarity:    ~0.99  (correctly says same topic)
```

Two documents about the same thing — one just longer. Euclidean distance fails here. Cosine similarity handles it correctly, which is why it is the standard metric for semantic search.

---

**How the vector DB uses it:**

At query time the DB computes cosine similarity between the query vector and every stored chunk vector (via HNSW, not brute force), then returns the top-k highest scores:

```python
# Simplified — what the DB is doing internally:
scores = {chunk_id: cosine_similarity(query_vec, chunk_vec)
          for chunk_id, chunk_vec in index.items()}

top_k = sorted(scores, key=scores.get, reverse=True)[:5]
# → the 5 chunk IDs most semantically similar to the query
```

**One-line summary:**
> Cosine similarity measures the angle between two vectors, not their distance — so it compares meaning regardless of document length, and is the standard metric for finding semantically relevant chunks in a vector database.

---

### Q: What are bi-encoders and cross-encoders, how do they differ, and how does re-ranking use both to improve retrieval precision?

**The problem re-ranking solves:**

Your vector DB returns top-5 chunks. But the genuinely best chunk — the one that actually answers the question — might be sitting at position #7, just outside the window. You missed it.

The root cause: the bi-encoder that powers vector search is fast but imprecise. Re-ranking fixes this with a two-stage approach.

---

**Bi-encoder — fast but blind to the pair:**

A bi-encoder embeds the query and each chunk **separately**, then compares their vectors.

```
Query: "What are Tesla's biggest risks?"   →  embed  →  [0.19, -0.06, 0.89, ...]
                                                                   ↕  cosine similarity
Chunk: "Tesla faces supply chain risk..."  →  embed  →  [0.21, -0.08, 0.91, ...]
```

The key word is *separately*. The encoder never sees query and chunk together — it scores based on how similar their individual vectors are, which is a proxy for relevance, not a direct measurement of it. Subtle query-specific relevance gets lost.

**The upside:** chunks are pre-embedded at index time. A query only needs one encoding, then it's just distance calculations. Very fast.

---

**Cross-encoder — precise but slow:**

A cross-encoder takes the query and a chunk as a **single combined input** and outputs one relevance score.

```
Input:  ["What are Tesla's biggest risks?", "Tesla faces supply chain risk..."]
              ↓  full transformer attention across both texts
Output: 0.94   ← one relevance score for this exact pair
```

Because both texts go in together, the model attends across them and sees exactly how well *this chunk* answers *this question*. Much more accurate.

**The downside:** there is nothing to pre-compute. The score depends on the pair, so you must run it fresh for every (query, chunk) combination. Running it over 50,000 chunks per query is too slow.

---

**The insight — use both together:**

```
Stage 1 — RETRIEVE WIDE (bi-encoder, fast)
  query → vector search → top 25 candidates     ← high recall, some noise

Stage 2 — RE-RANK NARROW (cross-encoder, slow)
  (query, chunk_1)  → score: 0.94
  (query, chunk_2)  → score: 0.41
  ...
  (query, chunk_25) → score: 0.12
                                 → take top 4   ← high precision, fed to LLM
```

You pay the expensive cross-encoder on only 25 items, not 50,000. The chunk that ranked #7 in stage 1 can jump to #1 after re-ranking — now it is in the prompt.

---

**Concrete example:**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# Stage 1 — bi-encoder retrieves wide (fast)
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
coll = chromadb.Client().get_collection("filings")

question = "What are Tesla's biggest supply chain risks?"
q_vec = bi_encoder.encode([question])

candidates = coll.query(
    query_embeddings=[q_vec[0].tolist()],
    n_results=25,                           # cast wide net
)["documents"][0]

# Stage 2 — cross-encoder re-ranks narrow (accurate)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = cross_encoder.predict([(question, chunk) for chunk in candidates])

ranked = sorted(zip(scores, candidates), reverse=True)
top4 = [chunk for _, chunk in ranked[:4]]  # drop these into the prompt
```

Say the bi-encoder returned these as its top 5:

```
Rank 1 (0.91): "Tesla revenue grew 15% YoY..."           ← semantically close, wrong answer
Rank 2 (0.89): "Tesla faces semiconductor sourcing risk"  ← correct answer
Rank 3 (0.87): "Supply chain terms defined..."            ← generic, not Tesla-specific
Rank 4 (0.85): "Tesla's gross margin compressed..."       ← related but not about risks
Rank 5 (0.83): "Single-source supplier dependency..."     ← actually very relevant
```

After cross-encoder re-ranking:

```
Rank 1 (0.94): "Tesla faces semiconductor sourcing risk"  ← jumped from #2
Rank 2 (0.91): "Single-source supplier dependency..."     ← jumped from #5
Rank 3 (0.72): "Tesla's gross margin compressed..."
Rank 4 (0.41): "Tesla revenue grew 15% YoY..."            ← dropped to #4
```

The revenue chunk was vector-similar to the query (both are about Tesla), but the cross-encoder correctly scored it low because it does not answer "what are the risks."

**One-line summary:**
> A bi-encoder embeds query and chunk separately for fast approximate matching; a cross-encoder scores them together for precise relevance — re-ranking uses the bi-encoder to cast a wide net cheaply, then the cross-encoder to reorder that shortlist accurately.

---

### Q: What are dense and sparse retrieval, how does each fail, and how does hybrid search combine them to cover both failure modes?

**Dense search (semantic / embeddings):**

This is what the vector DB does. Every chunk is embedded into a vector that captures meaning, and you retrieve chunks by measuring vector similarity.

It works well for synonyms, paraphrase, and conceptually related text — even when zero words overlap. It fails on exact phrases, legal terms, product codes, and rare jargon. The phrase `"substantial doubt"` (an accounting red flag) gets semantically blurred into nearby finance words, so the chunk containing that exact phrase may not rank highest.

```
Query: "substantial doubt"
Chunk: "There is substantial doubt about the company's ability..."  → similarity: 0.61  ✗ missed
Chunk: "liquidity concerns remain elevated..."                       → similarity: 0.74  ✓ retrieved (wrong one)
```

---

**Sparse search (keyword / BM25):**

BM25 is the classic keyword ranking algorithm — no vectors involved. It scores a chunk by how often query terms appear in it, dampened by how common those terms are across the whole corpus (rare words count more).

It works well for exact phrases, legal jargon, product codes, and names. It fails entirely on synonyms and paraphrase — no word overlap means a score of zero, regardless of meaning.

```
Query: "parts sourcing problems"
Chunk: "supply chain disruption from single-source suppliers"  → BM25 score: 0  ✗ missed
```

---

**The symmetry — why hybrid works:**

Dense and sparse fail in exactly opposite directions:

```
                    DENSE (embeddings)      SPARSE (BM25)
Synonyms            ✓ great                 ✗ misses
Paraphrase          ✓ great                 ✗ misses
Exact phrases       ✗ misses                ✓ great
Codes / SKUs        ✗ misses                ✓ great
Legal jargon        ✗ misses                ✓ great
```

One retriever's blind spot is the other's strength. Running both and fusing the results covers the full space.

---

**Hybrid search — fusing with RRF:**

Run both retrievers independently, then combine their ranked lists. The fusion problem: dense scores are cosine similarities (0–1) and BM25 scores are term-frequency numbers (0–50+) — different scales, can't be added directly.

**Reciprocal Rank Fusion (RRF)** solves this by using position in each list, not raw scores:

```python
def rrf(dense_ids, sparse_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

Rank #1 in either list gets `1/(60+0) = 0.0167`. A chunk that ranks well in both lists accumulates scores from both — it rises to the top. No normalization needed.

---

**Concrete example — Tesla filing query:**

Query: `"substantial doubt supply chain"` — mixes a legal exact phrase + a semantic concept.

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

chunks = [
    "There is substantial doubt about Tesla's going-concern ability",   # id: 0
    "Tesla faces supply chain risk from single-source suppliers",        # id: 1
    "Revenue increased 15% year over year to $97.7 billion",            # id: 2
    "Procurement disruptions may delay production timelines",            # id: 3
]

# Dense retrieval result (vector similarity):
# → ["1", "3", "0", "2"]   chunk 0 ("substantial doubt") ranks #3 — near miss

# Sparse retrieval result (BM25 exact match):
# → ["0", "1", "3", "2"]   chunk 0 ranks #1 — exact phrase found

fused = rrf(dense_ids=["1","3","0","2"], sparse_ids=["0","1","3","2"])
# → ["1", "0", "3", "2"]
```

What happened:

```
Chunk 0 "substantial doubt..."    Dense: #3   Sparse: #1  → Fused: #2  ✓ rescued by BM25
Chunk 1 "supply chain risk..."    Dense: #1   Sparse: #2  → Fused: #1  ✓ strong in both
Chunk 3 "procurement disruption"  Dense: #2   Sparse: #3  → Fused: #3
Chunk 2 "revenue 15%..."          Dense: #4   Sparse: #4  → Fused: #4  ✗ correctly buried
```

Chunk 0 would have been missed by dense alone. BM25 caught the exact phrase `"substantial doubt"` and RRF rescued it into the final top 2.

**One-line summary:**
> Dense search matches meaning but misses exact strings; sparse search (BM25) matches exact strings but misses synonyms — hybrid search runs both and fuses their rankings with RRF so neither failure mode survives.

---

## Vector Databases

### Q: In Stage 3 (Vector Databases) of the RAG pipeline — what is it actually doing, and what is HNSW?

**The problem Stage 3 solves:**

After Stage 2 (Embeddings) you have thousands of chunk-vectors sitting in memory:

```
Chunk 1: "Tesla faces supply chain risk..."     → [0.21, -0.08, 0.91, ...]
Chunk 2: "Revenue increased 15% YoY..."         → [0.54,  0.33, 0.12, ...]
Chunk 3: "Substantial doubt about liquidity..." → [-0.3,  0.71, 0.44, ...]
...
Chunk 50,000: "..."                             → [...]
```

A user asks: *"What are Tesla's biggest risks?"*

You embed that question → `[0.19, -0.06, 0.89, ...]`

**The naive approach:** compare this question vector against all 50,000 chunk vectors one by one. That's 50,000 similarity calculations per query. Too slow for production.

A vector database solves this — it's an indexed structure that finds the nearest vectors without scanning everything.

---

**HNSW — the algorithm behind fast search:**

**HNSW** stands for Hierarchical Navigable Small World. Mental model: a multi-layer graph.

Imagine a huge library with 50,000 books. You want books similar to *"supply chain disruption."*

- **Brute force** = walk every single aisle and check every book. Correct but slow.
- **HNSW** = the library has floors, each with different granularity:

```
Floor 3 (sparse)   Only landmark books — 1 per genre
                   [Finance] ──── [Tech] ──── [Law]

Floor 2            Subcategory books
                   [Auto Risk] ── [Supply Chain] ── [Litigation]

Floor 1 (dense)    Every single book
                   book1 ─ book2 ─ book3 ─ ... ─ book50000
```

How search works:
1. Start at Floor 3 — big jumps, find the right genre (Finance)
2. Drop to Floor 2 — medium jumps, find the right subcategory (Supply Chain)
3. Drop to Floor 1 — small precise steps, find the 5 nearest books

You went from checking 50,000 items → checking maybe 50. That's the speedup.

The word **"Approximate"** in ANN (Approximate Nearest Neighbor) means it might miss the absolute #1 most similar chunk and return #2 instead. For RAG this tradeoff is fine — you're fetching the top 5, and a slightly wrong ordering doesn't matter.

---

**What a vector DB actually stores per entry — three things:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Entry for Chunk 7                                               │
│                                                                 │
│  vector:   [0.21, -0.08, 0.91, ...]   ← for similarity search  │
│  document: "Tesla faces supply chain  ← the actual text you'll  │
│             risk from single-source    drop into the prompt     │
│             semiconductor suppliers"                            │
│  metadata: { company: "TSLA",         ← for filtering before   │
│              year: 2025,               searching                │
│              section: "Risk Factors",                           │
│              chunk_id: 7 }                                      │
└─────────────────────────────────────────────────────────────────┘
```

The metadata is the part people underestimate. Without it, "Tesla's risks this year" searches across all companies and all years. With a metadata filter, you first narrow to TSLA 2025, then run the vector search inside that subset — faster and far more accurate.

---

**Concrete example: Tesla 10-K filing**

Setup (indexing, happens once):

```python
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
coll = client.create_collection("filings")

chunks = [
    "Tesla faces supply chain risk from single-source semiconductor suppliers",
    "Revenue increased 15% year over year to $97.7 billion",
    "Substantial doubt about going-concern liquidity if cash burns continue",
]

vectors = model.encode(chunks)

coll.add(
    documents=chunks,
    embeddings=[v.tolist() for v in vectors],
    metadatas=[
        {"company": "TSLA", "year": 2025, "section": "Risk Factors"},
        {"company": "TSLA", "year": 2025, "section": "Financials"},
        {"company": "TSLA", "year": 2025, "section": "Risk Factors"},
    ],
    ids=["tsla-2025-0", "tsla-2025-1", "tsla-2025-2"],
)
```

Query (every time the user asks something):

```python
question = "What are Tesla's biggest risks?"
q_vec = model.encode([question])   # same model — critical

results = coll.query(
    query_embeddings=[q_vec[0].tolist()],
    n_results=2,
    where={"company": "TSLA", "year": 2025},   # metadata filter first
)

print(results["documents"][0])
# → [
#     "Tesla faces supply chain risk from single-source semiconductor suppliers",
#     "Substantial doubt about going-concern liquidity if cash burns continue"
#   ]
# The revenue chunk is NOT returned — it was far away in meaning-space
```

The revenue chunk had a low cosine similarity to "biggest risks" so the DB didn't surface it. That's the semantic matching working correctly.

---

**Why metadata filter is the cheapest accuracy win:**

```
Without filter:  search all 50,000 vectors → top 5 results might include
                 Ford 2019, Apple 2022, random other filings

With filter:     first narrow to TSLA 2025 (maybe 800 vectors),
                 then find top 5 → all from the right filing
```

One line of code (`where={"company": "TSLA", "year": 2025}`). Massive accuracy improvement. This is what your notes mean by "metadata earns its keep."

---

**Which DB to pick (decision tree):**

```
Learning / local dev?              → Chroma    (zero setup, runs in-process)
Already using Postgres?            → pgvector  (vectors live next to your relational data)
Need hybrid search built in?       → Qdrant or Weaviate
Millions of vectors, production?   → Pinecone or Milvus
```

For the Module 2 Filing Analyst project, Chroma is the right call. It can be swapped later.

**One-line summary:**
> A vector database is a fast nearest-neighbor index (using HNSW) that trades a tiny bit of accuracy for a massive speedup — it stores the vector, the original text, and metadata together, and the metadata filter is what makes retrieval precise, not just fast.

---
