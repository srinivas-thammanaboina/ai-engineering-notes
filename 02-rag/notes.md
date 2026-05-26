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
