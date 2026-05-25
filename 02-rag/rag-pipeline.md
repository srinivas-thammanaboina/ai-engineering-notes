# The RAG Pipeline

> The mechanism that finds the right knowledge to fill the context window. Context engineering (previous file) decided *what* belongs; RAG decides *how to find it*.

## The one-sentence truth

Retrieval-Augmented Generation = chop your documents into chunks, turn each chunk into a vector that captures its *meaning*, store the vectors, and at query time fetch only the handful of chunks most relevant to the question — then hand those to the model.

## The core loop

RAG has two phases. **Indexing** happens once, offline, when you ingest documents. **Querying** happens every time a user asks something.

```
INDEXING (offline, once per document)
  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
  │ Document │──▶│  Chunk   │──▶│  Embed    │──▶│ Store vectors │
  │  (10-K)  │   │ (split)  │   │ (→ vector)│   │ (vector DB)   │
  └──────────┘   └──────────┘   └───────────┘   └──────────────┘

QUERYING (every request)
  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
  │  User    │──▶│  Embed   │──▶│ Retrieve  │──▶│ Augment      │──▶ Generate
  │ question │   │ question │   │ top-k     │   │ (chunks into │    (LLM
  └──────────┘   └──────────┘   │ similar   │   │  the prompt) │     answers)
                                │ chunks    │   └──────────────┘
                                └───────────┘
```

The trick that makes it work: the question and the chunks live in the **same vector space**, so "find chunks relevant to this question" becomes "find chunks whose vectors are *near* the question's vector." Geometry does the matching.

Let's walk each stage.

---

## Stage 1: Chunking

You can't embed a whole 10-K as one vector — you'd lose all granularity (one vector can't represent 80,000 tokens of distinct topics). So you split it into chunks. **How** you split is the single most underrated decision in RAG; bad chunking quietly caps the quality of everything downstream.

### The strategies, worst to best for our use case

**Fixed-size** — split every N characters/tokens. Simple, but brutal: it slices mid-sentence, mid-table, mid-thought.
```
"...the company faces substantial competi" | "tion in all markets, particularly..."
                                           ↑ split lands mid-word, meaning destroyed
```

**Recursive** — split on a hierarchy of separators (paragraphs → sentences → words), only going finer when a chunk is still too big. This respects natural boundaries and is the sane default. LangChain's `RecursiveCharacterTextSplitter` is the workhorse here.

**Semantic** — embed sentences, detect where the topic *shifts* (a similarity drop between adjacent sentences), and break there. Best quality, more expensive to compute. Worth it when chunk boundaries really matter.

### Why filings break naive chunkers

10-Ks are not clean prose. They have section headers (Item 1A. Risk Factors), tables of financial data, footnotes, and legal boilerplate. A fixed-size splitter will cut a risk factor in half and split a financial table across two chunks — so retrieval finds a fragment that's missing the number or the context. For filings you want **structure-aware** chunking: split on the document's own section markers first, *then* recursively within sections.

### The two dials: size and overlap

- **Chunk size** — too big wastes context budget and dilutes the embedding (many topics → one mushy vector); too small loses context (a chunk says "this increased 40%" with no subject). For dense documents like filings, ~500–1000 tokens is a common starting range.
- **Overlap** — chunks share a little text at their edges so a sentence split across a boundary still appears whole in one chunk. ~10–15% overlap is typical.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # target chunk size (chars here; can use tokens)
    chunk_overlap=100,     # ~12% overlap so boundary sentences survive
    separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph, then line, then sentence...
)
chunks = splitter.split_text(filing_text)
# each chunk is now a coherent passage you can embed and retrieve
```

---

## Stage 2: Embeddings

An **embedding** is a list of numbers (a vector) that represents the *meaning* of a piece of text. Text with similar meaning lands at nearby points in the vector space, regardless of exact words.

```
"supply chain disruption"   →  [0.21, -0.08, 0.91, ... ]  ┐
"problems sourcing parts"   →  [0.19, -0.05, 0.88, ... ]  ┘ near each other
"quarterly dividend policy" →  [-0.7,  0.42, 0.03, ... ]    far away
```

This is *why* RAG can match a question to a chunk that shares **zero keywords** with it — "problems sourcing parts" retrieves a chunk about "supply chain disruption" because they're close in meaning-space. (This is also RAG's weakness, which is why we add keyword search back in the advanced file.)

### How "nearby" is measured: cosine similarity

The standard metric is **cosine similarity** — the cosine of the angle between two vectors. It ignores magnitude and measures *direction* (semantic alignment). Range: 1.0 = identical meaning, 0 = unrelated, -1 = opposite.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1.0 → same direction → same meaning; ~0 → unrelated
```

Conceptually, every vector points somewhere on a high-dimensional sphere; similarity is "how close do they point?"

```
        chunk about supply chain
              ●
             /  small angle = high similarity
            /
   origin ●─────────● question "parts sourcing problems"
            \
             \  large angle = low similarity
              ●
        chunk about dividends
```

### Choosing an embedding model

The thing you must respect: **you have to embed your chunks and your queries with the *same* model.** Vectors from different models aren't comparable. Beyond that, the tradeoffs are dimensions (more = richer but more storage/compute), max input length (must fit your chunk size), and whether it's an API (OpenAI `text-embedding-3-small`, Voyage) or local/open (the `sentence-transformers` family, e.g. `all-MiniLM-L6-v2`). For learning and for keeping the Filing Analyst cheap, a small local model is perfectly fine.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, runs locally
vectors = model.encode(chunks)   # one vector per chunk; same model for queries later
```

---

## Stage 3: Vector databases

Once you have thousands of chunk-vectors, you need to find the nearest ones to a query vector *fast*. Comparing the query against every stored vector (brute-force / exact) works at small scale but doesn't — at millions of vectors you can't afford a full scan per query.

### ANN and HNSW

Vector DBs use **Approximate Nearest Neighbor (ANN)** search: trade a tiny bit of accuracy for a massive speedup. The dominant algorithm is **HNSW** (Hierarchical Navigable Small World). Mental model: a multi-layer graph of vectors where the top layers are sparse "express lanes" for big jumps across the space and lower layers are dense for fine-grained local search. A query enters at the top, greedily hops toward its neighborhood, and descends — reaching the right region in log-ish time instead of scanning everything.

```
Layer 2 (sparse)   ●──────────────●──────────────●     ← long hops, coarse
                   │              │
Layer 1            ●────●────●────●────●────●           ← medium hops
                        │         │
Layer 0 (dense)    ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●         ← every vector, fine search
                         ▲
                    query lands here, refines locally
```

### The options (you don't need to memorize these, just the shape)

- **Chroma** — dead simple, runs in-process, great for learning and local dev. Good first choice for the Filing Analyst.
- **FAISS** — a library (not a server) from Meta, extremely fast, you manage storage yourself.
- **pgvector** — vector search *inside Postgres*. Big advantage if you already have Postgres: your vectors live next to your relational data and metadata.
- **Pinecone / Weaviate / Qdrant / Milvus** — managed or production-grade servers with filtering, scaling, hybrid search built in. Reach for these when you outgrow local.

A vector DB stores three things per entry: the **vector**, the original **text** (so you can put it in the prompt), and **metadata** (company, filing year, section, source URL — crucial for filtering and citations).

```python
import chromadb

client = chromadb.Client()
coll = client.create_collection("filings")

coll.add(
    documents=chunks,                       # the text, for the prompt later
    embeddings=[v.tolist() for v in vectors],
    metadatas=[{"company": "TSLA", "year": 2025, "section": "Risk Factors"}] * len(chunks),
    ids=[f"tsla-2025-{i}" for i in range(len(chunks))],
)
```

---

## Stage 4: Retrieval

At query time: embed the question (same model!), ask the DB for the **top-k** nearest chunks, and hand them to the model.

```python
q_vec = model.encode(["What are Tesla's biggest stated risks this year?"])

results = coll.query(
    query_embeddings=[q_vec[0].tolist()],
    n_results=4,                                  # top-k
    where={"company": "TSLA", "year": 2025},      # metadata filter
)
# results["documents"] → the chunks you drop into the prompt
```

### The three retrieval levers

- **top-k** — how many chunks to fetch. Too few risks missing the answer; too many burns context budget and reintroduces lost-in-the-middle. Start around k=3–5 and tune.
- **similarity metric** — cosine is the default; match whatever your embeddings were trained for.
- **metadata filtering** — this is huge for filings. "Tesla's risks *this year*" should only search TSLA's 2025 filing — filtering by `company` and `year` *before* the vector search makes retrieval both faster and far more accurate. This is where structured metadata earns its keep.

### Closing the loop: augment + generate

Retrieval hands you the chunks. **Augment** = drop them into the context-engineered prompt (strongest chunk nearest the question, per the previous file). **Generate** = the model answers using them, and because each chunk carries its `section` metadata, every claim can cite its source — which is the whole point of the Filing Analyst.

---

## Where naive RAG falls short (the cliffhanger)

This pipeline works, but it has real failure modes:
- Pure semantic search **misses exact terms** — a query for the literal phrase "substantial doubt" (the going-concern red flag) might not rank the chunk containing it highest, because embeddings match meaning, not exact strings.
- The **top-k cut is blunt** — the truly best chunk might rank #7, just outside your k=5 window.
- **Vague or messy questions** retrieve vague, messy chunks.

Fixing these is the **Advanced RAG** file: re-ranking, hybrid search, and query rewriting.

## Mental model to carry forward

- RAG = **index once** (chunk → embed → store), **query every time** (embed → retrieve → augment → generate).
- **Chunking quality caps everything** — be structure-aware for filings.
- Embeddings put question and chunks in **one meaning-space**; cosine similarity finds neighbors.
- Vector DBs use **HNSW/ANN** for fast approximate nearest-neighbor search.
- **Metadata filtering** (company, year, section) is the cheapest accuracy win you have.
