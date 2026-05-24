# Module 01 — Foundations: Personal Notes & Q&A

All questions, clarifications, and learnings from working through this module go here.

---

## How LLMs Work

### Q: What is a vector? How is it different from a list or a map?

A vector is a fixed-length list of numbers where each position has a fixed meaning, and you can do math on them to measure similarity.

**Start with a plain list:**

```python
list = [10, 3, 99, 7]
```

A plain list is just values in order. The positions don't mean anything in particular, and you can't meaningfully compare two lists.

**A vector is a list with a coordinate system:**

Each position represents a dimension of meaning. The model learns what each dimension encodes during training — you don't define them manually.

```
"Paris"  →  [0.9,  0.1,  0.85, 0.2,  ...]  # 768 numbers
"London" →  [0.88, 0.12, 0.82, 0.18, ...]  # similar values → close in space
"banana" →  [0.1,  0.7,  0.05, 0.9,  ...]  # very different values → far apart
```

**Why not just use a dict/map?**

| | Map / dict | Vector |
|---|---|---|
| Keys | Named, flexible | Positional (index 0, 1, 2...) |
| Size | Variable | Fixed (e.g. always 768 dims) |
| Math | Hard to compare two maps | Built for math — cosine similarity, dot product |
| Meaning of keys | You define it | Model learns it from data |

The critical difference: **you can calculate distance between two vectors.** That's what "similar meanings sit close together" actually means — closeness is a calculable number, not a judgment call.

**The geometry trick:**

Vectors can be added and subtracted like arrows in space. The classic example:

```
king - man + woman ≈ queen
```

This works because the relationship "royalty + gender shift" is encoded as a direction in vector space. With 768+ dimensions, the model encodes thousands of such relationships simultaneously — grammar, meaning, facts, tone — all in the same vector.

**One-line summary:**
> A vector is a fixed-length list of numbers where position encodes meaning, distance measures similarity, and arithmetic encodes relationships — making it the right data structure for representing language mathematically.

This is the foundation of the entire RAG module — you retrieve documents by finding vectors closest to your query vector.

---

### Q: What does "coreference" mean in the multi-head attention section?

Coreference means two different words in a sentence pointing to the same real-world thing.

The file already has the perfect example:

> *"The animal didn't cross the street because **it** was tired"*

Here `it` and `the animal` refer to the same thing. Figuring out which earlier word a pronoun points to is called **coreference resolution**.

More everyday examples:
- *"Sarah finished her coffee"* → `her` = `Sarah`
- *"The team won. They celebrated."* → `They` = `The team`
- *"I put the cake on the table and then ate it"* → `it` = `the cake`, not `the table`

**Why it matters for multi-head attention:**

Each attention head specializes in a different type of relationship. The file lists three examples:

- **Head 1 (grammar)** — subject, verb, object structure
- **Head 2 (coreference)** — which pronoun points back to which noun
- **Head 3 (topic)** — what the sentence is broadly about

These aren't manually assigned — the model discovers them during training. Coreference is common enough in language that a head tends to specialize in it naturally.

---

### Q: How many attention heads do typical modern LLMs run on average?

This is a two-part answer.

**Heads per layer — typical numbers:**

| Model | Size | Heads per layer | Layers |
|---|---|---|---|
| LLaMA 3 | 8B | 32 | 32 |
| LLaMA 3 | 70B | 64 | 80 |
| Mistral | 7B | 32 | 32 |
| GPT-3 | 175B | 96 | 96 |
| GPT-2 small | — | 12 | 12 |

Rough average for a modern mid-size model: **32 heads per layer.**

But all those heads run at every layer. LLaMA 3 8B has 32 heads × 32 layers = **1,024 attention operations per forward pass.** That's the real scale.

**The modern twist — GQA (Grouped Query Attention):**

Most 2023+ models don't run full multi-head attention anymore. They use GQA, which splits heads into two roles:

- **Query heads** — full 32, one per "question being asked"
- **KV heads (Key + Value)** — reduced to just 8, shared across query groups

```
Full MHA:  32 Q heads,  32 K heads,  32 V heads  ← expensive
GQA:       32 Q heads,   8 K heads,   8 V heads  ← ~4x cheaper memory
```

The K and V matrices eat GPU memory fast. Sharing them across groups keeps quality nearly identical while making inference much faster and cheaper.

**One-line summary:**
> Typical modern LLMs run 32–64 attention heads per layer, across 32–80 layers. Most now use GQA to reduce memory by sharing Key/Value heads across groups of Query heads — expressiveness without the full memory cost.

GQA comes up again naturally in the production and finetuning modules when thinking about inference costs.

---

### Q: What are Query, Key, and Value — and does each attention head have its own?

**The database lookup analogy:**

Think of attention as a fuzzy search against a database:

- **Query** — what you're searching *for*
- **Key** — the label/tag on each item in the database
- **Value** — the actual content you get back when there's a match

```
You search Google for "best pizza near me"   ← Query
Each webpage has tags: "food", "location"    ← Keys
The actual webpage content you read          ← Value
```

The match isn't exact — it's a similarity score. The closer your query is to a key, the more of that value gets pulled back.

**Applied to the "it" example:**

Sentence: *"The animal didn't cross the street because it was tired"*

When the model processes `it`:

```
Query from "it":   "I'm a pronoun — find what noun I refer to"
                        ↓ compare against every token's Key
Key of "animal":   "I'm a subject noun, an entity"     → HIGH match
Key of "street":   "I'm a location noun"               → LOW match
Key of "tired":    "I'm an adjective"                  → LOW match
                        ↓ weighted by match scores
Value of "animal": [the full vector meaning of animal]  → pulled in strongly
Value of "street": [vector meaning of street]           → almost ignored
```

Result: `it` gets its vector updated with information from `animal`. Now the model knows what `it` refers to.

**Yes — each head has its own separate Q, K, V:**

Each head learns its own weight matrices for Q, K, and V. So the same token `it` produces a different query in each head, because each head has learned to ask a different question:

```
Coreference head → Query from "it": "which noun do I refer to?"
Grammar head     → Query from "it": "what verb am I the subject of?"
Topic head       → Query from "it": "what is this sentence broadly about?"
```

Same token, three different queries, three different attention patterns, three different outputs — all running in parallel:

```
Token "it"
    ├── Head 1 (coreference) → looks at "animal"         ← Q/K/V set #1
    ├── Head 2 (grammar)     → looks at "was tired"      ← Q/K/V set #2
    └── Head 3 (topic)       → looks at "cross", "tired" ← Q/K/V set #3
```

All three outputs get combined into one enriched vector for `it` that carries all of that context.

**Keys vs Values — why they're separate:**

- **Key** = every token broadcasting "here's what kind of thing I am" — used to compute match scores against the query
- **Value** = the actual information you pull from a token if it matches — what gets blended into the output

Keys decide *how much* to attend to a token. Values decide *what information* you get from it. They're separate so a token can be easy to match against (strong key signal) while carrying different information in its value.

---
