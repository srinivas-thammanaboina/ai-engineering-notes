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

### Q: What is temperature — is it the distance between vectors? And how is it calculated?

**No — temperature has nothing to do with vector distance.** These are two separate concepts that operate at completely different stages.

- **Vector distance** — measures *meaning similarity* between words in embedding space. Happens inside the transformer layers while building up context.
- **Temperature** — controls *how decisive the model is* when picking the next token from its probability list. Happens at the very end, during sampling.

Where each fits in the pipeline:

```
Step 1 — vectors & attention    →  model builds up meaning and context
Step 2 — probability scores     →  model scores every possible next token (logits)
Step 3 — temperature applied    →  reshapes those scores before sampling
Step 4 — pick one token         →  that's the output
```

Temperature only shows up at step 3. Everything about vectors is done by step 1.

---

**How temperature is actually calculated:**

The model produces raw scores called **logits** for every token in the vocabulary. Temperature is applied by dividing all logits by T before running softmax:

```
P(token) = exp(logit / T) / sum of exp(all logits / T)
```

**Softmax** is the function that converts raw scores into probabilities (all values between 0–1, all adding up to 1). Temperature just slips in as a divisor before softmax runs.

**Concrete example with real numbers:**

Raw logits the model produced:
```
Paris:  4.0
London: 2.0
city:   1.0
```

**T = 1.0 — default, no change:**
```
exp(4)=54.6   exp(2)=7.4   exp(1)=2.7   total=64.7

Paris  = 54.6 / 64.7 = 0.84
London =  7.4 / 64.7 = 0.11
city   =  2.7 / 64.7 = 0.04
```

**T = 0.5 — low temperature (divide logits by 0.5 → they double):**
```
logits become: 8.0, 4.0, 2.0

exp(8)=2981   exp(4)=54.6   exp(2)=7.4   total=3043

Paris  = 2981 / 3043 = 0.98   ← almost certain
London =   55 / 3043 = 0.02
city   =    7 / 3043 = 0.00
```

**T = 2.0 — high temperature (divide logits by 2 → they shrink):**
```
logits become: 2.0, 1.0, 0.5

exp(2)=7.4   exp(1)=2.7   exp(0.5)=1.65   total=11.75

Paris  = 7.4  / 11.75 = 0.63
London = 2.7  / 11.75 = 0.23   ← gets a real chance
city   = 1.65 / 11.75 = 0.14   ← also gets a real chance
```

**Why dividing by T has this effect:**

- **T < 1** → dividing by a small number makes logits *bigger* → gaps between scores get amplified → winner pulls further ahead → distribution gets more peaked
- **T > 1** → dividing by a large number *compresses* logits → gaps between scores shrink → distribution flattens out
- **T = 1** → logits unchanged → standard softmax

```
T = 0.5  →  [0.98, 0.02, 0.00]   ← sharp, decisive
T = 1.0  →  [0.84, 0.11, 0.04]   ← normal
T = 2.0  →  [0.63, 0.23, 0.14]   ← flat, creative
```

Temperature controls how much the model *cares about the differences* between its scores. Low T amplifies differences — the best guess dominates. High T smooths them away — underdogs get a real chance.

**Practical guide:**

| Use case | Temperature |
|---|---|
| Data extraction, structured output | 0–0.2 |
| Factual Q&A, code generation | 0.2–0.5 |
| Conversational assistant | 0.7 |
| Creative writing, brainstorming | 0.9–1.2 |
| Experimental / maximum variety | 1.5+ |

**One-line summary:**
> Temperature divides the model's raw scores before softmax — low values amplify score differences (decisive), high values compress them (creative). It has nothing to do with vector distance.

---

### Q: What is softmax and where is it used?

Softmax takes a list of raw numbers (some big, some small, even negative) and squashes them into a list of probabilities that are all positive and add up to 1 — while keeping the *order* (the biggest stays biggest).

**The intuition first:**

Imagine a model looks at a photo and produces three raw "confidence scores":

```
cat:  2.0
dog:  1.0
bird: 0.1
```

These are just *scores* — they could be any number. They don't add to 1, so they're not yet probabilities. We can't say "70% cat" from them directly.

Softmax converts them into something you *can* read as percentages:

```
cat:  65.9%
dog:  24.2%
bird:  9.9%   → adds up to 100%
```

Think of it as a "vote-share calculator": each candidate has a raw popularity score, and softmax tells you what fraction of the total vote each one actually gets.

**Why not just divide each score by the sum?**

That's the obvious idea — but the problem is raw scores can be **negative**, and a simple divide breaks with negatives (you could get negative "probabilities").

Softmax fixes this by first running every score through `e^x` (exponentiation), which is **always positive**, and *then* dividing by the total. The formula:

```
softmax(x_i) = exp(x_i) / sum of exp(all x_j)
```

In plain words: *"raise e to the power of each score, then give each one its share of the total."*

The `e^x` step has a useful side effect: it **exaggerates differences**. A score that's a bit higher than the rest gets a *much* bigger probability. That's why it's called "soft**max**" — it leans toward the maximum, like a gentle version of just picking the single winner.

**Worked example (the actual arithmetic):**

Scores: `[2.0, 1.0, 0.1]`

Step 1 — exponentiate each:
```
e^2.0  = 7.389
e^1.0  = 2.718
e^0.1  = 1.105
```

Step 2 — add them up:
```
total = 7.389 + 2.718 + 1.105 = 11.212
```

Step 3 — each divided by the total:
```
cat:  7.389 / 11.212 = 0.659  (65.9%)
dog:  2.718 / 11.212 = 0.242  (24.2%)
bird: 1.105 / 11.212 = 0.099  ( 9.9%)
```

Done — positive, ordered, sums to 1.

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e / e.sum()

softmax(np.array([2.0, 1.0, 0.1]))
# array([0.659, 0.242, 0.099])
```

That `- np.max(x)` trick doesn't change the result — it just stops `e^x` from overflowing on big numbers. Worth knowing because real code always does it.

**Where it's used (and why it matters for AI engineering):**

1. **The last layer of a classifier.** Any model that picks one label out of many (image classes, sentiment, spam/not-spam) ends with softmax to turn scores into a probability per class.

2. **Inside every LLM — the most relevant one here.** When an LLM predicts the next token, it produces one raw score (a "logit") for *every* word in its vocabulary (~50,000+ of them). Softmax converts those logits into a probability distribution over the whole vocabulary, and the model samples its next word from that distribution.

   This is exactly where **temperature** lives (see the temperature note above). Before softmax, the logits are divided by a temperature `T`:
   - **Low T (e.g. 0.2)** → differences get exaggerated → distribution gets "spiky" → model almost always picks the top word → predictable, repetitive.
   - **High T (e.g. 1.5)** → differences flatten out → more words get a real chance → creative, riskier output.

   So when you tune `temperature` in an API call, you're literally reshaping the softmax distribution.

3. **Attention (the heart of transformers).** Inside attention, each token computes scores for how much it should "pay attention" to every other token. Softmax turns those into weights that sum to 1, so attention becomes a weighted average — "spend 100% of your attention, distributed across these tokens."

**One-line summary:**
> Softmax is the bridge from "raw scores the model computed" to "a probability distribution you can sample from or read as percentages." Every time an LLM picks a word, attention focuses, or a classifier decides — softmax is doing that conversion.

---
