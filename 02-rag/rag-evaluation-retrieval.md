# RAG Evaluation — Part 1: Retrieval & Ranking

> Scope: how to measure whether the **retriever** found the right documents and ranked them well. This is the first of two failure modes in a RAG system (the other being generation, covered in Part 2).

---

## 0. Why evaluate retrieval separately

**Overview (simple terms):** A RAG system has two stages — retrieve relevant chunks, then generate an answer from them. If the answer is bad, the cause is either "we never gave the model the right context" (retrieval failure) or "we had the right context but the model misused it" (generation failure). Evaluating retrieval on its own lets you tell these apart.

**How it's used:** You build a *golden dataset* — a set of queries paired with the document chunks that are known to be relevant (ground truth). You run your retriever over each query and compare its top-k results against the ground truth using the metrics below.

**Practical example:** Golden set has the query *"What is the refund window?"* with ground-truth relevant chunk = `policy_doc#refunds`. Your retriever returns 5 chunks; you check whether (and where) `policy_doc#refunds` shows up.

**Key consideration:** Your metrics are only as good as your golden dataset. A small, biased, or stale labeled set makes every number below misleading. Budget real effort into labeling, and refresh it as your corpus changes.

---

## 1. Recall@k (Hit Rate)

**Overview:** The fraction of queries where at least one relevant document lands in the top-k results. Answers: "did we find it at all, within k?"

**How it's used:** Pick k to match how many chunks you actually feed the LLM (e.g. k=5 if your prompt uses the top 5). A low Recall@k means relevant content isn't even reaching the generator — fix retrieval (embeddings, chunking, query rewriting) before touching prompts.

**Practical example:** 3 queries, k=3. Q1 has a relevant doc at rank 2 (hit), Q2 has none in top-3 (miss), Q3 has one at rank 1 (hit). Recall@3 = 2/3 ≈ 0.67.

**Key consideration:** It's binary per query — a relevant doc at rank 1 and at rank 5 both count equally. It tells you *whether* you found something, not *where*. Pair it with a position-sensitive metric (MRR/NDCG).

### 1a. Fraction Recall@k (strict IR definition)

**Overview:** The textbook information-retrieval definition of recall. Instead of "did at least one relevant doc show up" (hit rate), it asks *what fraction of all relevant docs you actually retrieved* in the top-k.

$$\text{Recall@k} = \frac{|\text{retrieved}_k \cap \text{relevant}|}{|\text{relevant}|}$$

So the two senses answer different questions: hit rate is binary per query (1 if any relevant doc is in top-k); fraction recall is continuous and credits you for *how many* of the relevant set you caught.

**How it's used:** Use it when multiple documents are relevant and you need most or all of them — multi-hop questions, research aggregation, compliance/legal review where missing any relevant doc is costly. Use plain hit rate instead when a single good chunk answers the question (most simple Q&A RAG), so finding one is enough.

**Practical example:** A query has 4 relevant docs in the corpus; your top-5 retrieval contains 2 of them.
- Fraction recall@5 = 2/4 = **0.5** (you got half the relevant set)
- Hit rate@5 = **1.0** (at least one showed up)

Same retrieval, very different numbers — because they measure different things.

**Key consideration — why most RAG evals avoid it:** Fraction recall requires the *complete* set of relevant docs labeled for every query. In a large corpus that means a human exhaustively reviewing many candidates per query to mark every relevant chunk — heavy, expensive manual annotation. If any relevant doc is left unlabeled, the denominator is wrong and recall is overstated. Hit rate is far cheaper: you only need *one* known-relevant doc per query, so labeling is "is this answer chunk relevant: yes/no" rather than "find everything relevant in the whole corpus." That cost gap is why RAG evals usually default to hit rate, and why this file uses the hit-rate sense by default.

---

## 2. Precision@k

**Overview:** Of the top-k retrieved docs, what fraction are actually relevant. Measures signal-to-noise.

**How it's used:** High recall but low precision means you're stuffing the context window with junk, which raises cost and can distract the generator. Use it to tune k and to compare retrievers that all "find" the answer but differ in how much noise they drag along.

**Practical example:** Top-3 results contain 1 relevant + 2 irrelevant chunks → Precision@3 = 1/3 ≈ 0.33.

**Key consideration:** Precision and recall trade off as you change k. Larger k usually raises recall and lowers precision. Report them together, not in isolation.

---

## 3. MRR (Mean Reciprocal Rank)

**Overview:** The average of 1/(rank of the *first* relevant doc) across queries. Rewards putting a relevant doc near the top.

**How it's used:** Best when you mainly care about the single best hit being high — e.g. a Q&A bot where one good chunk answers the question. A relevant doc at rank 1 scores 1.0, rank 2 scores 0.5, rank 5 scores 0.2.

**Practical example:** Q1 first relevant at rank 2 → 0.5; Q2 none → 0; Q3 at rank 1 → 1.0. MRR = (0.5 + 0 + 1.0)/3 = 0.5.

**Key consideration:** It only looks at the *first* relevant doc and ignores the rest. If multiple relevant chunks matter, MRR understates quality — use MAP or NDCG instead.

---

## 4. MAP (Mean Average Precision)

**Overview:** For each query, compute precision at every rank where a relevant doc appears, average those, then average across queries. Rewards finding *many* relevant docs and ranking them all high.

**How it's used:** Good for "research-style" retrieval where several chunks together form the full answer and you care about all of them surfacing early.

**Practical example:** A query with relevant docs at ranks 1 and 3. Precision at rank 1 = 1/1 = 1.0; precision at rank 3 = 2/3 ≈ 0.67. Average Precision = (1.0 + 0.67)/2 ≈ 0.83. MAP averages this AP over all queries.

**Key consideration:** Assumes binary relevance (relevant or not). If your labels have grades of relevance, NDCG is the better fit.

---

## 5. NDCG@k (Normalized Discounted Cumulative Gain)

**Overview:** The go-to metric when relevance is *graded* (e.g. perfect / partial / irrelevant) rather than yes/no. It sums relevance gains, discounts them by position (lower ranks count less), and normalizes against the ideal ordering to give a score in [0, 1].

**How it's used:** Use when some chunks are clearly more useful than others and order matters. Becomes the headline retrieval metric in mature eval setups because it captures both relevance grade and position in one number.

**Practical example:** Relevance grades [3, 2, 0] at ranks 1–3. DCG = 3/log2(2) + 2/log2(3) + 0 = 3 + 1.26 = 4.26. Ideal ordering [3,2,0] gives the same IDCG here = 4.26, so NDCG = 4.26/4.26 = 1.0. Reorder to [2,3,0] and NDCG drops below 1.0.

**Key consideration:** Requires graded relevance labels, which are more expensive to produce than binary ones. The log discount is a convention — be consistent so scores are comparable across experiments.

---

## 6. Binary vs. graded relevance

**Overview:** Binary = a doc is either relevant or not. Graded = relevance on a scale (e.g. 0–3). The choice dictates which metrics you can use.

**How it's used:** Binary labels support Recall@k, Precision@k, MRR, MAP. Graded labels are required for NDCG and give a richer picture. Pick based on how much your use case distinguishes "perfect" from "partially helpful" chunks.

**Practical example:** For "What's the capital of France?", one chunk is exactly right (grade 3) and another mentions France generally (grade 1) — graded labels capture that difference; binary would call both simply "relevant."

**Key consideration:** Graded labeling needs a clear, written rubric or annotators won't agree. Start binary if you're early; move to graded once retrieval is "good enough" and you need finer signal.

---

## Quick reference

| Metric | Question it answers | Relevance type | Position-aware |
|---|---|---|---|
| Recall@k / Hit Rate | Found at least one, within k? | Binary | No |
| Fraction Recall@k (IR) | What share of *all* relevant docs found? | Binary set | No |
| Precision@k | How much of top-k is signal? | Binary | No |
| MRR | How high is the *first* hit? | Binary | Yes |
| MAP | Are *all* relevant docs ranked high? | Binary | Yes |
| NDCG@k | Best graded ranking, position-weighted? | Graded | Yes |

**Workflow tip:** Start with Recall@k to confirm content reaches the generator at all, then use MRR/NDCG to improve ordering, and Precision@k to control noise/cost.
