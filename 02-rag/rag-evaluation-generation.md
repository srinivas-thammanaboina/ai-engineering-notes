# RAG Evaluation — Part 2: Generation, Methods & Production

> Scope: measuring whether the **generator** used the retrieved context well, the methods you use to score things, and how evaluation runs in production. This is the second failure mode (Part 1 covers retrieval).

---

## A. Generation Metrics

These evaluate the produced answer, given the retrieved context — not the retriever.

### A1. Faithfulness / Groundedness

**Overview:** Does every claim in the answer follow from the retrieved context, with nothing made up? This is the hallucination check.

**How it's used:** Break the answer into individual claims and verify each against the context, usually with an LLM-as-a-judge. Score = supported claims / total claims. Low faithfulness with good retrieval points to a generation problem (prompt, model, temperature).

**Practical example:** Context says "refunds within 30 days." Answer says "refunds within 30 days, and free shipping on returns." The shipping claim isn't in the context → unfaithful, even though part of the answer is correct.

**Key consideration:** An answer can be faithful but useless (faithfully restating irrelevant context). Always pair faithfulness with answer relevance.

### A2. Answer Relevance

**Overview:** Does the answer actually address the user's question — regardless of whether it's grounded?

**How it's used:** Scored by an LLM-as-a-judge comparing the answer to the original query. Catches off-topic, partial, or evasive answers.

**Practical example:** Query: "What's the refund window?" Answer: "We have a customer-friendly returns culture." On-topic-ish but doesn't answer → low relevance.

**Key consideration:** Relevance and faithfulness are orthogonal. You want both high; tracking only one hides half the failure space.

### A3. Context Precision

**Overview:** Of the retrieved chunks, how many were actually needed/used for the answer. A signal-to-noise measure for what the generator received.

**How it's used:** Diagnoses whether your retriever sends too much filler. High noise inflates cost and can degrade answers.

**Practical example:** 5 chunks retrieved, only 2 contributed to the answer → context precision ≈ 0.4.

**Key consideration:** Overlaps conceptually with retrieval Precision@k but is judged relative to what the *answer* needed, not just topical relevance.

### A4. Context Recall

**Overview:** Did retrieval pull *all* the context needed to fully answer the question?

**How it's used:** Requires a ground-truth answer; you check whether each fact in it is supported by the retrieved context. Missing facts = retrieval gap.

**Practical example:** Ground-truth answer needs both "30-day window" and "original packaging required." Retrieved context only has the window → context recall = 1/2 = 0.5.

**Key consideration:** This is really a *retrieval* diagnostic surfaced via the answer — low context recall means fix retrieval, not the prompt.

### A5. Hallucination detection

**Overview:** Identifying answer content not supported by context (the failure that faithfulness quantifies). Worth tracking as its own flagged category for debugging.

**How it's used:** Flag unsupported claims, log them, and feed examples into failure analysis. Often the highest-stakes issue in production RAG.

**Practical example:** Answer cites a "60-day window" found nowhere in the retrieved docs → logged as a hallucination instance.

**Key consideration:** Don't conflate "wrong" with "unfaithful." An answer can be unfaithful but coincidentally correct, or faithful to wrong context. Define clearly what you're flagging.

---

## B. Evaluation Methods

### B1. LLM-as-a-judge

**Overview:** Using a strong LLM to score outputs (faithfulness, relevance, etc.) against a rubric, instead of human raters.

**How it's used:** Cheap, fast, scalable scoring for large eval sets. You give the judge the query, context, answer, and a scoring instruction.

**Practical example:** Prompt: "Given this context and answer, rate faithfulness 1–5 and list any unsupported claims." Aggregate the scores across your dataset.

**Key consideration:** Judges have biases (length, position, self-preference) and aren't perfectly reliable. Calibrate against a sample of human labels and pin the judge model/version so scores stay comparable.

### B2. Human evaluation / annotation

**Overview:** People scoring outputs or labeling ground truth.

**How it's used:** Gold standard for building the golden dataset and for calibrating LLM judges. Reserve for high-value or ambiguous cases.

**Practical example:** Two annotators independently label query–chunk relevance; you measure agreement and reconcile disagreements.

**Key consideration:** Expensive and slow; needs a clear rubric to keep annotators consistent.

### B3. Reference-based vs. reference-free

**Overview:** Reference-based metrics compare against a ground-truth answer; reference-free ones judge the output on its own (e.g. faithfulness against context).

**How it's used:** Use reference-free where ground truth is hard to write (open-ended answers); reference-based where you have authoritative answers (context recall, exact-match QA).

**Practical example:** Faithfulness = reference-free (only needs context + answer). Context recall = reference-based (needs the ground-truth answer).

**Key consideration:** Reference-free is easier to scale but can miss correctness; reference-based is stricter but bottlenecked by labeling.

### B4. Frameworks

**Overview:** Tooling that packages these metrics. RAGAS is the common core for the four generation metrics above (faithfulness, answer relevance, context precision, context recall). Others: TruLens, DeepEval, Arize Phoenix.

**How it's used:** Wire your pipeline outputs into the framework, point it at your dataset, get metric dashboards without reimplementing scorers.

**Practical example:** Feed RAGAS your query/context/answer/ground-truth columns; it returns per-metric scores per query plus aggregates.

**Key consideration:** Frameworks are opinionated about definitions — read how each computes a metric so you know what the number means before trusting it. (Tool details change fast; verify current docs.)

---

## C. Failure Analysis

**Overview:** Systematically finding *why* answers fail, not just *that* they fail.

**How it's used:** The 2×2 mental model — retrieval vs. generation, and "got the right stuff" vs. "used it well." Low context recall → retriever. Low faithfulness with good context → generator. Categorize failures, count them, attack the biggest bucket first.

**Practical example:** Of 50 bad answers, 35 have low context recall (retrieval gap) and 15 are hallucinations (generation) → prioritize chunking/embeddings before prompt tweaks.

**Key consideration:** Build an error taxonomy early and tag every failure. Aggregate metrics tell you something's wrong; per-query debugging tells you what to fix.

---

## D. Production & Observability

**Overview:** Evaluation isn't one-time. Offline eval runs on a fixed golden set before shipping; online eval monitors live traffic after.

**How it's used:**
- **Offline:** regression-test every change against the golden set so you don't silently degrade quality.
- **Online:** sample live queries, run LLM-as-a-judge or collect user feedback, watch for drift.
- **Tracing/observability:** log query → retrieved chunks → prompt → answer for each request so any failure is reconstructable.

**Practical example:** A new embedding model raises Recall@5 offline but online faithfulness drops a week later because the corpus shifted — tracing lets you see which chunks changed.

**Key consideration:** Continuous evaluation is the point — corpora, models, and user behavior all drift. Keep the golden set version-controlled and treat eval scores like test suites: block regressions before they ship.

---

## How the two parts connect

| Symptom | Likely stage | Metrics that localize it |
|---|---|---|
| Answer omits known facts | Retrieval | Context recall, Recall@k |
| Context window full of junk | Retrieval | Precision@k, context precision |
| Right facts retrieved, answer makes things up | Generation | Faithfulness, hallucination rate |
| Answer grounded but off-topic | Generation | Answer relevance |

Diagnose retrieval first (Part 1) — if the right context never arrives, no prompt fixes the generator.
