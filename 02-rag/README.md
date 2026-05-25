# 02 · Grounding AI with RAG & Context Engineering

*~3 weeks*

Module 1 ended on a hard wall: the context window is finite, and a single 10-K filing is ~80,000 tokens. This module is the answer to one question — **how do you get the *right* knowledge into that scarce window?** Everything here (context engineering, RAG, the advanced patterns) is a different answer to "what do I put in the prompt, and how do I find it?"

## Topics

- **Context engineering** — the context window as a budget; the four ingredients (instructions, knowledge, tools, memory) and how they compete; *where* you place information (lost-in-the-middle); why "stuff the whole document in" fails on cost, latency, *and* accuracy
- **The RAG pipeline** — the index → retrieve → augment → generate loop; chunking strategies (fixed, recursive, semantic) and why filings break naive chunkers; embeddings (what a vector "means", cosine similarity, choosing a model); vector databases (HNSW, the major options); retrieval (top-k, similarity metrics, metadata filtering)
- **Advanced RAG** — re-ranking with cross-encoders; hybrid search (dense + sparse/BM25); query rewriting and expansion; and where evals fit (forward reference to Module 5)

## Papers

RAG (Lewis et al. 2020) · Dense Passage Retrieval (Karpukhin et al.) · Lost in the Middle (Liu et al.)

## Project

Extend the Filing Analyst into a **citation-grounded Q&A copilot over SEC filings**: ingest a handful of companies' 10-Ks, chunk and embed them into a vector DB, and answer questions like *"What are Tesla's biggest stated risks this year?"* with every claim cited back to the exact filing section. Add hybrid search + re-ranking once basic retrieval works.

## Done when

I can explain, in plain language, why a question gets the answer it does — which chunks were retrieved and why — and I've shipped a copilot that answers questions over real filings with citations I can click and verify.
