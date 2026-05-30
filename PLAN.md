# AI Engineer Curriculum — Learning Plan

My structured path to learning AI engineering in depth through hands-on building. Steady pace of ~10 hours/week.
Core modules (1–5) take roughly **13 weeks (~3 months)**; Module 6 runs in parallel throughout.

**Weekly split:** ~6 hrs hands-on building · ~2 hrs paper/concept reading · ~2 hrs writing up notes.

---

## Module 1 — Foundations: Gen AI Building Blocks
*~2 weeks · folder: `01-foundations/`*

Everything downstream assumes this. Learn how LLMs actually work and the constraints that shape every real decision.

- [ ] How LLMs work: tokenization, attention (conceptual), next-token prediction
- [ ] The modern AI stack: model providers, APIs, orchestration layers
- [ ] Practical constraints: context windows, latency, cost per token, rate limits

**Papers:** Attention is All You Need · GPT · BERT
**Project:** A CLI or web app that calls an LLM API, handles streaming, and tracks token cost per request.

---

## Module 2 — Grounding AI with RAG & Context Engineering
*~3 weeks · folder: `02-rag/`*

The highest-leverage skill in AI engineering and the most common real-world task.

- [ ] Context engineering: what goes in the prompt and why
- [ ] RAG pipeline: chunking strategies, embeddings, vector databases, retrieval
- [ ] Advanced RAG: re-ranking, hybrid search, query rewriting

**Papers:** RAG (Lewis et al. 2020)
**Project:** A RAG app over a document set I care about (this notes repo is a good candidate).

---

## Module 3 — The Agentic Leap
*~3 weeks · folder: `03-agents/`*

Agents are RAG plus tool use plus control flow — this builds directly on Module 2.

- [ ] Agentic architecture: planning, tool calling, memory
- [ ] Protocols: MCP and A2A
- [ ] Orchestration: LangChain and LangGraph

**Papers:** Chain-of-Thought (Wei et al.)
**Project:** A multi-step agent using 2–3 tools to complete a real task, with a diagrammable LangGraph state machine.

---

## Module 4 — Finetuning & Local Models
*~2.5 weeks · folder: `04-finetuning/`*

Placed after RAG so I learn when *not* to finetune before learning how.

- [ ] Finetuning fundamentals: when it helps, when it doesn't
- [ ] Efficient methods: LoRA / QLoRA
- [ ] Running models locally with Ollama

**Papers:** LoRA · PEFT survey (Lialin et al.) · Mixture of Experts (Switch Transformer / Mixtral)
**Project:** Fine-tune a small open model with LoRA on a narrow task, run it locally via Ollama, compare before/after.

---

## Module 5 — Evals, Security & Production Readiness
*~2.5 weeks · folder: `05-production/`*

You can only evaluate and harden what you've already built.

- [ ] Evaluation: golden datasets, failure analysis, LLM-as-a-judge
- [ ] Observability and monitoring platforms
- [ ] Security: hallucinations, prompt injection, data privacy, red teaming, governance

**Project:** Add an eval harness and basic observability to an earlier project, plus a short red-team writeup of how I broke and fixed it.

---

## Module 6 — Becoming AI-Native
*Ongoing · runs in parallel from Module 2 onward · folder: `06-ai-native/`*

Don't save this for the end — fluency with AI tooling compounds across every other module.

- [ ] Adopt AI-native productivity tools (agentic coding, LLMs as a daily work multiplier) into the actual workflow
- [ ] Build the habit of writing up what I learn — turning each build into an explanation I can reason from
- [ ] Reflect on how working *with* AI changes how I design and debug

**Project (meta):** Each module's project is a real hands-on build, paired with a writeup that explains *why* it works — written to cement my own understanding, not as a deliverable.

---

## Supplementary paper reading list

Read alongside the relevant module — these explain *why* things work. They support the build track but aren't a prerequisite gate.

- Attention is All You Need
- GPT
- BERT
- ViT
- VAE
- GANs
- Diffusion Models (DDPM → Stable Diffusion / Latent Diffusion)
- RAG (Lewis et al. 2020)
- Chain-of-Thought (Wei et al.)
- InstructGPT / RLHF
- LoRA
- PEFT survey (Lialin et al. + Prefix / Prompt Tuning)
- Mixture of Experts (Switch Transformer or Mixtral)
- FlashAttention

---

## Progress

- **Started:** _(fill in)_
- **Current module:** 1 — Foundations
- **Target completion (core):** ~13 weeks from start
