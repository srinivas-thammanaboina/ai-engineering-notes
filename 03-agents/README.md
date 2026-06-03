# 03 · Agentic Systems — The Agentic Leap

*~3 weeks*

Agents are RAG plus tool use plus control flow — this builds directly on Module 2.

## Topics

- **Agentic architecture** — the agent loop, planning, tool calling, memory
- **Reasoning & planning** — ReAct, Chain-of-Thought, reflection, plan-and-execute
- **Protocols** — MCP (tools) and A2A (agent-to-agent)
- **Orchestration** — LangChain and LangGraph

## Concept files

Read in this order — each builds on the last:

1. `agentic-architecture.md` — what an agent *is*: the loop, the four moving parts (model, tools, memory, control flow), and why an agent is just RAG with a feedback loop bolted on
2. `reasoning-and-planning.md` — how the model decides what to do next: ReAct, Chain-of-Thought, reflection, and single-step vs. multi-step planning
3. `protocols-mcp-a2a.md` — the plumbing: MCP for connecting tools to models, A2A for connecting agents to each other
4. `orchestration-langchain-langgraph.md` — the frameworks: LangChain's chains/agents vs. LangGraph's state-machine model, and why graphs win for anything non-trivial

## Papers

- **Chain-of-Thought** — Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (arXiv:2201.11903)
- **ReAct** — Yao et al. 2022, "ReAct: Synergizing Reasoning and Acting in Language Models" (arXiv:2210.03629) — added because it is the architecture nearly every framework implements
- **Reflexion** — Shinn et al. 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning" (arXiv:2303.11366) — the canonical reference for the reflection loop

A study-guide PDF (`agents-study-guide.pdf`) synthesises all four concept files in our practitioner voice, with rendered diagrams and code — same role the Module 1 and 2 study guides played.

## Project

Evolve the **Filing Analyst Copilot** from a retrieval system into an *agent*. Module 2 gave it the ability to answer one grounded question from retrieved 10-K/8-K chunks. Module 3 gives it the ability to plan and use 2–3 tools to answer a question it could not answer in a single retrieval pass — e.g. *"How did the company's reported risk factors change between the 2022 and 2023 10-K, and did revenue move in the same direction?"*

That question needs: a retrieval tool (the Module 2 pipeline, now callable), a filing-metadata/lookup tool (find the right two filings), and a small numeric tool (compare two revenue figures). The agent has to plan the order, call each tool, and compose a cited answer — and it has to know when to *stop*.

Built as a **LangGraph state machine** so the control flow is an explicit, diagrammable graph rather than hidden inside a framework's agent executor.

## Done when

I have a working agent that plans and uses tools against real SEC filings, and I can diagram its control flow from memory — naming every node, every edge, and the condition on each edge.
