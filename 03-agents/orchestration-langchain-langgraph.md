# Orchestration — LangChain & LangGraph

**Takeaway:** LangChain gives you the building blocks (model wrappers, tool abstractions, prebuilt agents) but hides control flow inside a black-box loop. LangGraph makes the control flow an explicit, inspectable *state machine* — nodes, edges, and conditions you can draw. For anything beyond a toy agent, you want the graph.

## Why you need a framework at all

The `while` loop in `agentic-architecture.md` works, but you saw its weaknesses: no clean way to branch, retry, persist state, or resume. You *could* hand-roll all of that, and for a learning project that's actually instructive. But two things recur in every agent — (1) a typed list of tools the model can call, and (2) a controlled loop with stopping logic — and frameworks exist so you don't rebuild them each time. The question is which framework, and how much it hides.

## LangChain — the building blocks

LangChain is the older, broader library. Its value is a big set of **standard abstractions** so you can swap pieces without rewriting:

- **Model wrappers** — one interface over Claude, GPT, Gemini, local models. Swap providers by changing one line.
- **Tools** — a standard way to define a callable the model can use (name, description, args schema) — the same shape as Module 1 function calling.
- **Prompt templates** — parameterised prompts.
- **Chains** — fixed sequences: prompt → model → parse → next prompt. Deterministic, no model-driven branching.
- **Agents** — a prebuilt ReAct loop: give it tools and a model, and it runs the reason→act→observe cycle for you (its `AgentExecutor`).

```python
# LangChain's prebuilt agent: you supply tools + model, it runs the loop.
from langchain.agents import create_react_agent, AgentExecutor

tools = [search_filings_tool, lookup_filing_tool, compare_numbers_tool]
agent = create_react_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=6)

result = executor.invoke({"input": "Did revenue grow between the 2022 and 2023 10-K?"})
# The reason→act→observe loop runs INSIDE executor.invoke — you don't see it.
```

The catch is in that last comment. The loop runs *inside* `AgentExecutor`. You hand it tools and a question; it hands back an answer. When it does something weird — calls the same tool twice, loops to the cap, picks the wrong tool — the decision-making is buried in the executor. For a demo that's fine. For something you need to debug, control, and *diagram*, the black box is the problem.

> **Mental model:** A LangChain **chain** is a fixed pipeline — the developer decides the order (like Module 2's RAG: retrieve → generate). A LangChain **agent** is a model-driven loop — the model decides the order. The agent is more flexible but, in classic LangChain, less inspectable.

## LangGraph — control flow as an explicit graph

LangGraph (from the same team) is the answer to the black box. Instead of hiding the loop, you **draw it**: an agent is a graph of **nodes** (steps) connected by **edges** (transitions), passing a shared **state** object around. You're building an explicit state machine.

The three concepts:

- **State** — a typed object that flows through the graph and accumulates results. For your copilot: the question, retrieved chunks so far, extracted numbers, the running plan, the draft answer. Every node reads and updates it.
- **Nodes** — units of work. A node is just a function that takes the state and returns an updated state. "Call the model," "run the retrieval tool," "reflect on the answer" are each a node.
- **Edges** — transitions between nodes. A **normal edge** always goes A→B. A **conditional edge** runs a function on the state to *decide* where to go next — this is how you express "if the model asked for a tool, go to the tool node; if it produced an answer, go to END."

```
                    ┌─────────────────────────────────────┐
                    │              STATE                    │
                    │  question, chunks, numbers, answer    │
                    └─────────────────────────────────────┘
                              ▲ read/write at every node

   START ─► ┌────────┐  conditional edge:  ┌──────────┐
            │ MODEL  │ ── tool needed? ──►  │  TOOLS   │
            │  node  │                      │  node    │
            └────────┘ ◄──── result ──────  └──────────┘
                 │
                 │ answer ready?
                 ▼
               END
```

That conditional edge out of the MODEL node *is* the agent loop — but now it's a labelled arrow you can point at, not a hidden `if` inside an executor. This is exactly why the project asks you to build on LangGraph: the "done when" is *"I can diagram its control flow from memory,"* and LangGraph makes the diagram and the code the same thing.

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)        # AgentState = your typed state object

graph.add_node("model", call_model)   # decide next action
graph.add_node("tools", run_tools)    # execute the requested tool

graph.set_entry_point("model")

# Conditional edge: look at the state, decide where to go next.
def route(state):
    return "tools" if state["needs_tool"] else END

graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
graph.add_edge("tools", "model")      # after a tool runs, back to the model

app = graph.compile()
```

Read that against the diagram: `model` is the brain node, `tools` is the hands node, `route` is the conditional edge, and `add_edge("tools", "model")` is the loop-back. The whole control flow is four lines and it matches a picture you can draw.

> **Mental model:** LangChain's agent is a loop you *trust*. LangGraph is a loop you *draw*. Same ReAct cycle underneath — the difference is whether the control flow is hidden in a library or sitting in front of you as nodes and edges you can inspect, branch, and resume.

## What the graph buys you that the black box doesn't

Making control flow explicit isn't academic — it directly fixes the production weaknesses from `agentic-architecture.md`:

- **Branching** — a conditional edge can route "retrieval came back empty → go to a query-rewrite node → try once more" vs. "good chunks → go straight to answer." You can't express that cleanly inside `AgentExecutor`.
- **Persistence & resume** — because state is an explicit object, LangGraph can checkpoint it. You can pause an agent, inspect the state, and resume — essential for human-in-the-loop approval ("let me OK this before you call the tool").
- **Inspectability** — when something goes wrong, you look at which node ran and what the state was, instead of reverse-engineering a hidden loop.
- **The reflection node** — the self-check from `reasoning-and-planning.md` is just another node with a conditional edge back to "revise" or forward to END.

## How it ties back to the protocols

The tools a LangGraph node calls can be hand-wired functions *or* tools exposed by an MCP server. LangChain/LangGraph can consume MCP servers as tool sources, so the protocol layer (`protocols-mcp-a2a.md`) and the orchestration layer compose: MCP supplies *what the agent can do*, LangGraph controls *when and in what order it does it*.

## Connecting to the project

This is the file the project is built on. Your Filing Analyst Copilot becomes a LangGraph state machine: a `model` node that plans and picks tools, a `tools` node wrapping the Module 2 retrieval plus a metadata-lookup and a number-comparison tool, a conditional edge that loops until the model has what it needs, and optionally a `reflect` node that checks every claim is cited before reaching END. Keep it small — three tools, a handful of nodes — because the win condition is being able to *draw the whole thing from memory*, naming every node, every edge, and the condition on each edge.

## See also

- `agentic-architecture.md` — the raw loop that LangGraph turns into an explicit graph
- `reasoning-and-planning.md` — ReAct (the loop) and reflection (a node) made concrete here
- `protocols-mcp-a2a.md` — MCP tools that a LangGraph node can call
