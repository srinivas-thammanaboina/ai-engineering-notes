# Agentic Architecture

**Takeaway:** An agent is a loop. A model decides on an action, something executes it, the result goes back to the model, and it decides again — repeating until the model decides it's done. Everything else (planning, tools, memory) is detail hung on that loop.

## The one idea: a model in a loop with tools

In Module 2 you built a pipeline: question → retrieve → stuff context → generate → answer. It runs once, start to finish, and it's over. The model never gets a second look at the problem.

An agent breaks that straight line into a loop:

```
                  ┌─────────────────────────────────────┐
                  │                                       │
                  ▼                                       │
   question ─► [ MODEL decides next action ]              │
                  │                                       │
                  ├── "I need to call tool X with args Y" │
                  │            │                          │
                  │            ▼                          │
                  │     [ EXECUTE the tool ]              │
                  │            │                          │
                  │            ▼                          │
                  │     observation (result) ────────────┘
                  │
                  └── "I'm done — here's the answer" ─► STOP
```

The model's job each turn is small: *given everything I know so far, what is the single next thing to do?* It either picks a tool to call, or it decides it has enough to answer. The loop runs that decision over and over.

That's the whole thing. If RAG is "look something up, then answer," an agent is "keep looking things up, in whatever order makes sense, until you can answer." The model is steering instead of riding.

> **Mental model:** RAG retrieves *once* and the developer fixed the order. An agent retrieves (and computes, and looks up) *as many times as it needs*, and the model decides the order at runtime. The cost of that flexibility is unpredictability — which is why most of this module is really about *controlling* the loop.

## The four moving parts

Every agent, however it's dressed up, is built from four things:

**1. The model (the brain).** An LLM that, given the conversation so far plus a list of available tools, outputs either a tool call or a final answer. This is the only "intelligent" part — the rest is plumbing. The model doesn't *run* anything; it just emits structured text saying what it wants to happen.

**2. Tools (the hands).** Functions the model can ask to run: search the filings, look up a ticker, do arithmetic, call an API. Each tool has a name, a description, and a typed schema for its arguments — exactly the function-calling mechanism from Module 1. The model never executes a tool itself; it *requests* a call, your code runs it, and you feed the result back.

**3. Memory (the notebook).** What the agent carries between loop turns. Short-term memory is the running transcript of "I called search, got these chunks, then called compare, got this number" — it lives in the context window. Long-term memory is anything persisted across sessions (past conversations, a vector store of facts). Memory is what stops the agent from being a goldfish that re-solves the same sub-problem every turn.

**4. Control flow (the rails).** The logic that decides whether to loop again or stop, what to do when a tool errors, and how many times the agent is allowed to try before you cut it off. In a naive agent this is a `while` loop with a turn cap. In a real one it's an explicit graph — which is the whole point of LangGraph later in this module.

```
   ┌──────────────────────────────────────────────────────┐
   │                       AGENT                            │
   │                                                        │
   │   ┌─────────┐     wants to      ┌──────────────┐       │
   │   │  MODEL  │ ───── call ─────► │    TOOLS     │       │
   │   │ (brain) │ ◄─── result ───── │   (hands)    │       │
   │   └────┬────┘                   └──────────────┘       │
   │        │ reads / writes                                │
   │        ▼                                               │
   │   ┌─────────┐         ┌──────────────────────┐         │
   │   │ MEMORY  │         │   CONTROL FLOW        │         │
   │   │(notebook)│        │ (loop? stop? retry?)  │         │
   │   └─────────┘         └──────────────────────┘         │
   └──────────────────────────────────────────────────────┘
```

## How one turn actually works

Concretely, here's what happens on a single pass of the loop, using the tool-calling interface you saw in Module 1:

```python
# Tools are declared with a name, description, and typed args schema.
tools = [
    {
        "name": "search_filings",
        "description": "Retrieve relevant chunks from SEC filings for a query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    # ... more tools
]

messages = [{"role": "user", "content": "Did revenue grow between the 2022 and 2023 10-K?"}]

while True:
    resp = model.create(messages=messages, tools=tools)

    if resp.stop_reason != "tool_use":
        # Model chose to answer instead of calling a tool — we're done.
        print(resp.text)
        break

    # Model asked for a tool. Run it ourselves.
    call = resp.tool_call
    result = run_tool(call.name, call.args)      # <-- our code executes it

    # Feed the model's request AND the result back into the transcript,
    # then loop. This growing transcript IS the short-term memory.
    messages.append({"role": "assistant", "content": resp.raw})
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

Three things worth burning in:

- The model **never executes anything**. It emits a request; *your* `run_tool` runs it. This is the security boundary — the model can ask to delete the database, but it can only delete it if you wrote a tool that does and chose to run it.
- The transcript (`messages`) is the memory. Every turn appends the request and the result, so the model sees its whole trail. This is also why long agent runs blow the context budget — the notebook keeps growing.
- The loop ends only when the model stops asking for tools. If it never stops, *you* must (the turn cap). An agent with no stop condition is a bug, not a feature.

## Why control flow is the hard part

The model is genuinely good at deciding "what's the next action." What it's bad at is the boring operational stuff: knowing when to give up, not calling the same tool five times in a row, recovering from a tool that returned an error instead of crashing, and not running forever.

A first agent looks like the `while True` above and works in demos. It falls over in production because:

- **No turn cap** → a confused model loops forever, burning tokens.
- **No error handling** → one tool exception kills the whole run.
- **No state** beyond the transcript → you can't inspect, resume, or branch.
- **Linear only** → you can't say "if retrieval failed, rewrite the query and try once more, otherwise give up."

Every one of those is a *control-flow* problem, not an intelligence problem. The progression in this module is exactly this: start with the loop (this file), make the model's per-turn decision smarter (reasoning & planning), standardise how tools and other agents plug in (protocols), then make the control flow explicit and robust (LangGraph).

> **Mental model:** The model supplies intelligence; the framework supplies discipline. A good agent system spends most of its code not on the LLM call but on the rails around it — caps, retries, state, and the decision of when to stop.

## Connecting to the project

Your Filing Analyst Copilot becomes an agent the moment it can answer a question that *needs more than one lookup*. "How did risk factors change between the 2022 and 2023 10-K, and did revenue move the same way?" can't be answered by one retrieval — the agent has to find both filings, retrieve from each, pull two revenue numbers, compare them, and compose a cited answer. That's the loop earning its keep: the model decides "first find the filings, then retrieve, then compare," and runs each step in turn.

The Module 2 retrieval pipeline doesn't get thrown away — it becomes one *tool* (`search_filings`) that the agent can call. That's the through-line of the whole curriculum: each module's output becomes a component the next module orchestrates.

## See also

- `reasoning-and-planning.md` — how the model makes the per-turn "next action" decision well
- `orchestration-langchain-langgraph.md` — turning the naive `while` loop into a real, diagrammable state machine
- Papers: ReAct (Yao et al. 2022) is the canonical formalisation of this loop
