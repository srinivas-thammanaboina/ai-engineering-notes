# Reasoning & Planning

**Takeaway:** The agent loop only works if the model's per-turn decision is good. Reasoning techniques (Chain-of-Thought, ReAct) make the model *think before it acts*; planning techniques decide how much to figure out up front vs. as it goes; reflection lets it notice and fix its own mistakes.

## The problem this solves

In the architecture file, the model's job each turn was "pick the next action." Left alone, an LLM is bad at this for multi-step problems — it jumps straight to an answer or a tool call without working through *why*. Ask "did revenue grow between two filings?" and a naive model might call the comparison tool before it has fetched either number.

The fixes are all variations on one trick: **make the model produce its reasoning as text before it commits to an action.** Tokens are the model's scratch space. Forcing it to write down its thinking measurably improves the decisions that follow.

## Chain-of-Thought: think in steps before answering

Chain-of-Thought (CoT) is the foundational idea, and the paired paper for this module (Wei et al. 2022). The finding is almost embarrassingly simple: if you prompt a model to show its working — "let's think step by step" — it gets multi-step reasoning problems right far more often than if you ask for the answer directly.

```
Direct:    Q: [multi-step math word problem]
           A: 27                      ← often wrong; no room to reason

CoT:       Q: [same problem] Let's think step by step.
           A: First, ... Then, ... So the total is ...
              Therefore the answer is 31.   ← far more often right
```

Why it works: a transformer does a fixed amount of computation per token. A hard problem needs more computation than fits "between" the question and a one-token answer. Writing out intermediate steps gives the model more tokens — and therefore more compute — to reach the conclusion. The reasoning text is the model thinking *on paper* because it can't think silently.

> **Mental model:** The model can't pause and ponder. Its only way to "spend more compute" on a hard problem is to generate more tokens. Chain-of-Thought is just permission — and a nudge — to use those tokens for working rather than blurting the answer.

CoT on its own is a *reasoning* technique for a single answer. The leap to agents is combining it with *acting*.

## ReAct: reason, then act, then observe — repeat

ReAct (Yao et al. 2022) is the pattern almost every agent framework implements, including LangChain's classic agent. It interleaves Chain-of-Thought reasoning with tool calls. Each loop turn has three parts:

```
   Thought:      I need the 2023 revenue. I should search the 2023 10-K.
   Action:       search_filings(query="2023 total revenue", year=2023)
   Observation:  "Total revenue was $94.0B ..."        ← result fed back

   Thought:      Now I need the 2022 figure to compare.
   Action:       search_filings(query="2022 total revenue", year=2022)
   Observation:  "Total revenue was $81.5B ..."

   Thought:      94.0 > 81.5, so revenue grew. I can answer now.
   Action:       finish("Revenue grew from $81.5B to $94.0B ...")
```

The **Thought** is Chain-of-Thought applied to the *next action* instead of to a final answer. The **Action** is a tool call. The **Observation** is the result, fed back into the transcript. Then the cycle repeats. This is literally the loop from `agentic-architecture.md`, with an explicit reasoning step bolted onto the front of every turn.

Why interleaving beats reasoning-then-acting-separately: the model's plan can adapt to what it actually finds. If the 2023 search returns nothing, the next Thought can say "that failed, let me try a different query" — something a fixed up-front plan can't do.

```python
# ReAct in essence: prompt the model to emit a Thought, then either an
# Action or a Finish, and loop on the Observation.
REACT_PROMPT = """Answer the question by reasoning and using tools.
Always respond in this format:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<args>)   OR   Finish(<final answer>)

Available tools: search_filings, lookup_filing, compare_numbers
"""

while turns < MAX_TURNS:
    step = model.create(REACT_PROMPT + transcript)   # emits Thought + Action
    if step.is_finish:
        return step.answer
    obs = run_tool(step.action)                       # execute
    transcript += f"\n{step.thought}\n{step.action}\nObservation: {obs}\n"
```

> **Mental model:** ReAct = Chain-of-Thought *for actions*, in a loop. The "Thought" lines are the model reasoning about its next move; the "Observation" lines are reality pushing back. That feedback is what makes an agent more than a fancy prompt.

## Planning: figure it all out now, or as you go?

There are two ends of a spectrum for *how much* the agent plans before acting.

**Plan-and-execute (plan up front).** First call: the model writes a full plan — "1. find both filings, 2. retrieve revenue from each, 3. compare, 4. write answer." Then it executes the steps. Pro: cheaper and more predictable; you can inspect the plan before running it. Con: brittle — if step 2 returns something unexpected, the rigid plan can't adapt.

**ReAct-style (plan as you go).** No up-front plan; the model decides each next step from what it's learned so far. Pro: adapts to surprises. Con: can wander, repeat itself, or never converge — and costs more tokens because it's reasoning every turn.

```
   PLAN-AND-EXECUTE                    REACT (plan-as-you-go)

   ┌─────────────┐                     ┌──────────────┐
   │ make a plan │                     │ think + act  │◄──┐
   │  1,2,3,4    │                     │  (one step)  │   │
   └──────┬──────┘                     └──────┬───────┘   │
          │                                   │           │
   ┌──────▼──────┐                     ┌──────▼───────┐   │
   │ run step 1  │                     │  observe     │───┘
   │ run step 2  │   ← can't adapt     └──────────────┘
   │ run step 3  │      mid-flight       adapts every turn
   └─────────────┘
```

In practice, mature systems use a **hybrid**: plan up front for structure, but allow re-planning when an observation breaks the plan. For your Filing Analyst Copilot, a light plan-and-execute is the right default — the question types are predictable enough that a 3–4 step plan usually holds, and predictability is worth a lot when you're learning to *diagram* the control flow.

## Reflection: let the agent check its own work

Reflection adds a self-critique step: after producing an answer (or a plan), the model is prompted to evaluate it — "Is this answer fully supported by the retrieved chunks? Did I miss anything?" — and revise if not. Reflexion (Shinn et al. 2023) is the canonical reference: the agent keeps a short verbal record of what went wrong on previous attempts and uses it to do better next time.

```
   attempt ──► [ produce answer ]
                     │
                     ▼
              [ critique: is this right / complete / grounded? ]
                     │
            ┌────────┴────────┐
       good │                 │ flawed
            ▼                 ▼
          DONE        [ revise using the critique ] ──► loop back
```

Reflection is powerful but not free — every critique is another model call. Use it where mistakes are expensive and checkable (e.g. "did I cite a source for every claim?") and skip it where a wrong answer is cheap. For a *citation-grounded* copilot, a single reflection pass that asks "is every sentence backed by a retrieved chunk?" is high-value: it's exactly the check that turns "plausible-sounding" into "actually grounded."

> **Mental model:** Reasoning improves the *next step*. Reflection improves the *whole attempt*, after the fact. One looks forward; the other looks back. Strong agents do both.

## Putting the techniques in order

These aren't competing options — they stack:

1. **Chain-of-Thought** — the base trick: reason in tokens before committing.
2. **ReAct** — CoT applied to actions, in a loop. This is your agent's per-turn engine.
3. **Planning** — how much to decide up front (plan-and-execute) vs. as you go (ReAct-style); hybrid in practice.
4. **Reflection** — a backward-looking self-check layered on top, for expensive or verifiable mistakes.

## Connecting to the project

Your agent's loop runs ReAct: each turn it writes a Thought ("I still need the 2022 figure"), takes an Action (a tool call), and reads an Observation. A light up-front plan keeps it on rails. And because the copilot's whole promise is *grounded* answers, a final reflection step — "is every claim cited?" — is the difference between a demo and something trustworthy.

## See also

- `agentic-architecture.md` — the loop these techniques run inside
- `orchestration-langchain-langgraph.md` — how a graph makes plan / act / reflect into explicit, separate nodes
- Papers: Chain-of-Thought (Wei et al. 2022), ReAct (Yao et al. 2022), Reflexion (Shinn et al. 2023)
