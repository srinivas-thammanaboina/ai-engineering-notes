# Context Engineering

> What goes in the prompt, and why. This is the strategy layer; RAG (next file) is the mechanism that feeds it.

## The one-sentence truth

The prompt isn't one thing you write — it's a **budget** you allocate. Context engineering is the discipline of deciding, per request, what earns a seat in the finite context window, and *where* it sits.

## Why this is a discipline and not just "writing a good prompt"

Module 1 left us with a hard limit: the context window is finite, and a single 10-K is ~80,000 tokens. The naive instinct is "models have 200k windows now, just paste the whole filing in." That fails for three separate reasons, and you need all three in your head:

1. **Cost** — you pay per input token. Stuffing 80k tokens into *every* question is expensive at any real query volume.
2. **Latency** — time-to-first-token scales with input length. Big contexts are slow contexts.
3. **Accuracy** (the non-obvious one) — models get measurably *worse* at using information buried in the middle of a long context, even when it's right there. This is the **lost-in-the-middle** effect (Liu et al.).

That third point is the one that surprises engineers. It's not just "can it fit" — it's "can the model actually *use* what you gave it." Accuracy is highest when the relevant passage is near the **start** or **end** of the context, and dips in the middle:

```
retrieval
accuracy
  high │●                                             ●
       │ ●                                          ●
       │   ●                                      ●
       │      ●                              ●
       │          ●                     ●
       │              ●●●●●●●●●●●●●●●
   low │                  (the dip)
       └────────────────────────────────────────────────
        start          middle of context            end
              position of the relevant passage
```

So the goal of this entire module reframes: **don't give the model everything — give it the right thing, in the right place.**

## The four ingredients competing for the budget

Everything you put in a context window is one of four things. They all draw from the same finite token budget, so they *compete*.

```
┌─────────────────────────────────────────┐
│  CONTEXT WINDOW  (finite token budget)   │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ INSTRUCTIONS                        │ │  system prompt, role, rules, format
│  ├────────────────────────────────────┤ │
│  │ KNOWLEDGE                           │ │  retrieved chunks  ← this is RAG
│  ├────────────────────────────────────┤ │
│  │ TOOLS                               │ │  function defs, tool results
│  ├────────────────────────────────────┤ │
│  │ MEMORY                              │ │  chat history, summaries, state
│  └────────────────────────────────────┘ │
│                  │                       │
│                  ▼                       │
│             [ the model reads → generates ]
└─────────────────────────────────────────┘
```

- **Instructions** — who the model is, what rules it follows, what output format you want. Cheap in tokens, high in leverage. (The "answer only from the provided filing excerpts; if the answer isn't there, say so" instruction is what later lets the Filing Analyst *refuse* instead of hallucinate.)
- **Knowledge** — the facts the model doesn't have. For us: the retrieved chunks of the actual 10-K. This is the slot RAG fills, and it's usually the biggest consumer of budget.
- **Tools** — definitions of functions the model can call, plus the results that come back. Each tool definition costs tokens whether or not it's used.
- **Memory** — conversation history, running summaries, persisted state. Grows over a session and quietly eats budget.

The discipline is in the tradeoffs: every token of retrieved knowledge is a token you *can't* spend on conversation history. Every extra tool definition crowds out instructions. There's no "just add more" — there's only "what do I cut to make room."

## Placement: the lever most people don't pull

Because of lost-in-the-middle, *order* matters as much as content. A few rules of thumb that hold up in practice:

- **Put the most important retrieved chunk last** — closest to the question, in the high-accuracy zone at the end of the context.
- **Keep stable, reusable content at the top** — system instructions, tool definitions. This also helps with prompt caching (you pay less for the cached, unchanging prefix).
- **Put the user's actual question at the very end**, after the retrieved knowledge, so the model reads the evidence then the question.

A good default layout for a RAG prompt:

```
[ system instructions ]      ← top: stable, cacheable
[ tool definitions ]         ← stable
[ retrieved chunk #3 (weakest) ]
[ retrieved chunk #2 ]
[ retrieved chunk #1 (strongest) ]   ← strongest evidence nearest the question
[ user question ]            ← very end: high-accuracy zone
```

## Context rot: the slow failure

Over a long agent run or a long chat, the context fills with stale tool outputs, superseded answers, and dead ends. The model starts attending to irrelevant history — quality degrades even though nothing "broke." This is **context rot**, and the fixes are all about *curation*:

- **Summarize** old turns instead of carrying them verbatim.
- **Evict** tool results once they've been used.
- **Re-retrieve** fresh knowledge per question rather than accumulating it.

You'll feel this hard in Module 3 (agents), where loops can run for many steps. For now, just internalize: context is a resource you actively manage, not a log you append to.

## Worked example: Filing Analyst

A user asks: *"What are Tesla's biggest stated risks this year?"*

The wrong way (and why it fails):
```
[ entire Tesla 10-K, ~80,000 tokens ]
[ question ]
```
Fails on all three axes — expensive, slow, and the risk factors section is somewhere in the middle where the model reads it worst.

The context-engineered way:
```
[ system: "You answer only from the provided filing excerpts.
            Cite the section for every claim. If the answer
            isn't in the excerpts, say so." ]
[ excerpt: Risk Factors, supply chain        (chunk #3) ]
[ excerpt: Risk Factors, regulatory          (chunk #2) ]
[ excerpt: Risk Factors, competition         (chunk #1, strongest match) ]
[ question: "What are Tesla's biggest stated risks this year?" ]
```
~2,000 tokens instead of 80,000, every piece in a high-attention position, and the instruction makes the model grounded and refusable.

But notice the gap this leaves: **how did we know *which* excerpts to pull?** We can't read Tesla's mind about which chunks match the question. That selection problem — finding the right knowledge to fill the KNOWLEDGE slot — is exactly what RAG solves. That's the next file.

## Mental model to carry forward

- The prompt is a **budget**, not a blank page.
- Four ingredients compete: **instructions, knowledge, tools, memory**.
- **Placement matters** because of lost-in-the-middle — strongest evidence and the question go near the end.
- Context is **curated, not accumulated** — watch for context rot.
- Context engineering decides *what* belongs; **RAG decides *how to find it*.**
