# Protocols — MCP & A2A

**Takeaway:** MCP standardises how an agent connects to *tools and data*; A2A standardises how an agent connects to *other agents*. They're complementary plumbing — MCP is the agent's connection to its hands, A2A is the connection to its colleagues. Both exist to kill bespoke glue code.

## The problem both protocols solve: N×M glue

In `agentic-architecture.md` you wired a tool in by hand — name, description, JSON schema, and a `run_tool` function. That's fine for one tool. But the industry has *N* tools (GitHub, Slack, your filing store, a database) and *M* model front-ends (Claude, ChatGPT, Gemini, your custom agent). Wiring every tool to every front-end by hand is N×M custom integrations, and every new pairing needs new glue.

Protocols collapse that. Build a tool once against a standard, and any compliant agent can use it. Build an agent once against a standard, and it can talk to any compliant agent. MCP solves it for tools; A2A solves it for agents.

```
   WITHOUT a protocol (N×M glue)        WITH a protocol (N+M)

   ChatGPT ─┬─ GitHub                   ChatGPT ─┐
            ├─ Slack                     Claude  ─┤
            └─ Filings                    Gemini ─┴─► [ PROTOCOL ] ─┬─ GitHub
   Claude ──┬─ GitHub                                               ├─ Slack
            ├─ Slack                                                └─ Filings
            └─ Filings        ← every line          build each side once,
   ...        is custom code               they all interoperate
```

> **Mental model:** A protocol is a standard wall socket. Without it, every appliance needs a custom wire to every power source. With it, anything with the right plug works anywhere. MCP is the socket for tools; A2A is the socket for agents.

## MCP — connecting agents to tools and data

The **Model Context Protocol (MCP)**, introduced by Anthropic in late 2024 and now governed by the Linux Foundation (donated to the Agentic AI Foundation in December 2025), is the open standard for connecting models to external tools, data, and prompts. It's been adopted across the industry — Anthropic, OpenAI, Google, and Microsoft all support it — so a tool you expose via MCP works with all of them.

### The three roles

MCP uses a **host → client → server** architecture, built on JSON-RPC 2.0 messages:

- **Host** — the application the user interacts with (Claude Desktop, an IDE like Cursor, or your custom agent runtime). The host owns the model's context window, decides when to invoke a tool, and feeds results back into the conversation. It's the "conversation controller."
- **Client** — lives inside the host and maintains a *one-to-one* connection to a single server. It translates the model's requests into MCP's format, handles the session, and parses responses back. One host can run many clients.
- **Server** — where the actual capabilities live. A server exposes some set of tools/data (your SEC filing store, a GitHub connector, a database) and answers the client's calls.

```
   ┌──────────────────────── HOST (your agent app) ─────────────────┐
   │   manages the model + context window + when to call tools      │
   │                                                                │
   │   ┌──────────┐        ┌──────────┐        ┌──────────┐         │
   │   │ client A │        │ client B │        │ client C │         │
   │   └────┬─────┘        └────┬─────┘        └────┬─────┘         │
   └────────┼───────────────────┼───────────────────┼──────────────┘
            │ JSON-RPC          │                   │
       ┌────▼─────┐       ┌─────▼────┐        ┌──────▼─────┐
       │ server:  │       │ server:  │        │ server:    │
       │ filings  │       │ github   │        │ database   │
       └──────────┘       └──────────┘        └────────────┘
```

### What a server exposes — the three primitives

An MCP server can offer three kinds of things, each with standard list/get/call methods:

- **Tools** — executable actions the model can call (e.g. `search_filings(query)`). This is the function-calling mechanism from Module 1, standardised.
- **Resources** — read-only data the host can pull into context (e.g. a specific filing's text, a file). Think "things to *read*" vs. tools' "things to *do*."
- **Prompts** — reusable prompt templates the server provides (e.g. a vetted "summarise this 10-K section" template).

### How a connection starts

Every MCP session begins with a **capability-negotiation handshake**: the client sends an `initialize` request with its protocol version and capabilities; the server replies with its own; the client confirms. Both sides then know what features they can use. The protocol is versioned by date (e.g. `2025-06-18`), and version negotiation here prevents mismatched client/server from talking past each other.

> **Mental model:** MCP turns "I hand-wrote a `run_tool` for this one tool" into "I run an MCP client, point it at any server, and the tools show up." Your Module 2 retrieval pipeline can become an MCP *server* exposing one `search_filings` tool — and then *any* MCP host, not just your agent, can use it.

## A2A — connecting agents to other agents

The **Agent2Agent (A2A) protocol**, announced by Google in April 2025 and donated to the Linux Foundation in June 2025, solves the *other* half: letting agents built by different teams, frameworks, or vendors discover each other, delegate tasks, and coordinate — without exposing their internal logic, memory, or tools. Before A2A, a LangChain agent could only easily talk to other LangChain agents; cross-vendor handoffs meant bespoke glue.

A2A runs over familiar web tech — HTTP, Server-Sent Events, and JSON-RPC 2.0 — and rests on a few core ideas:

- **Agent Card** — a metadata document (JSON, served over HTTP) describing an agent's identity, capabilities, and how to reach it. This is the *discovery* mechanism: a client agent reads another agent's card to learn what it can do, the same way a browser reads a site's metadata. Capabilities can be extended via the card (e.g. a payments extension declares itself here).
- **Tasks** — the unit of delegated work, with a defined lifecycle (pending → in-progress → completed / failed). One agent hands another a task and tracks its state to completion.
- **Security** — built-in support for auth (OAuth 2.0, API keys, mTLS) so you control which agents may participate.

```
   ┌──────────────┐   1. read Agent Card    ┌──────────────┐
   │  Agent A     │ ──────(discover)──────►  │  Agent B     │
   │ (orchestrator)│                          │ (specialist) │
   │              │   2. delegate a Task      │              │
   │              │ ──────(HTTP/JSON-RPC)──►   │              │
   │              │ ◄─── 3. task updates ────  │              │
   │              │      (pending→done)        │              │
   └──────────────┘                           └──────────────┘
```

## MCP vs. A2A — the one-line distinction

This is the comparison interviewers reach for, so lock it in:

| | **MCP** | **A2A** |
|---|---|---|
| Connects | agent → **tools & data** | agent → **other agents** |
| Created by | Anthropic (Nov 2024) | Google (Apr 2025) |
| Governance | Linux Foundation (AAIF) | Linux Foundation |
| Transport | JSON-RPC 2.0 | HTTP + SSE + JSON-RPC 2.0 |
| Discovery unit | server primitives (tools/resources/prompts) | Agent Card |
| Work unit | a tool call | a Task (with lifecycle) |
| Question it answers | "what can this agent *use*?" | "who can this agent *delegate to*?" |

They're **complementary, not competing** — a real multi-agent system uses both. An orchestrator agent uses A2A to delegate a sub-task to a specialist agent; that specialist uses MCP to call the actual tools and data it needs to do the work.

> **Mental model:** MCP is the agent's connection to its *hands* (tools). A2A is its connection to its *colleagues* (other agents). One agent can — and often does — use both at once.

## Connecting to the project

Module 3's project is a *single-agent* system, so MCP is the directly relevant protocol: your Module 2 retrieval pipeline can be wrapped as an MCP server exposing `search_filings`, making it reusable by any MCP host rather than hard-wired into one agent. A2A only comes into play if you later split the copilot into multiple cooperating agents (e.g. a "retriever agent" and an "analyst agent") — worth understanding now so the architecture choice is deliberate, but you don't need it to ship the project.

For the build, you don't *have* to use MCP — a hand-wired tool (the `agentic-architecture.md` approach) is simpler for a learning project and keeps the control flow visible. The value of knowing MCP is (a) interviews will ask, and (b) it's how you'd make `search_filings` reusable beyond this one agent.

## See also

- `agentic-architecture.md` — the hand-wired tool-calling that MCP standardises
- `orchestration-langchain-langgraph.md` — frameworks that can consume MCP servers as tools
- Spec sources: modelcontextprotocol.io, a2a-protocol.org
