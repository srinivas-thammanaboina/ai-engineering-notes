# PEFT & Mixture of Experts

> **Takeaway:** PEFT is the family of "train a tiny slice, freeze the rest" methods — LoRA is one sibling among several, and they all share the same goal of getting full-finetune behavior at a fraction of the cost. Mixture of Experts (MoE) is a *different* idea entirely: it's a way to build a bigger, cheaper-to-run *base* model by activating only part of it per token. PEFT is about cheap *training*; MoE is about cheap *capacity*. They land in the same module but answer different questions.

---

## Part 1 — PEFT: the family LoRA belongs to

**PEFT** stands for **Parameter-Efficient Fine-Tuning**: any method that adapts a frozen pretrained model by training only a small number of *new or selected* parameters. You already met the headliner (LoRA) in the last file. This section places it among its relatives so you understand the design space, not just one point in it.

The shared premise across all of PEFT: a pretrained model already knows almost everything it needs: language, reasoning, world structure. Adapting it to a task is a *small* change, so you shouldn't have to pay to retrain *all* of it. Every PEFT method is a different answer to "where do I put the small trainable piece?"

There are three broad places to put it.

### 1. Add a small module inside the network — *Adapters*

The original PEFT idea (Houlsby et al., 2019). Insert tiny new layers — "adapter modules" — between the existing frozen layers, and train only those. The bottleneck shape (squeeze the dimension down, then back up) keeps them small.

- **Pro:** modular, and the base is untouched.
- **Con:** because the adapters sit *in the middle* of the forward pass as extra layers, they add a little latency at inference — every token has to pass through them. This is the practical drawback LoRA was designed to avoid (LoRA's side path can be *merged* away; an inserted layer can't be).

### 2. Reparameterize the weight update — *LoRA* (and relatives)

Don't add layers; express the *change* to existing weights in a cheaper form. LoRA's low-rank B·A factorization is the dominant member. The whole previous file was this category.

- **Pro:** mergeable (zero inference cost after merge), tiny, stable, swappable. This combination is why LoRA won in practice.
- **Con:** the low-rank assumption has to hold for your task (it usually does).

### 3. Steer the model with trainable "virtual tokens" — *Prefix / Prompt Tuning*

Instead of touching the model's internals at all, **prepend trainable vectors to the input** and train *only those vectors*. The model is 100% frozen; you're learning the perfect "soft prompt" — a sequence of made-up tokens that aren't real words but that steer the frozen model toward your task.

- **Prompt tuning** (Lester et al.): adds soft tokens only at the input layer. Extremely few parameters; works best on large models.
- **Prefix tuning** (Li & Liang): adds trainable vectors at *every* layer, not just the input — more capacity, more parameters than prompt tuning.

The mental hook: this is the bridge between prompting and finetuning. A normal prompt is words *you* choose; a soft prompt is a vector *gradient descent* chooses — a prompt optimized in the model's own internal language rather than English. You're finetuning the *input* instead of the *weights*.

```
WHERE EACH PEFT METHOD PUTS THE TRAINABLE PIECE

  input ──▶ [soft prompt] ──▶ ┌─────────┐ ──▶ ┌─────────┐ ──▶ output
            ▲ prefix/prompt    │ layer 1 │     │ layer 2 │
            │ tuning           │ +adapter│     │ +adapter│  ◀── adapters:
            (trainable          └────┬────┘     └────┬────┘      new layers
             input vectors)         │               │           inside
                                ┌───┴───┐       ┌───┴───┐
                                │ B·A   │       │ B·A   │   ◀── LoRA:
                                └───────┘       └───────┘       low-rank
                                  side path       side path     side path

  Three families = three answers to "where does the small trainable bit go?"
  • input   → prefix/prompt tuning  (freeze everything, steer the input)
  • inside  → adapters              (insert small trainable layers)
  • alongside → LoRA                (reparameterize the weight change)
```

### How to hold the family in your head

| Method | What's trained | Inference cost added | Notes |
|---|---|---|---|
| Full finetune | all weights | none | baseline; brutally expensive to train |
| Adapters | inserted small layers | small (always) | first PEFT method; not mergeable |
| LoRA / QLoRA | low-rank side path | none after merge | the practical default |
| Prefix tuning | per-layer input vectors | small | model fully frozen |
| Prompt tuning | input-layer vectors only | tiny | fewest params; needs a big base |

The reason LoRA dominates: it's the only one that's *both* tiny to train *and* free at inference (after merging), without inserting anything into the forward pass. The others remain worth knowing because they reveal the design space — and because "finetune the input, not the weights" (prompt tuning) is a genuinely different and useful idea.

---

## Part 2 — Mixture of Experts: a different question entirely

Here's the honest framing first, because it's easy to conflate with PEFT: **MoE is not a finetuning method.** It's an architecture for the base model itself. It's in this module because it's the other major "do less compute than the model's size suggests" idea, and seeing it next to PEFT sharpens what each one actually saves. PEFT saves on *training*; MoE saves on *inference compute per token*. Different lever, different problem.

### The idea

A normal ("dense") model runs *every* parameter for *every* token. If the model is 70B parameters, all 70B do work on each token. That's why bigger models are slower and pricier — capacity and compute are welded together.

MoE breaks that weld. Replace one big feed-forward block with **many smaller "expert" blocks plus a "router"** (also called a gate). For each token, the router picks just a few experts (commonly the top 2) to actually run; the rest sit idle for that token.

```
DENSE LAYER                    MoE LAYER
                               
 token ─▶ [ one big FFN ] ─▶    token ─▶ [ router ] picks top-2 of N experts
          all params run                   │
                                           ├─▶ [expert 3] ✓ runs
                                           ├─▶ [expert 1]   idle
                                           ├─▶ [expert 7] ✓ runs
                                           └─▶ [expert 2]   idle
                                                   │
                                           weighted sum ─▶ output

  N experts exist (huge total capacity)
  but only ~2 run per token (small compute per token)
```

This splits two things that used to be one:

- **Total parameters (capacity)** = all the experts added up → *large*. This is the model's "knowledge ceiling."
- **Active parameters (compute per token)** = just the few experts that ran → *small*. This is what each token actually costs.

So a MoE model can have, say, the *capacity* of a 47B model while only doing the *compute* of a ~13B model per token. Mixtral 8×7B is the famous example: eight experts, two active per token. You get big-model quality at small-model inference speed — paid for in **memory**, since all experts must be loaded even though most are idle at any moment.

### The router and its one hard problem

The router is a small learned layer that scores the experts per token and sends the token to the top-k. The catch — and the thing the papers spend their effort on — is **load balancing**. Left alone, the router tends to fall in love with a few experts and route everything to them, leaving the rest untrained and useless (a rich-get-richer collapse). The fix is an **auxiliary load-balancing loss** that penalizes lopsided routing and pushes traffic to spread across all experts. Switch Transformer's main contribution is making this routing simple (route to just *one* expert, top-1) and stable at scale.

> **Mental model:** A dense model is a single generalist who handles every token. An MoE model is a large team of specialists with a dispatcher (the router) who hands each token to the two most relevant people. The team's *total* expertise is huge, but each token only pays for two people's time. The dispatcher's hard job is not overloading the same two stars every time.

### Why it's in *this* module, and where it stops

You will **not train an MoE** in this module — building and balancing experts is a pretraining-scale undertaking, not a finetune. What you *should* carry away:

- MoE explains why some open models (Mixtral, and many frontier models) punch above their inference cost — capacity and compute are decoupled.
- It clarifies the PEFT contrast by contrast: PEFT = cheap *adaptation* of a frozen model; MoE = cheap *capacity* in the base model's architecture. Neither is a substitute for the other; a MoE model can itself be LoRA-finetuned.
- When you pick a local model in the next file, knowing whether it's dense or MoE tells you its *memory* demands (MoE: all experts loaded) versus its *speed* (MoE: only a few active) — directly relevant to your Mac's ~4GB ceiling.

---

## Tying it to the project

- **PEFT** is the direct toolset: your project uses LoRA/QLoRA (the reparameterization branch) because it's mergeable and tiny. Knowing the siblings means that if your format-finetune underperforms, you can reason about *why* a different placement (e.g. prefix tuning) might or might not help — rather than treating LoRA as magic.
- **MoE** is context, not a task: if you choose an MoE-based small model to finetune locally, you now understand why it may need more memory than its active-parameter count suggests. That's a real constraint on a Mac, and a reason you might pick a small *dense* model for local serving instead.

---

## Papers

- **PEFT survey — Lialin et al., *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.*** The map of the whole family in Part 1; read this to see LoRA in context.
- **Prefix tuning (Li & Liang, 2021)** and **Prompt tuning (Lester et al., 2021)** — the "finetune the input" branch, if you want primary sources for the soft-prompt idea.
- **Switch Transformer (Fedus et al., 2021)** — the clearest MoE read: top-1 routing and the load-balancing loss. Start here for MoE.
- **Mixtral (Jiang et al., 2024)** — a widely used open MoE (8 experts, top-2); the concrete modern example behind Part 2.
