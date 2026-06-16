# Finetuning Fundamentals

> **Takeaway:** Finetuning changes the model's *weights* to bake in a new behavior; prompting and RAG change the model's *input* to steer behavior it already has. Most problems people reach for finetuning to solve are actually input problems — so the first real skill is knowing when finetuning is the wrong tool.

---

## The one distinction everything hangs on

A language model is two things stitched together: a giant pile of frozen numbers (the **weights** — the model's baked-in knowledge and instincts) and the **context** you hand it at runtime (your prompt, retrieved documents, examples, chat history).

When you want to change how the model behaves, you can reach for one of two levers:

- **Change the context** — prompting, few-shot examples, RAG. The weights stay frozen; you're feeding the same brain a better question.
- **Change the weights** — finetuning. You're rewiring the brain itself so the new behavior is permanent and needs no instructions at runtime.

Almost every "should I finetune?" question collapses to: *is this a context problem or a weights problem?* Hold that question in your head for the rest of the module — it's the whole game.

A quick analogy. Prompting is **giving someone instructions for a task**. RAG is **handing them the reference binder** so they can look facts up. Finetuning is **sending them to training** so the skill becomes second nature and they no longer need the instructions or the binder. Training is powerful, slow, and expensive — you don't send someone to a six-week course to answer a question they could've Googled.

---

## What finetuning actually does to the weights

When you finetune, you show the model examples of the input→output behavior you want, and you nudge its weights — using the same gradient-descent training that built the model in the first place — so it's more likely to produce those outputs. "Nudge the weights" means: for every example, measure how wrong the model was, then make a tiny adjustment to millions (or billions) of numbers so it's slightly less wrong next time. Repeat over thousands of examples.

The key word is **nudge**. You're not retraining from scratch — you're taking a model that already knows English, reasoning, and world facts, and tilting it toward a narrow behavior. That's why finetuning needs far less data than pretraining: you're editing, not building.

What finetuning is genuinely good at:

- **Form and format** — teaching a consistent output structure, house style, or response shape that's hard to fully specify in a prompt.
- **Behavior and tone** — a reliable persona, a domain register, refusal patterns.
- **Narrow skills** — a classification or extraction task where you have lots of labeled examples and want it fast and cheap at inference (no long prompt, smaller model).
- **Compression** — folding a long, elaborate prompt into the weights so you stop paying for it on every call.

What finetuning is bad at — and this is the part people get wrong:

- **Adding fresh knowledge.** Finetuning is a poor way to teach the model *new facts*. It tends to teach the model the *shape* of your examples rather than reliably memorizing their content, and it can make the model confidently wrong when asked about things adjacent to but not exactly in the training set. New facts belong in context (RAG), not in weights.
- **Keeping knowledge current.** Weights are frozen at training time. Anything that changes — prices, filings, news — can't be finetuned in without retraining. RAG handles change for free.

> **Mental model:** Finetuning teaches a *skill or a style*. RAG supplies *knowledge*. If what you need is "know this fact," you almost never want finetuning.

---

## The decision: prompt → RAG → finetune, in that order

There's a deliberate ladder here, cheapest and fastest first. Climb it only when the rung below genuinely fails.

**1. Prompting (and few-shot).** Start here always. Write a clear instruction. If the behavior is inconsistent, add 2–5 examples in the prompt (few-shot — literally showing the model a handful of input/output pairs so it pattern-matches). Costs nothing but tokens, ships in minutes, changes in seconds. A surprising amount of "we need to finetune" turns out to be "we hadn't written a good prompt yet."

**2. RAG.** Reach here when the problem is *the model doesn't know the relevant facts* — your documents, your data, anything specific or changing. You retrieve the right context and put it in the prompt. The model's job becomes reading comprehension over material you supplied, not recall from training. This is your entire Module 2.

**3. Finetuning.** Reach here only when **both** of these are true:
   - The desired behavior is about *form, style, or a narrow skill* — not about supplying facts (because facts → RAG).
   - Prompting can't get you there reliably, *or* the prompt that works is so long/expensive that baking it into weights pays off at your call volume.

A compact way to decide:

```
Is the problem that the model lacks specific or changing FACTS?
    └─ Yes → RAG. (Finetuning will not fix this well.)
    └─ No → Is it a BEHAVIOR/FORMAT/STYLE/SKILL problem?
              └─ Can a good prompt (+ few-shot) achieve it reliably?
                    └─ Yes → Prompt. Stop. You're done.
                    └─ No  → Is it worth the cost? (data, training, serving,
                             staleness, maintenance)
                               └─ Yes → Finetune.
                               └─ No  → Live with the prompt, or combine:
                                        RAG for facts + light finetune for form.
```

Notice the bottom branch: **RAG and finetuning are not rivals.** The strongest real systems often do both — RAG supplies current, grounded facts in context, and a light finetune locks in the output format and behavior. They solve different problems and stack cleanly.

---

## Full finetuning vs. PEFT (why the next file exists)

When you finetune by adjusting *every* weight in the model, that's **full finetuning**. It works, but it's brutally expensive: to train, you need to hold the model, its gradients, and the optimizer's bookkeeping all in GPU memory at once — roughly several times the model's own size. For anything but small models, that's data-center hardware, not your Mac and not a free Colab GPU.

This expense is the entire reason **PEFT** (Parameter-Efficient Fine-Tuning — methods that train only a tiny fraction of the weights) exists, and why **LoRA/QLoRA** dominate in practice. That's the next file. The thing to carry forward: full finetuning and PEFT produce similar behavior changes, but PEFT does it by training maybe 1% of the parameters, which is what makes finetuning possible on hardware you actually have.

---

## Failure modes to respect

**Catastrophic forgetting.** Push a model hard on a narrow task and it can lose general ability it used to have — like someone who crams so hard for one exam they forget everything else. The narrower and more aggressive your finetune, the bigger the risk. PEFT methods reduce this (the original weights stay frozen — more on that next file), but it's never zero.

**Teaching the shape, not the substance.** Finetune on a small dataset and the model may learn superficial patterns — "answers in this domain start with a bold header and three bullets" — without learning the actual reasoning. It then produces confident, well-formatted nonsense. This is why finetuning is a bad knowledge-injection tool and why your eval set matters.

**Data quality dominates.** A finetune is only as good as its examples. A few hundred clean, consistent, correct examples beat thousands of noisy ones. Garbage in, garbage baked permanently into the weights.

**Distribution mismatch.** The model gets good at inputs that *look like your training examples* and can degrade on anything that doesn't. If your real traffic differs from your training data, the finetune can quietly hurt.

---

## Tying it to the project

Your running project is the **Filing Analyst Copilot** — citation-grounded Q&A over SEC filings. Walk it through the ladder honestly:

- **The facts** (what a given 10-K says) → **RAG**, always. These are specific, per-company, and they change every filing cycle. Finetuning facts in would be exactly the mistake this file warns about.
- **The behavior** (always cite, answer in your house format, refuse cleanly when the retrieved context doesn't support an answer) → this is *form and skill*, prompt-able first, and a legitimate **finetuning** candidate if the prompt that enforces it gets long and you want it cheap and reliable.

So the module's project — finetuning a small model to emit your citation-grounded house format — is a genuine, correct use of finetuning: it's a *behavior* finetune sitting on top of a RAG system that still does all the *knowledge* work. That's the prompt→RAG→finetune ladder working as designed, not a toy.

---

## Paper

Finetuning fundamentals are best learned through the PEFT landscape rather than a single fundamentals paper. The anchor read for this module is the **PEFT survey (Lialin et al., *Scaling Down to Scale Up*)**, which frames why parameter-efficient methods exist and maps the family — we cover it properly in `peft-and-moe.md`. For the "finetuning vs. retrieval for knowledge" question specifically, the RAG paper (Lewis et al., 2020) from Module 2 is the relevant counterpoint: it's the empirical case that knowledge belongs in retrieved context, not weights.
