# LoRA & QLoRA

> **Takeaway:** Full finetuning is expensive because *training* a weight costs far more memory than *storing* it. LoRA sidesteps this by freezing the original weights and learning a tiny pair of low-rank matrices alongside them — training ~1% of the parameters for nearly the same result. QLoRA goes further: it also squashes the frozen model down to 4-bit so the whole thing fits on hardware you actually have.

---

## Why full finetuning is so expensive

The surprising fact: **storing a model is cheap; training it is not.** A model that takes X memory to *run* takes roughly 3–4× X to *train*. Understanding why is the key that unlocks why LoRA exists.

When you train, GPU memory holds four things, not one:

1. **The weights** — the model's numbers. Call this size W.
2. **The gradients** — for every weight, a number saying which direction to nudge it. Same count as the weights, so another ~W.
3. **The optimizer state** — modern optimizers (Adam, the standard choice) keep *two* extra bookkeeping numbers per weight (a running average of recent gradients and of their squares). That's another ~2W.
4. **Activations** — intermediate values from the forward pass, needed to compute gradients. Varies, but non-trivial.

So full finetuning of a model costs roughly **W + W + 2W = 4W** in memory before activations. A 7-billion-parameter model is ~14GB just to store in the usual half-precision; full finetuning it needs ~56GB+ — multiple high-end data-center GPUs. That's the wall LoRA was built to get around.

```
FULL FINETUNING — memory per weight
┌──────────────┬──────────────┬───────────────────────────┐
│  weights     │  gradients   │  optimizer state (×2)       │
│   ~W         │   ~W         │   ~2W                       │
└──────────────┴──────────────┴───────────────────────────┘
        every one of these scales with the FULL model size
```

The crucial realization: items 2 and 3 — gradients and optimizer state, three-quarters of the cost — only exist for weights you're *actually training*. Freeze a weight and it costs you W-storage and nothing else. **So what if we froze almost everything and trained only a tiny add-on?** That's LoRA.

---

## The low-rank idea (the heart of LoRA)

LoRA stands for **Low-Rank Adaptation**. To get it, you need one piece of intuition about what finetuning *changes*.

When you finetune, the end result is just the original weights plus some change: `W_finetuned = W_original + ΔW`, where ΔW (delta-W) is the total adjustment finetuning made. Full finetuning learns ΔW directly, and ΔW is the same enormous size as W.

The LoRA insight, backed by the paper: **that change ΔW has low "intrinsic rank."** In plain terms — the adjustment finetuning makes is far simpler than its size suggests. It contains a lot of redundancy and can be reconstructed from much less information, the way a smooth photo gradient can be described by a few numbers instead of every pixel. "Rank" here is a measure of how much genuinely independent information a matrix holds; "low rank" means *not much*.

If ΔW is genuinely low-rank, you don't need to store or learn the whole thing. You can **factor** it into two skinny matrices whose product reconstructs it:

```
ΔW  (big: d × d)        ≈        B (d × r)   ×   A (r × d)
                                  tall, skinny    short, wide
   millions of numbers              r is TINY (e.g. 8, 16, 32)

Example, d = 4096, r = 8:
  ΔW  full   : 4096 × 4096  = 16,777,216 numbers
  B    : 4096 × 8     =     32,768
  A    :    8 × 4096  =     32,768
  LoRA total          =     65,536  numbers  →  ~0.4% of ΔW
```

`r` is the **rank** — the one knob that controls how much capacity the adapter has. Small `r` (8–16) is usually plenty; bigger `r` gives more expressive power at more cost. That ~0.4% is the whole trick: you train *those two skinny matrices only*, and because they're tiny, the gradient + optimizer cost (the expensive 3W from before) is now ~0.4% of what it was.

---

## How LoRA runs — frozen base, trainable side path

Mechanically, LoRA leaves the original weight matrix W completely **frozen** and adds the low-rank product as a parallel side path. For any input x, the layer's output becomes:

```
                ┌─────────────────┐
        ┌──────▶│  W  (FROZEN)    │──────┐
        │       └─────────────────┘      │
   x ───┤                                ▼
        │       ┌─────┐   ┌─────┐      (  +  ) ───▶  output
        └──────▶│  A  │──▶│  B  │────────┘
                └─────┘   └─────┘
                 trainable  trainable
                 (scaled by alpha / r)

   output  =  W·x   +   (alpha/r) · B·(A·x)
              frozen        trainable adapter
```

Two parameters you'll set:

- **`r` (rank)** — adapter capacity, as above. Start at 8 or 16.
- **`alpha`** — a scaling factor on the adapter's contribution. The adapter's effect is scaled by `alpha / r`, which keeps the adapter's influence steady even as you change `r`. A common convention is `alpha = 2 × r`. Treat `alpha/r` as "how loud the adapter is."

At initialization, A is random and **B is all zeros** — so `B·(A·x)` is zero and the model starts out behaving *exactly* like the untouched base. Training then grows B away from zero. This is a deliberate, elegant detail: the finetune starts as a no-op and only departs from the base as far as the data demands, which is part of why LoRA is stable and resists catastrophic forgetting (the base weights are literally never touched).

Two consequences worth internalizing:

- **Adapters are tiny and swappable.** A trained LoRA adapter is just those A/B matrices — a few megabytes. You can keep one base model in memory and hot-swap different adapters for different tasks. Compare that to full finetuning, where each task is a whole separate multi-gigabyte model.
- **You can merge for deployment.** Since `W + (alpha/r)·B·A` is just arithmetic, you can fold the adapter back into the weights once, producing a normal standalone model with no runtime overhead. You'll do exactly this in the project before serving via Ollama. (Merged = simple to serve but baked in; unmerged = swappable but adds a little inference cost. Both are valid.)

---

## QLoRA — fitting it on hardware you have

LoRA shrinks the *training* cost (gradients + optimizer). But you still have to hold the frozen base model in memory to run the forward pass — and a 7B model is still ~14GB. On a free Colab GPU or a Mac, that's tight or impossible. **QLoRA** ("Quantized LoRA") closes that last gap.

The core move is **quantization**: store the frozen base model's numbers at lower precision. Precision is how many bits you spend per number — fewer bits, less memory, slightly less exactness. QLoRA stores the frozen base in **4-bit** instead of the usual 16-bit, cutting the base's memory footprint by ~4×. The ~14GB model becomes ~3.5GB and now fits.

The clever part is *how* it quantizes without wrecking quality. Three pieces, intuition-level:

- **4-bit NormalFloat (NF4).** A number format tuned for the fact that neural-net weights cluster around zero in a bell-curve shape. Instead of spacing its 16 possible values evenly, NF4 places them where the weights actually are — fine-grained near zero, coarser in the tails — so you lose less information than naive 4-bit would. Think of it as a ruler with more tick marks where you actually measure things.
- **Double quantization.** Quantization needs small "scaling constants" to map values back to their real range. QLoRA quantizes *those constants too* — a small saving that adds up over a large model. Quantizing the quantization metadata.
- **Paged optimizers.** A safety valve: when a memory spike would overflow the GPU, optimizer state is temporarily shuffled to CPU memory instead of crashing — the way an OS pages memory to disk. Lets you train near the edge of your GPU's capacity without out-of-memory failures.

The combination that makes QLoRA click: **the frozen base sits in cheap 4-bit, and the trainable LoRA adapter stays at full precision.** You get tiny, accurate adapters trained on top of a heavily compressed base — so finetuning a 7B model fits in a single consumer or free-tier GPU. That's why QLoRA is your default for the project's Colab training.

```
QLoRA memory picture
┌────────────────────────────┐
│  base model, FROZEN, 4-bit  │   ← ~4× smaller, never trained
│         ~3.5GB for 7B       │
├────────────────────────────┤
│  LoRA adapter, full precision│  ← tiny, this is all you train
│         a few MB            │
└────────────────────────────┘
   forward pass dequantizes 4-bit weights on the fly as needed
```

> **Mental model:** LoRA freezes the model and trains a tiny side path, so you stop paying the 3× gradient-and-optimizer tax on the whole thing. QLoRA additionally crushes the frozen part to 4-bit so even *holding* it is cheap. LoRA makes training affordable; QLoRA makes it fit.

---

## Picking settings in practice

You don't need to tune much. Sensible starting points:

- **`r` = 8 or 16.** Bump to 32+ only if the task is complex and you have the data to justify it.
- **`alpha` = 2 × r.** A standard convention; controls adapter "loudness."
- **Which layers to adapt.** LoRA is usually applied to the attention projection matrices; many libraries default to adapting more layers, which often helps. Start with the library default.
- **Dropout** on the adapter (a small value like 0.05) helps regularize on small datasets.

The honest guidance from your `CLAUDE.md` framing: don't over-tune. Get a working LoRA finetune first, observe the before/after, *then* adjust one knob at a time if you want to understand its effect. The point is to feel what `r` does, not to chase a leaderboard.

---

## Tying it to the project

In the module project you'll QLoRA-finetune a small open model on Colab to emit your Filing Copilot's citation-grounded house format. The pieces map directly:

- **QLoRA on Colab** because the free GPU can't hold the base model otherwise — this is exactly the constraint QLoRA was built for.
- **Tiny adapter out** — your finetune output is a few-MB A/B adapter, not a new multi-GB model.
- **Merge before Ollama** — you'll fold the adapter into the base (`W + (alpha/r)·B·A`) to produce one standalone model, then convert and serve it locally. The merge step is why the "frozen base + side path" structure matters even at deployment time.

---

## Papers

- **LoRA (Hu et al., 2021)** — *LoRA: Low-Rank Adaptation of Large Language Models.* The low-rank-ΔW hypothesis, the frozen-base + A/B side-path design, and the evidence that small `r` suffices. The source for everything in the first half of this file.
- **QLoRA (Dettmers et al., 2023)** — *QLoRA: Efficient Finetuning of Quantized LLMs.* Introduces NF4, double quantization, and paged optimizers — the second half here. Read LoRA first; QLoRA assumes it.
- **PEFT survey (Lialin et al.)** — places LoRA within the broader family of parameter-efficient methods; covered next in `peft-and-moe.md`.
