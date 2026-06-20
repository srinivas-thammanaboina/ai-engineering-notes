# Local Models & Ollama

> **Takeaway:** Running a model locally means trading away cloud horsepower for privacy, cost, and control — which only works if you shrink the model enough to fit your machine. The shrinking is **quantization** (fewer bits per weight); the packaging is **GGUF** (a file format built for local inference); and **Ollama** is the tool that hides the plumbing so you can pull, configure, and serve a model in a few commands. This is where your finetuned adapter stops being a Colab artifact and becomes something running on your Mac.

---

## Why run a model locally at all

Everything so far has assumed a model lives behind an API. Local serving flips that — the weights sit on your own machine and inference runs on your CPU/GPU. The reasons you'd want this are concrete:

- **Privacy / data control.** The text never leaves your machine. For SEC filings that's not critical, but for sensitive data (legal, medical, internal company docs) it's often the whole reason.
- **Cost.** No per-token bill. Once it's on your disk, inference is "free" (you pay in electricity and hardware, not API charges).
- **Offline / no dependency.** Works on a plane, in a locked-down environment, with no rate limits and no provider outage.
- **Control and learning.** You see and tune every layer of the stack — exactly the "build and break it" depth your curriculum optimizes for.

The cost of all this is **capability and capacity.** Your Mac is not a data center. You can run *small* models well, *medium* models slowly, and *large* models not at all. The entire craft of local models is fitting the biggest useful model into the memory you have — which is what quantization is for.

---

## Quantization: the lever that makes local possible

You met quantization in QLoRA as a *training* trick. For local serving it's a *deployment* trick, and the same intuition carries: **store each weight in fewer bits to use less memory.**

A model's weights are normally 16-bit numbers (half-precision). Quantizing to 8-bit halves the memory; to 4-bit, quarters it. The trade is exactness — fewer bits means each number is a coarser approximation of the original — but for inference, models tolerate this remarkably well, especially down to 4-bit. Below 4-bit, quality starts dropping off more noticeably.

A rough memory rule for a model with P billion parameters:

```
APPROX. MEMORY TO RUN  ≈  P billion × (bits per weight ÷ 8)  GB

  7B model:
    16-bit (fp16)  →  7 × 2.0  ≈  14 GB     ← won't fit a typical Mac
     8-bit (Q8)    →  7 × 1.0  ≈   7 GB     ← tight
     4-bit (Q4)    →  7 × 0.5  ≈   3.5 GB   ← fits comfortably
     
  (Add some overhead for context/activations on top of these.)
```

This table *is* the reason your project uses small (2–4GB) quantized models: a 4-bit small model is the sweet spot that runs comfortably on a Mac while staying genuinely useful.

### Reading quantization labels (the `Q4_K_M` business)

Local model files come with cryptic suffixes. They decode simply once you know the pattern:

- **The number** = bits per weight. `Q4` = 4-bit, `Q5` = 5-bit, `Q8` = 8-bit. Lower = smaller and faster, higher = more faithful.
- **The letter group** = the quantization *scheme*. `_K` means a "k-quant," a smarter method that spends bits unevenly — more precision on the weights that matter most. Plain `Q4_0` is the older, simpler, slightly-worse scheme.
- **`_S` / `_M` / `_L`** = small / medium / large variant within that scheme — a fine-grained quality-vs-size dial.

In practice **`Q4_K_M`** is the go-to default: 4-bit, k-quant, medium — the widely-agreed best balance of size, speed, and quality for most local use. Reach for `Q5_K_M` or `Q8_0` if you have memory to spare and want more fidelity; drop to `Q3` only if you're desperate for space and willing to accept quality loss.

---

## GGUF: the file format for local inference

A model you download for local use almost always comes as a **GGUF** file. It's worth knowing what it is because your project produces one.

GGUF (it superseded an earlier format called GGML) is a single-file container designed for **inference on consumer hardware, CPU included.** It bundles everything an inference engine needs in one place:

- the quantized weights,
- the model architecture and metadata,
- the tokenizer,
- and the chat/prompt template.

The "single file with everything baked in" design is the point — it makes a model trivially portable and loadable without a Python environment or the original training framework. The engine underneath (llama.cpp) is written to run these efficiently on CPUs and Apple Silicon, which is exactly why GGUF + a Mac is a workable combination at all.

The pipeline relevance: your finetune comes out of Colab as PyTorch weights + a LoRA adapter. To serve it locally you'll **merge the adapter into the base, then convert that merged model to a quantized GGUF.** GGUF is the bridge between "trained in PyTorch on a cloud GPU" and "running in Ollama on my laptop."

---

## Ollama: the tool that hides the plumbing

You *could* run llama.cpp directly, juggling GGUF files and command-line flags. **Ollama** is a wrapper that makes local models feel as easy as an API. It does three jobs:

1. **Pulls models** from a registry, like Docker pulls images. `ollama pull llama3.2` fetches an already-quantized GGUF and stores it.
2. **Serves models** behind a local HTTP API (on `localhost:11434`) that mimics the shape of cloud APIs — so your application code barely changes between a cloud model and a local one.
3. **Manages models** — loading into memory on demand, unloading when idle, swapping between models.

### The everyday commands

```bash
ollama pull llama3.2          # download a model
ollama run llama3.2           # chat with it in the terminal
ollama list                   # see what's installed
ollama ps                     # see what's loaded in memory right now
ollama serve                  # run the API server (usually auto-started)
```

And the API call your *application* would make — note how close it is to a cloud API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Summarize the risk factors section.",
  "stream": false
}'
```

That `localhost:11434` endpoint is the whole reason Ollama matters for your project: your Filing Copilot code can point at it instead of the Anthropic API and run entirely locally, no other changes needed.

### The Modelfile — how your finetune becomes an Ollama model

This is the piece that ties the module together. A **Modelfile** is a small recipe (the Dockerfile of Ollama) that tells Ollama how to assemble a custom model — including one built from *your own* GGUF. A minimal one for your finetuned Filing Copilot model:

```dockerfile
# Modelfile
FROM ./filing-copilot-merged-Q4_K_M.gguf

# the model's default behavior, baked in
SYSTEM """You answer questions about SEC filings. Every claim must
cite the filing section it came from. If the retrieved context does
not support an answer, say so plainly rather than guessing."""

# generation settings
PARAMETER temperature 0.2
PARAMETER num_ctx 8192
```

Then:

```bash
ollama create filing-copilot -f Modelfile   # register your custom model
ollama run filing-copilot                    # use it like any other
```

Three things the Modelfile controls, each worth understanding:

- **`FROM`** — the base. Either a registry model name *or a path to your own GGUF* (the latter is your project case).
- **`SYSTEM`** — a default system prompt baked into the model, so the behavior travels with it. Useful, but note the lesson from the agents module: the system prompt should drive *behavior*, and shouldn't be relied on to *self-certify* groundedness — your actual citation-checking still lives in your application logic, not here.
- **`PARAMETER`** — defaults like temperature (low for a factual citation task) and context window size.

---

## The Mac memory reality (and when to give up and use Colab)

A blunt summary of what fits, so you set expectations correctly:

- A 4-bit **small model (≈3B params, ~2–4GB)** runs comfortably on a typical Mac and is your project's serving target.
- A 4-bit **7–8B model (~4–5GB)** runs but leaves less headroom; fine on 16GB+ machines, sluggish on less.
- Anything **larger** either swaps painfully or won't load. MoE models (last file) are deceptive here — their *active* compute is small but **all experts must be in memory**, so a Mixtral-class model needs far more RAM than its speed suggests.

The clean division of labor your project uses:

- **Colab for training.** Finetuning needs the gradient + optimizer memory (the 3× tax from the LoRA file) plus a real GPU. Even with QLoRA, this is a cloud-GPU job. Free Colab is sized for exactly this.
- **Mac for serving.** Inference on a merged, 4-bit-quantized small model is light enough to run locally. This is where Ollama lives.

Train heavy in the cloud, serve light on the laptop. That split is the practical shape of nearly every "local models" workflow, and it's why your project is staged the way it is.

---

## Tying it to the project

The final stretch of the module project is exactly this file in action:

1. **Merge** your QLoRA adapter into the base model (from the LoRA file: `W + (alpha/r)·B·A`).
2. **Convert** the merged model to GGUF and **quantize** to `Q4_K_M`.
3. **Write a Modelfile** with `FROM ./your.gguf`, a citation-enforcing `SYSTEM` prompt, and a low `temperature`.
4. **`ollama create`** it, then point your Filing Copilot code at `localhost:11434`.
5. **Run the before/after comparison** — base model vs. your finetuned local model on the held-out eval set — closing the loop the whole module was building toward.

By the end you'll have done the complete circuit: understood *when* to finetune, *how* LoRA/QLoRA make it affordable, *where* it sits among PEFT methods, and *how* to get the result running on your own machine. That's a finetuning workflow from first principles, end to end.

---

## Tools & references

- **Ollama** — `ollama.com`; the model library and Modelfile reference live in its docs.
- **llama.cpp** — the inference engine under Ollama; the GGUF format and the conversion/quantization scripts (`convert_hf_to_gguf.py`, `llama-quantize`) come from here.
- **GGUF** — format spec lives in the llama.cpp repo; worth a skim to see what's bundled in the single file.
- No paper for this file — it's the practical, build-it layer. The understanding comes from running the commands, not reading theory.
