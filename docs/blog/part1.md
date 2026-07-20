# Differential-MoE, Part 1 — First Light from the Miniature

> **Build log · Part 1 of 2**
> Does differential attention help a Mixture-of-Experts language model — and does it still help once you stack the two? Before spending real compute, I built the smallest honest version of the experiment and turned it on.

| | |
|---|---|
| **Project** | Differential-MoE |
| **Stage** | miniature / instrument check |
| **Hardware** | Kaggle 2×T4 |
| **Data** | BabyLM strict-small (~13M tokens) |
| **Model** | 15.7M active params · 4,096 vocab · 8 layers |

---

## The question

Modern language models borrow tricks constantly, and papers tend to arrive with the trick already declared a winner. This project takes two of them and asks a narrower, more falsifiable question: at a scale small enough to run for the cost of electricity, does each one actually earn its place — alone, and together?

The honest way to answer that is a controlled comparison, not a single big model with everything switched on. So the whole project is built around a **2×2 grid**: standard attention vs. differential attention, crossed with a dense feed-forward vs. a Mixture-of-Experts feed-forward. Four models, everything else held identical — same tokenizer, same data in the same order, same schedule, same seed.

**Idea 01 — Differential attention.** Standard attention computes one softmax map over the sequence, and some of its weight lands on irrelevant tokens — call it attention noise. Differential attention computes *two* maps and subtracts one from the other, so shared noise cancels and the signal is what survives. Noise-cancelling headphones for context.

**Idea 02 — Mixture of Experts.** Instead of one feed-forward block per layer, keep a bank of them — the *experts* — and a small router that sends each token to just two. The model holds far more parameters than it spends on any single token, so capacity grows without the compute-per-token growing with it.

---

## The discipline: the comparison is only worth anything if it's fair

A bigger model beating a smaller one tells you nothing. The entire value of this experiment is in what's held constant, so two rules are enforced in code and checked by tests:

- **Attention parity.** Differential attention uses half the heads at double the width, so both attention variants cost exactly the same parameters per layer — the difference is the mechanism, not the budget.
- **Active-parameter parity.** Each expert is sized so the two the router picks add up to exactly the dense feed-forward. The MoE model stores more, but spends the same per token. Every comparison is equal-compute by construction.

And because the tokenizer is deliberately tiny, every result is also reported in **bits-per-byte** — a measure that doesn't care which tokenizer you used, so the numbers stay honest when the scaled-up run switches to a bigger vocabulary.

---

## First light: 15.7M parameters on a laptop-sized corpus

Before committing a week of GPU quota to a 300M-parameter run, the point of the miniature is unglamorous: prove every wire is connected. Does the model train? Does distributed training work? Do the two mechanisms actually do what the papers say — and does the instrumentation catch it when they don't?

The miniature is ~15.7M active parameters, an 8-layer decoder with a 4,096-token vocabulary trained on [BabyLM](https://babylm.github.io/) strict-small — about 13 million tokens of child-directed speech, dialogue, and simple prose. It ran on two Kaggle T4s. Here is what came back.

### The pipeline runs — and runs fast

Both configurations trained cleanly across two GPUs, the baseline at **108,000 tokens per second**, using under a gigabyte of the 16 GB each card offers. Training loss falls the way it should.

> **Fig. 01 — Training loss (wandb: `train/loss`).** Cross-entropy in nats vs. optimizer step, both runs. Baseline falls 4.09 → 1.80 over its 1,000 steps before the Kaggle session ended; differential+MoE falls 7.91 → 1.07 across 5,600+ steps. Falling training loss is necessary, not sufficient — the story is where *validation* goes.

---

## Both ideas are demonstrably doing their job

A training loss that goes down doesn't prove differential attention is *differencing* or that the experts are *specializing*. Those need their own instruments — and both read clean.

### Differential attention learns a per-layer balance

The subtraction is scaled by a learnable value, λ, one per layer, initialized on a gentle schedule that grows with depth. If the mechanism were inert, λ would sit where it started. It doesn't:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **init** | 0.20 | 0.36 | 0.47 | 0.56 | 0.62 | 0.67 | 0.70 | 0.73 |
| **learned** | 0.21 | 0.52 | 0.69 | 0.63 | 0.70 | 0.77 | 0.72 | 0.76 |

> **Fig. 02 — Learned λ by depth vs. initialization (wandb: `train/lambda_mean_L*`).** The model didn't keep its initialization — it moved every layer while preserving the increasing-with-depth shape, strongest in the middle stack. Concrete evidence the differential mechanism is live and being tuned, not decorative.

### The experts stay balanced — no collapse

The classic failure of Mixture-of-Experts is collapse: the router discovers one expert, sends everything there, and the rest atrophy. The original version of this repository had no defense against it. The rebuild adds a load-balancing loss, and the routing entropy confirms it works:

| MoE layer | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|
| **norm. entropy** | 0.969 | 0.990 | 0.988 | 0.982 | 0.998 | 0.980 |

> **Fig. 03 — Routing entropy per MoE layer (wandb: `val/expert_entropy_L*`).** Normalized entropy of the expert assignment over validation, layers 2–7 (first two layers dense). Every value sits between 0.97 and 1.00, where 1.0 is a perfectly even split across all eight experts. The load-balancing loss holds; nothing collapsed.

---

## The finding: the instrument caught something — the model memorized

Here is where the miniature earned its keep. The differential+MoE run's training loss kept dropping beautifully, all the way to a perplexity under 3. Its validation loss did the opposite.

| Step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **val NLL** | 3.98 | **3.68** | 3.69 | 3.81 | 4.00 | 4.22 | 4.47 | 4.71 | 4.98 | 5.25 | 5.46 |

Validation reached its best point early, around step 1,000, and then climbed — steadily, for another 4,500 steps — while training loss kept falling to 1.07. That gap is the signature of a model memorizing its training set rather than learning the language. On a 13-million-token corpus, a schedule of 7,000 steps is roughly **70 passes over the same data**. Far too many.

> **Fig. 04 — Validation NLL, the overfit in one picture (wandb: `val/nll`).** The differential+MoE run bottoms out near step 1,000, then rises for the rest of the run — its usable checkpoint is that early minimum, not the final state. The baseline was still descending when its session was cut off.

**Why this is a good outcome for a Part 1.** The miniature was never meant to answer "does differential attention win." It was meant to prove the experiment is trustworthy — and it did, by flagging its own broken run instead of quietly reporting a memorized training score as success. The over-epoching here is the exact failure the scaled-up run is now built to avoid: a throughput probe measures real tokens-per-second and sets the step budget from the corpus size, not from a habit.

---

## Reading the runs honestly: what these two are — and aren't

It would be easy, and wrong, to put the baseline's 1.87 next to the differential+MoE's final 5.46 and declare a winner. One run stopped early; the other ran long enough to overfit. They saw different amounts of data. This is a status report on the instrument, not a verdict on the architecture.

| Run | State | Best val NLL | Best val PPL | Tokens seen | Reading |
|---|---|---|---|---|---|
| standard + dense | cut short | 1.87 | 6.46 | 131M | Healthy, still improving at cutoff |
| differential + MoE | overfit | 3.68 | 39.7 | 742M | Best at step ~1,000, memorized after |

**Verified working:** DDP across 2×T4 · fp16 + grad-scaler · checkpoint/resume · per-layer λ logging · expert entropy & imbalance · aux + router-z loss · bits-per-byte · held-out test split.

---

## Next — Part 2: scaling up, with the budget set by the corpus

Part 2 runs the same 2×2, but grown: ~295M active parameters, a frontier cl100k tokenizer in place of the tiny custom one, and the full BabyLM *strict* track — 168 million tokens, measured, not estimated. This time the step count comes from a throughput probe rather than a guess, so no run over-epochs itself into memorizing. That's when the four models finally become comparable — and when the actual question gets an answer.

---

*Differential-MoE — build log · Part 1 · the miniature · metrics logged to Weights & Biases · differential attention: Ye et al. 2024 · [BabyLM Challenge](https://babylm.github.io/)*
