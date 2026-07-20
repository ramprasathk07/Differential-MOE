# Diff-MoE Rebuild Plan — BabyLM @ Kaggle

Goal: resurrect this repo as a **small, trainable, measurable** research project:
*"Does Differential Attention help a Mixture-of-Experts LM at 16M–300M scale?"*
Train on the BabyLM Challenge corpus inside Kaggle's free GPU quota, report
NLL / perplexity / bits-per-byte / expert-utilization, publish results + blog posts.

Three tiers, same 2×2 ablation each: **A** ~16M (custom 4k BPE, strict-small),
**B** ~55M (custom 8k BPE), **S** ~295M active (cl100k frontier tokenizer,
strict). Each tier is a full week's Kaggle quota — run one at a time.

---

## 1. Current-state audit (what the rebuild must fix)

| # | Bug | Where | Effect |
|---|-----|-------|--------|
| 1 | `ModelArgs = load_model_args_from_yaml(...)` shadows the class with an instance at import time | `model/modelargs.py:102` | `ModelArgs()` crashes anywhere else |
| 2 | `@torch.inference_mode()` on `Transformer.forward` | `model/layers.py:769` | no gradients — training impossible |
| 3 | K/V written into `register_buffer` cache during training; attention reads the cache | `model/layers.py:417-466` | gradient never reaches `wk`/`wv` (silent) |
| 4 | Cache buffers sized `max_batch × max_seq × heads × 2·head_dim` at init | same | ~300 GB allocation with config.yaml values |
| 5 | tiktoken cl100k vocab (100,277) vs `vocab_size: 32000` in config | `training/data.py` + `config.yaml` | embedding index OOB on first batch |
| 6 | `from kernel import ...` absolute import + unconditional `tilelang` | `model/layers.py:9` | package broken; uninstallable on Kaggle/Windows |
| 7 | `CosineAnnealingWarmRestarts(T_0=1)` stepped per optimizer step, no warmup | `train.py:133` | LR restarts every step |
| 8 | `GradScaler(enabled=dtype=='bf16')` | `training/trainer.py:43` | scaler is fp16-only; T4 has no bf16 |
| 9 | `IterableDataset` + `num_workers=4`, no worker sharding | `training/data.py` | 4× duplicate data |
| 10 | No load-balance / z-loss for MoE router | everywhere | expert collapse guaranteed |
| 11 | Diff-attn projects 2×dim for Q,K,V,O with full head count (8d² params vs paper's 4d²) | `model/layers.py:396-399` | unfair vs standard attention baselines |
| 12 | Three disagreeing configs (README 18-22B / config.yaml 6144×48 / dataclass 2560×24) | docs | no single source of truth |

Keep: diff-attention math (λ reparam, per-head RMSNorm, (1−λ_init) scaling),
DeepSeek-style gate/group routing logic, YaRN RoPE precompute, trainer
logging/checkpoint skeleton.

---

## 2. Target experiment (the scientific core)

2×2 ablation, everything else identical (tokens, tokenizer, schedule, seed):

| Run | Attention | FFN | Question answered |
|-----|-----------|-----|-------------------|
| A | standard SDPA | dense | baseline |
| B | differential | dense | does diff-attn help alone? |
| C | standard SDPA | MoE top-2 | does sparsity help alone? |
| D | differential | MoE top-2 | do they compose? (repo's thesis) |

Parity rules:
- **Attention parity**: diff-attn uses *half the heads* with 2× per-head width
  (paper's setting) → both attentions cost exactly 4d² params/layer.
- **FFN parity**: expert `inter_dim = dense_inter / top_k` → active FFN params
  identical to dense by construction; MoE just has more *total* params.
- Same tokenizer, same data order (seeded), same LR schedule, same token budget.

**Parameter accounting (used everywhere in this project)**
- **Raw (total) params**: every weight stored in the checkpoint — all experts
  included. Determines memory footprint, checkpoint size, optimizer-state size.
- **Active params**: weights actually used per token in one forward pass —
  attention + embeddings + norms + router + only the `top_k` selected experts
  (+ shared expert if any). Determines FLOPs/token, i.e. compute cost.
- Dense models: raw = active. MoE models: raw > active; the parity rule pins
  **active** equal to the dense baseline, so all comparisons are equal-compute.
  All README/blog tables must show both columns.

### Model tiers

**Tier A — ablation workhorse (~16M params, ~11M non-embedding active)**
```yaml
vocab_size: 4096        # custom BPE trained on BabyLM, tied embeddings
dim: 384
n_layers: 8
seq_len: 512
# attention (both variants cost 4·d² = 590K/layer)
standard: n_heads 6, head_dim 64
diff:     n_heads 3, head_dim 64, per-component 2×64
# dense FFN: SwiGLU inter 1024  (3·d·inter = 1.18M/layer)
# MoE layers (layers 2..7, first 2 dense):
n_experts: 8
top_k: 2
expert_inter: 512       # 2 × 512 active = 1024 → parity with dense
shared_expert: 0        # (+1 shared ablation optional)
aux_loss_coef: 0.01     # Switch-style load balance
router_z_coef: 0.001
```
Approximate counts (verify with `params.py`): dense variant ≈ **15.7M raw =
15.7M active**; MoE variant ≈ **37M raw / 15.7M active** (2.35× capacity at
equal compute).

**Tier B — headline run (~55M total dense-equivalent)**
```yaml
vocab_size: 8192
dim: 512
n_layers: 12
seq_len: 512
heads: 8 std / 4 diff (head_dim 64)
dense inter: 1536
moe: 16 experts, top-2, expert_inter 768, 1 shared
```
Approximate counts: ≈ **220M raw / 57M active** (fp16 weights + AdamW states
≈ 3.5 GB — fits T4 16 GB; full checkpoint ≈ 2.6 GB, keep only last + best on
Kaggle disk).

**Tier S — scale-up ablation (~295M active, frontier tokenizer)**
```yaml
vocab_size: 100352        # cl100k (100,263) padded to a multiple of 128
dim: 896
n_layers: 16
seq_len: 512
heads: 14 std / 7 diff (head_dim 64)
dense inter: 3584         # 4 x dim
moe: 8 experts, top-2, expert_inter 1792, 0 shared
batch 4 x accum 32 x 2 GPU = 256 seqs = 131k tok/step
```
Measured by `src/params.py`: **295.5M active** for all four configs (dense
295.47M, diff 295.50M — 0.01% apart; MoE **700.2M raw / 295.6M active**, i.e.
2.4× stored capacity at equal compute). Configs: `configs/s_{dense,diff,moe,diffmoe}.yaml`.

Why this shape rather than a wider/deeper one: with a 100k vocab the embedding
table is ~90M params on its own, so "300–400M total" is a narrower model than
it sounds. At a fixed 7.5h/run budget, dim 1024 × 24 layers (505M) would halve
the tokens each run sees — worse-trained models for the same wall-clock.

**Honest limitation to report, not hide.** At 7.5h/run on 2×T4 each tier-S run
sees ~380M tokens ⇒ ≈1.3 tokens/param, roughly GPT-3's ratio but ~15× below
Chinchilla-optimal (~20). Compute-optimal for that budget would be ~75M params
× 1.5B tokens. So tier S is deliberately over-parameterised against a fixed
corpus: expect train/val divergence, and **do not expect it to beat a
well-trained small model on fluency**. That is on-topic for BabyLM — whose
entire premise is a fixed, small data budget — but it must be stated plainly in
the writeup rather than glossed. If fluency is the goal, the lever is more
unique tokens (FineWeb-Edu), not more parameters.

Exact counts printed by `src/params.py` (write it first; README table comes
from its output, never hand-computed).

### Tokenizer (adaptive) & batching

**Adaptive tokenizer = data-driven vocab selection**, not a fixed guess:
1. Train candidate BPE vocabs on the BabyLM mix: 2k, 4k, 8k, 16k.
2. Score each on the held-out dev split: **fertility** (tokens/word — lower is
   better) and byte-compression ratio.
3. Pick the knee of the fertility-vs-vocab-size curve. Measured on BabyLM:
   2048 → 1.399, 4096 → 1.278, 8192 → 1.226, 16384 → 1.216 — the drop per
   doubling collapses after 4k (−8.7%, −4.1%, −0.8%), so **4096** for tier A.
   Zero-GPU step; the sweep table goes in the README + blog.
4. Cross-vocab fairness in reports comes from **bits-per-byte** (§3), which is
   tokenizer-independent — so the choice can't silently rig PPL comparisons.

**Pretrained frontier tokenizers (tier S).** `train_tokenizer.py --pretrained
hf:<id>` scores cl100k / Qwen / GPT-2 in the same sweep, so the choice rests on
the measured fertility gap on *this* corpus rather than on "bigger is better".
The tradeoff is embedding cost — vocab × dim params — which is why a frontier
vocab only makes sense once the model is wide: GPT-2's 50k vocab is 123% of a
16M model but 14.7% at dim 1024. Two mechanical consequences:
- Vocab > 65535 no longer fits `uint16`, so `prepare.py` switches to `uint32`
  (doubling `.bin` size) and records the dtype in `meta.json`; `load_tokens`
  reads it back. Guarded by `tests/test_token_storage.py`.
- The fp32 logits tensor is `batch × seq × vocab × 4B` — at vocab 100k it
  dominates T4 memory, which is why tier S uses micro-batch 4 with accum 32
  rather than the 16×8 used at tier A.
`train.py` refuses to start if a config's `vocab_size` can't cover the data's
max token id — the original repo's fatal bug (§1, row 5), now impossible.

**Batching = packed windows, zero padding (deliberate).** The token stream is
`story <|endoftext|> story <|endoftext|> ...` chunked into fixed 512-token
windows. Every position is a real token — 100% compute utilization. Dynamic
padding (pad-to-longest-in-batch) only makes sense when examples must stay
separate; for LM pretraining it wastes compute on pad tokens and adds
attention-mask plumbing. Not used.

Known cost of packing: attention can look across story boundaries within a
window. Standard practice (GPT-2 onward) ignores this; stories are ~175–300
tokens so most windows contain 1–2 boundaries. Optional **boundary-masking
ablation** (quota permitting): block-diagonal attention mask that resets at
`<|endoftext|>`, one tier-A run, measures whether boundary leakage moves val
NLL at all. Low priority — same leakage applies equally to all 4 runs, so the
2×2 comparison stays fair without it.

### Training recipe
- Optimizer: AdamW, lr 3e-4 (tier A) / 2.5e-4 (tier B) / 2e-4 (tier S),
  β=(0.9, 0.95), wd 0.1 on ≥2-D non-embedding weights only (router weight and
  λ params excluded).
- Schedule: linear warmup 2% of steps → cosine to 10% of peak. One cycle, no restarts.
- Precision: fp16 AMP + GradScaler on T4 (router, λ params, softmax/exp in fp32).
- Batch: tier A/B 16 × 512 × accum 8; tier S 4 × 512 × accum 32 (large-vocab
  logits memory). Both give 65K tok/step per GPU, ×2 under DDP. Grad-clip 1.0.
- **Set `max_steps` from the corpus, not by habit.** BabyLM is small:
  strict-small ≈ 13M tokens, strict ≈ 167.5M (measured under cl100k). At 131K tok/step on 2 GPUs, the
  inherited `max_steps: 7000` is 918M tokens = **72 epochs** of strict-small —
  deep in memorisation territory. Target ≈ 3–4 epochs:
  `max_steps = target_tokens ÷ (batch × seq × accum × n_gpu)`, and record the
  resulting epoch count in a config comment. The notebook computes this from
  the probe's measured tok/s.
- Checkpoint + resume every 30 min (Kaggle preemption survival), seed logged.

### Kaggle budget math (30 h GPU/week, 12 h max session, T4 ×2)
- Throughput probe first (phase 3): measure tok/s, then commit. Every number
  below is an *estimate at ~25 effective TFLOP/s* and must be replaced by the
  measured value before a full run.
- Tier A (~16M): ~2–4 h/run × 4 runs = 8–16 h.
- Tier S (~295M): 30 h ÷ 4 runs = **7.5 h/run** ⇒ ~380M tokens/run
  (≈2.3 epochs of strict's measured 167.5M tokens, ≈1.3 tok/param). Consumes a
  full week's quota.
- Do not run tier A and tier S in the same week; each is a full quota.
- Resume makes multi-session safe — but `/kaggle/working` is wiped between
  sessions, so save it as a Dataset and copy back (never train into
  `/kaggle/input`, which is read-only).

---

## 3. Metrics & evaluation protocol

The numbers the whole project is judged by. Every run reports the same set,
measured identically.

**Primary (all runs)** — computed on a **fixed validation slice**: first ~1.6M
tokens of `val.bin`, identical windows for every run and every eval step.
- **Validation NLL**: mean per-token cross-entropy, in nats/token.
- **Perplexity**: `PPL = exp(NLL)`.
- **Bits per token**: `NLL / ln(2)`.
- **Bits per byte**: total nats over the slice `/ (ln(2) × total UTF-8 bytes
  of the slice's text)` — tokenizer-independent, the fair cross-vocab number;
  mandatory in all reported tables since the vocab is chosen adaptively (§2).
- **Token top-1 accuracy** (secondary sanity signal).

**MoE-specific (runs C, D)**
- Expert utilization histogram per MoE layer (fraction of routed assignments).
- **Normalized routing entropy**: `H(f) / log(n_experts)` — 1.0 = perfectly
  uniform, → 0 = collapse. Tracked over training.
- **Load imbalance ratio**: `max_e f_e / mean_e f_e`.
- Aux-loss and z-loss curves.

**Diff-attn-specific (runs B, D)**
- Learned λ per layer over training (should drift from λ_init, stay in ~[0, 1]).

**Training health (all runs)**
- Grad-norm, GradScaler scale (fp16 overflow detector), tokens/sec, wall-clock,
  LR curve.

**Qualitative**
- 20 fixed prompts, greedy + temperature 0.8 / top-p 0.9, logged at every ckpt.
- Optional: LLM-judge rubric (grammar / creativity / consistency / plot, scored
  1–10, TinyStories-paper style rubric) — same judge model + prompt for all runs, run
  once at the end on final checkpoints.

**Reporting rules**
- Eval every 500 steps on the fixed slice; the README table uses the
  best-validation checkpoint per run.
- Final numbers quoted with ± spread over the last 3 evals.
- Every number traceable to (config file, seed, W&B run id, checkpoint hash).

---

## 4. Rebuild phases

### Phase 0 — Freeze the archaeology (30 min)
- Tag current state: `git tag v0-archive`. Never rebase it away — it's blog material.
- Delete dead weight on new branch `rebuild`: `model/generate.py` (DeepSeek verbatim),
  `model/kernel.py` (FP8/tilelang), FP8 paths in Linear, parallel-linear classes.

### Phase 1 — Clean package (1 day)
```
src/
  model/
    config.py        # dataclass, from_yaml (no import-time side effects)
    attention.py     # StandardAttention (F.scaled_dot_product_attention)
                     # DifferentialAttention (half heads, headwise RMSNorm,
                     #   λ in fp32, no cache in training path)
    moe.py           # Gate (topk + aux/z-loss returned), Expert, MoE (dropless loop)
    block.py         # pre-norm block, plain residual (drop fused-add-norm cleverness)
    transformer.py   # tied embeddings, causal via SDPA is_causal=True
  data/
    babylm.py            # fetch the 6 BabyLM domain files (train/dev/test)
    tokenizer.py         # one adapter over custom BPE / HF / tiktoken + dtype choice
    train_tokenizer.py   # BPE sweep on BabyLM; --pretrained scores frontier vocabs
    prepare.py           # tokenize once → uint16/uint32 memmap + meta.json
    dataset.py           # memmap random-window sampler — kills streaming/worker bugs
  train.py           # AMP fp16, accum, warmup+cosine, aux losses, resume, wandb/CSV
  eval.py            # val NLL (nats + bits/token), PPL, top-1 acc, sample generations
  params.py          # exact param counts per config
configs/             # a_dense.yaml, a_diff.yaml, a_moe.yaml, a_diffmoe.yaml, b_final.yaml
tests/               # see phase 2
notebooks/kaggle_train.ipynb
```

### Phase 2 — Tests before training (half day; these catch the old bugs)
1. Grad-flow: after one backward, **every** parameter has non-None, non-zero grad
   (catches inference_mode + cache-detach class of bugs).
2. Causality: perturb token t, logits at <t unchanged.
3. Diff-attn ≡ paper: λ_init schedule, output shape, fp32 λ.
4. Param parity: A/B and C/D active params within 1%.
5. Router: aux loss decreases imbalance on synthetic skewed input; all-expert
   utilization > 0 after few steps.
6. Overfit one batch to ~0 loss in <500 steps (classic sanity).
7. Resume test: train 100 steps, checkpoint, resume, bit-identical loss curve vs
   uninterrupted run.

### Phase 3 — Pipeline + probe on Kaggle (1 day)
- Tokenizer sweep (2k/4k/8k/16k) → fertility/compression table → lock vocab (§2, CPU-only).
- Upload memmap tokens as Kaggle Dataset (private).
- 30-min throughput probe per tier → lock real token budgets.
- Verify fp16 stability (watch GradScaler scale; if λ exp() overflows, clamp).

### Phase 4 — Ablation runs (1 Kaggle week)
- 4 × tier A runs, identical budget, wandb project public.
- Log: train/val NLL, PPL, LR, grad-norm, tok/s, expert histogram per MoE layer,
  router entropy, aux-loss, λ per layer over time (diff runs).

### Phase 5 — Headline + analysis (2nd week)
- Tier B with winning config + 1 seed rerun of the closest A-pair (variance bar).
- Analysis notebooks:
  - Expert specialization: top tokens per expert, entropy over training.
  - Attention-noise mini-experiment: prepend distractor story, measure attention
    mass on distractor (diff-transformer paper's claim, small-scale version).
  - Generation quality: fixed 20 prompts, greedy + t=0.8; optional LLM-judge
    rubric (grammar/creativity/consistency, TinyStories-paper style).

### Phase 6 — Packaging (2 days)
See §8.

---

## 5. Timeline & gates (~3 weeks part-time)

| When | What | Gate to proceed |
|------|------|-----------------|
| Week 1, days 1–3 | Phases 0–2: package + tests, CPU smoke run | all 7 tests green; 20-step smoke loss decreases |
| Week 1, days 4–5 | Phase 3: tokenizer, prepare, Kaggle dataset upload, throughput probe | measured tok/s ≥ half of estimate |
| Week 2 | Phase 4: four tier-A runs (≤ 4 h each) | val-NLL curves clean, expert entropy > 0.8 normalized |
| Week 3 | Phase 5: tier-B run + seed rerun + analysis | — |
| Week 3, end | Phase 6: README, W&B report, HF upload, blog drafts | — |

**Probe-fail contingency**: if tier A trains < 15K tok/s, shrink to dim 320 /
6 layers rather than cutting the token budget — full-epoch training is the
non-negotiable (undertrained comparisons are meaningless).

---

## 6. Risks & mitigations

| Risk | Early signal | Mitigation |
|------|--------------|------------|
| fp16 divergence (λ exp, router softmax) | GradScaler scale shrinking, NaN loss | λ + router already fp32; clamp λ dot-products to ±10; last resort: tier A in fp32 (~3× slower, still fits quota) |
| Expert collapse despite aux loss | normalized entropy → 0, one expert hot | raise aux coef 0.01 → 0.05; verify with test 5 before any GPU time |
| Kaggle preemption / 12 h limit | session killed mid-run | checkpoint every 30 min; runs sized ≤ 4 h; resume test (test 7) guarantees continuity |
| Quota exhaustion | > 24 h used before tier B | tier B slips to following week — ablation alone is already publishable |
| Null result (diff-attn no gain) | flat 2×2 table | still ship it: blog 2 framed as honest null; seed-rerun variance bar makes the claim defensible |
| Noisy validation (small slice) | val NLL jitter between evals | fixed 1.6M-token slice + report ± spread over last 3 evals |
| HF datasets download flaky on Kaggle | prepare step stalls | pre-tokenized `.bin` files uploaded as Kaggle Dataset — zero HF dependency at train time |

---

## 7. Open decisions (defaults chosen; revisit before Phase 1)

| Decision | Default | Trade-off |
|----------|---------|-----------|
| Tokenizer | adaptive: sweep 2k/4k/8k/16k BPE, pick by fertility knee (§2); expected ~4k tier A / ~8k tier B | params go to the body, cleaner story; PPL not comparable to GPT-2-tokenizer repros — bits-per-byte covers that |
| Batching | packed 512-token windows, no padding | max utilization; boundary leakage accepted (equal for all runs); optional boundary-mask ablation |
| Shared expert | 0 in tier A ablation, 1 in tier B | keeps 2×2 clean; shared-expert effect shown once at tier B |
| Sequence length | 512 | stories are ~200–300 tokens; 512 covers most, halves attention cost vs 1024 |
| GPUs | single T4 by default; DDP available via `--ddp` (launch with `torchrun --nproc_per_node=2`) | opt-in, not automatic -- adds failure modes (rank-gated I/O, MoE `find_unused_parameters`, per-rank data sharding) for ~1.7× speed. Validated locally via 2-rank CPU (`gloo`) runs including resume. |
| Experiment tracking | CSV always + W&B optional flag | runs never blocked on wandb login |

---

## 8. Presentation for profile

**GitHub (pin the repo)**
- README rewrite: results table up top (PPL/NLL per run, param counts, tokens,
  wall-clock, hardware), loss-curve PNG, architecture diagram (mermaid), honest
  limitations section, "bugs I fixed from v0" section linking `v0-archive` tag.
- Badges: wandb report, HF model, Kaggle notebook.
- `results/` with CSVs + plots; every number reproducible from configs + seeds.

**Weights & Biases**: public report — the 2×2 dashboard is the centerpiece.

**Hugging Face Hub**: upload tier-B checkpoint + model card (metrics, config,
sample generations); optional tiny Gradio Space (CPU inference fine at 50M).

**Kaggle**: publish the training notebook (clean, documented, one-click).

**LinkedIn/X**: one post per blog; lead with the 2×2 result figure.

**Resume / profile bullet (draft)**
> Designed and trained a 2×2 ablation of Differential Attention × Mixture-of-Experts
> language models (16–55M params) from scratch on free Kaggle GPUs; built
> parameter-parity methodology, load-balanced top-2 routing, and a fully
> reproducible eval harness (NLL/PPL/expert-utilization); published results,
> checkpoints, and a 3-part blog series.

---

## 9. Blog posts

### Blog 1 — "Resurrecting a dead repo: 12 bugs between me and a trainable MoE"
Audience: practitioners. Every bug here is one someone is hitting right now.
1. Hook: repo claims 18–22B params and FP8 kernels; zero training steps ever ran.
2. Audit method: read every file assuming guilt; three configs that disagree.
3. Deep-dive top 5: `inference_mode` on forward; KV-cache silently eating
   gradients; the 300 GB buffer allocation; tokenizer/vocab OOB; 4× data
   duplication from unsharded streaming workers.
4. Each bug: symptom → root cause → the test that now guards it.
5. Meta-lesson: aspiration-driven vs verification-driven repos; write the tests
   before the training loop.
6. CTA: the rebuild plan + links to posts 2 and 3.

### Blog 2 — "Does Differential Attention help at 20M parameters? A $0 ablation"
Audience: research-curious. The parity discipline is the differentiator.
1. Why small-scale ablations are underrated; BabyLM as a microscope.
2. Parity methodology (the real contribution): 4d² attention parity, active-param
   FFN parity, identical data order.
3. Setup + budget math: free T4, 30 h/week, quota arithmetic.
4. Results: 2×2 NLL/PPL table, loss curves, seed-variance bar.
5. Attention-noise mini-experiment (distractor-story attention mass).
6. Honest conclusion — positive or null — plus limitations (one scale, one
   dataset, one seed pair).

### Blog 3 — "What 8 experts learn from bedtime stories"
Audience: broad ML. Highly visual.
1. MoE at a scale you can actually inspect: 8 experts, 4k vocab.
2. Router mechanics in three paragraphs: top-2, load-balance aux, z-loss.
3. Expert specialization: top tokens per expert, per-layer tables.
4. Routing entropy over training; the collapse demo (aux coef = 0).
5. Shared-expert effect (tier B).
6. What this suggests — and doesn't — about frontier-scale MoEs.

(Optional 4th: "Training LLMs on Kaggle's free tier: quota math, memmaps, and
resume-or-die" — logistics post, evergreen SEO.)

---

## 10. Non-goals
- FP8 / TileLang / MI300X anything (needs H100/MI300; Kaggle has T4).
- MLA, YaRN long-context, distributed training.
- Beating any external benchmark — the deliverable is a controlled comparison.

---

## 11. Execution checklist

- [ ] Phase 0: `v0-archive` tag + `rebuild` branch; dead code removed
- [ ] Phase 1: `src/` package builds; `params.py` table matches §2 targets
- [ ] Phase 2: all 7 tests green on CPU; smoke train loss decreases
- [ ] Phase 3: tokenizer sweep done + vocab locked; `train.bin`/`val.bin` on Kaggle; probe done; budgets locked
- [ ] Phase 4: runs A–D complete at equal token budget; metrics logged per §3
- [ ] Phase 5: tier B + seed rerun; analysis notebooks (experts, attention-noise, generations)
- [ ] Phase 6: README results table; W&B public report; HF checkpoint + card; Kaggle notebook public
- [ ] Blogs: post 1 drafted from §1 audit; posts 2–3 after results
