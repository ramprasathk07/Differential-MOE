# Diff-MoE Rebuild Plan — TinyStories @ Kaggle

Goal: resurrect this repo as a **small, trainable, measurable** research project:
*"Does Differential Attention help a Mixture-of-Experts LM at ~15–50M scale?"*
Train on TinyStories inside Kaggle's free GPU quota, report NLL / perplexity /
expert-utilization, publish results + 2–3 blog posts.

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

### Model tiers

**Tier A — ablation workhorse (~16M params, ~11M non-embedding active)**
```yaml
vocab_size: 4096        # custom BPE trained on TinyStories, tied embeddings
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

Exact counts printed by `src/params.py` (write it first; README table comes
from its output, never hand-computed).

### Training recipe
- Optimizer: AdamW, lr 3e-4 (tier A) / 2.5e-4 (tier B), β=(0.9, 0.95), wd 0.1
  on ≥2-D non-embedding weights only.
- Schedule: linear warmup 2% of steps → cosine to 10% of peak. One cycle, no restarts.
- Precision: fp16 AMP + GradScaler on T4 (router, λ params, softmax/exp in fp32).
- Batch: 16 × 512 tokens, grad-accum 8 → 65K tokens/step, grad-clip 1.0.
- Token budget: 1 epoch of TinyStories ≈ **~460M tokens** (Chinchilla-ish for
  ~20M params; tier B gets same budget — note as limitation).
- Checkpoint + resume every 30 min (Kaggle preemption survival), seed logged.

### Kaggle budget math (30 h GPU/week, 12 h max session, T4)
- Throughput probe first (phase 3): measure tok/s, then commit.
- Estimates: tier A ~30–60K tok/s → 2–4 h/run × 4 runs = 8–16 h.
  Tier B ~15–25K tok/s → 5–8 h, single run (winner config from ablation).
- Total ≈ 20–24 h → fits one week's quota; resume makes multi-session safe.

---

## 3. Rebuild phases

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
                     # DifferentialAttention (half heads, GroupNorm per head,
                     #   λ in fp32, no cache in training path)
    moe.py           # Gate (topk + aux/z-loss returned), Expert, MoE (dropless loop)
    block.py         # pre-norm block, plain residual (drop fused-add-norm cleverness)
    transformer.py   # tied embeddings, causal via SDPA is_causal=True
  data/
    train_tokenizer.py   # BPE 4k/8k on TinyStories (HF tokenizers)
    prepare.py           # tokenize once → uint16 memmap train.bin/val.bin (~1 GB)
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
See §4.

---

## 4. Presentation for profile

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

---

## 5. Blog posts

1. **"Resurrecting a dead repo: 11 bugs between me and a trainable MoE"**
   The audit as narrative: `inference_mode` on forward, KV-cache eating
   gradients, the 300 GB buffer, tokenizer/vocab mismatch, worker duplication,
   LR sawtooth. Each bug = symptom → diagnosis → test that now guards it.
   Audience: practitioners; every bug is one someone is hitting right now.

2. **"Does Differential Attention help at 20M parameters? A $0 ablation"**
   The 2×2 study: setup, parity methodology, NLL/PPL results, attention-noise
   experiment, honest conclusion (positive or null — null is still a post).
   Audience: research-curious; the parity discipline is the differentiator.

3. **"What 8 experts learn from bedtime stories"**
   MoE internals at readable scale: expert specialization tables, router
   entropy curves, what happens with aux-loss off (collapse demo), shared-expert
   ablation. Highly visual.

(Optional 4th: "Training LLMs on Kaggle's free tier: quota math, memmaps, and
resume-or-die" — logistics post, evergreen SEO.)

---

## 6. Explicit non-goals
- FP8 / TileLang / MI300X anything (needs H100/MI300; Kaggle has T4).
- MLA, YaRN long-context, distributed training.
- Beating any external benchmark — the deliverable is a controlled comparison.
