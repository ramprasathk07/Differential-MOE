# Current state and what we can infer

**Snapshot: 2026-08-01.** The 2×2 is complete. Every cell ran 2,900 steps; every evaluation below was recomputed against the final checkpoints under one identical protocol.

Companion documents: [`02-future-scope.md`](02-future-scope.md) — what to do next · [`03-full-2x2-results.md`](03-full-2x2-results.md) — the detailed results and the transferable learnings.

---

## 1. Status of the grid

| Cell | Attention | FFN | Status | Steps | Best val NLL | Test NLL |
|---|---|---|---|---|---|---|
| **A · Dense** | standard | dense | ✅ complete | 2900 / 2900 | 3.6397 | 3.0517 |
| **B · Diff-Dense** | differential | dense | ✅ complete | 2900 / 2900 | 3.6103 | 3.0238 |
| **C · MoE** | standard | MoE top-2 | ✅ complete | 2900 / 2900 | 3.5990 | **2.9226** |
| **D · Diff-MoE** | differential | MoE top-2 | ✅ complete (rerun) | 2900 / 2900 | **3.5822** | 2.9633 |

Cell D's first session died to a Kaggle timeout at step 2500. A rerun with the same seed and config completed all 2,900 steps and tracked the first run within ±0.007 nats at every shared checkpoint. **Every number in this document is from the completed run.** The rerun is a continuation, not a seed replicate — it does nothing to close the variance gap in §5.

---

## 2. What was actually run

Shared, byte-identical across all four runs (verified against W&B configs, not assumed):

```
seed 42 (init + data order) · lr 2e-4 cosine to 10% · warmup 60
AdamW β(0.9, 0.95) · weight decay 0.1 · grad clip 1.0 · fp16 AMP · DDP on 2×T4
batch 2 × accum 32 × 2 GPUs × seq 512 = 65,536 tokens per optimizer step
2,900 steps = 190.1M tokens ≈ 1.13 epochs of BabyLM-strict
aux_loss_coef 0.01 · router_z_coef 0.001 · eval every 250 steps
```

Trained geometry — **reconstructed from checkpoint weight shapes**, because the Kaggle notebook rewrote configs at runtime and no repo YAML matches what ran:

```
vocab 100352 (cl100k padded) · dim 768 · 14 layers · 12 heads · seq_len 512
MoE cells: n_dense_layers 2 · 6 experts · top-2 · expert_inter 1536 (= 3072 / 2)
```

**Parity, enforced in code and guarded by tests:**
- *Attention parity* — differential uses half the heads at double width; both cost exactly `4·dim²`/layer.
- *Active-parameter parity* — `expert_inter = dense_inter / top_k`, so the two routed experts sum exactly to the dense FFN they replace.

| Run | Raw params | Active params | Non-embed active |
|---|---|---|---|
| Dense | 209.21M | 209.21M | 132.14M |
| Diff-Dense | 209.24M | 209.24M | 132.17M |
| MoE | 379.14M | 209.27M | 132.20M |
| Diff-MoE | 379.16M | 209.29M | 132.22M |

Spread across all four: **0.08M active parameters (0.04%)**.

---

## 3. What was measured

### 3.1 Held-out test — full six-domain split

3,200 windows (1.64M tokens) spread evenly across the 16.1M-token test split. **Byte-identical windows for every model**, fp32, single RTX 3060.

| Run | Test NLL | PPL | bits/byte | Top-1 |
|---|---|---|---|---|
| Dense | 3.0517 | 21.15 | 1.1306 | 0.4621 |
| Diff-Dense | 3.0238 | 20.57 | 1.1203 | 0.4661 |
| **MoE** | **2.9226** | **18.59** | **1.0828** | 0.4703 |
| Diff-MoE | 2.9633 | 19.36 | 1.0979 | **0.4706** |

### 3.2 Measurement uncertainty — two resampling units, two answers

20,000 resamples over the 3,200 identical windows. The **clustered** column resamples the six *domains* rather than individual windows.

| Comparison | Δ test NLL | iid 95% CI | domain-clustered 95% CI | SE | Windows won |
|---|---|---|---|---|---|
| MoE − Dense | −0.1291 | [−0.1339, −0.1245] | [−0.1419, −0.0512] | 0.0024 | **96.0%** |
| Diff-Dense − Dense | −0.0279 | [−0.0300, −0.0258] | [−0.0437, −0.0156] | 0.0011 | 69.2% |
| Diff-MoE − Dense | −0.0884 | [−0.0912, −0.0857] | [−0.1133, −0.0570] | 0.0014 | 95.1% |
| Diff-MoE − Diff-Dense | −0.0605 | [−0.0626, −0.0584] | [−0.0722, −0.0380] | 0.0011 | 91.0% |
| Diff-MoE − MoE | +0.0407 | [+0.0366, +0.0449] | [**−0.0160**, +0.0550] | 0.0021 | 47.7% |

iid intervals sit **19–63σ** from zero. Clustering widens every interval **6.5–10.3×**, and flips exactly one conclusion: **Diff-MoE − MoE crosses zero.** See §4.4.

### 3.3 Validation, step for step (all four cells, step 2900)

| | Dense | Diff-Dense | MoE | Diff-MoE |
|---|---|---|---|---|
| **val NLL @ 2900** | 3.6397 | 3.6103 | 3.5990 | **3.5822** |
| **vs dense** | — | −0.0293 | −0.0407 | −0.0575 |
| **Δ at step 250** | — | +0.0722 | −0.0409 | +0.0383 |
| **crossover step** | — | ~1000 | never behind | ~1250 (vs MoE) |

Both differential runs start *behind* and cross over: Diff-Dense at ~step 1000, Diff-MoE (against MoE) at ~step 1250. Neither curve ever turned upward.

### 3.4 Per-domain test NLL

| Domain | Dense | Diff-Dense | MoE | Diff-MoE | MoE − Dense | Diff-MoE − MoE |
|---|---|---|---|---|---|---|
| childes | 2.1045 | 2.0903 | **1.9042** | 2.0267 | **−0.2004** | **+0.1226** |
| switchboard | 2.3936 | 2.3882 | 2.3671 | **2.3633** | −0.0265 | −0.0038 |
| open_subtitles | 3.5770 | 3.5391 | 3.5027 | **3.4925** | −0.0743 | −0.0102 |
| simple_wiki | 3.6938 | 3.6354 | 3.5689 | **3.5469** | −0.1249 | −0.0220 |
| bnc_spoken | 3.8080 | 3.7855 | 3.7639 | **3.7542** | −0.0441 | −0.0097 |
| gutenberg | 3.8159 | 3.8006 | 3.7414 | **3.7375** | −0.0745 | −0.0039 |

**Diff-MoE beats plain MoE on five of six domains and still loses overall**, because CHILDES is 40% of the test split. The two mechanisms' per-domain gain profiles correlate at only **r = +0.21** — they fix different text.

### 3.5 Cost

| Run | tok/s (2×T4) | vs Dense | Hours for 2,900 steps | Steps in 7.6 h | Val NLL there |
|---|---|---|---|---|---|
| Dense | 6,944 | 1.00× | 7.6 h | 2,900 | 3.640 |
| **Diff-Dense** | 6,674 | 0.96× | 7.9 h | 2,787 | **3.613** |
| MoE | 5,250 | 0.76× | 10.1 h | 2,192 | 3.636 |
| Diff-MoE | 5,139 | 0.74× | 10.3 h | 2,146 | 3.620 |

Differential attention costs **2–4%** throughput, not the 27% measured in the first (contended) Diff-MoE session. **Diff-Dense is the best use of a fixed GPU-hour budget of all four cells.** See §5.7 — this verdict reversed once, and that is itself the lesson.

### 3.6 Routing health and specialization

| Run | Routing entropy (min / mean) | Load imbalance (max) | Domain specialization (max TV from uniform) |
|---|---|---|---|
| MoE | 0.9996 / 0.9999 | 1.066 | 0.052 |
| Diff-MoE | 0.9994 / 0.9998 | 1.101 | 0.044 |

Load-balance loss sat at **12.05** against a floor of 12.0 from ~step 100 onward. No collapse. **No domain routing preference in either model.**

### 3.7 Mechanism — measured directly, not inferred

| Model | Effective support | Top-8 mass | Mean distance | Negative attention mass |
|---|---|---|---|---|
| Dense | 86.2 | 0.380 | 90.0 | 0 |
| Diff-Dense | 70.8 (−17.8%) | 0.428 | 78.6 | **31.3%** |
| MoE | 75.6 (−12.4%) | 0.419 | 82.0 | 0 |
| Diff-MoE | **50.2** (−41.7%) | 0.503 | 64.6 | **32.4%** |

Learned λ, both differential runs:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | 0.20 | 0.36 | 0.47 | 0.56 | 0.62 | 0.67 | 0.70 | 0.73 | 0.75 | 0.76 | 0.77 | 0.78 | 0.78 | 0.79 |
| Diff-Dense | 0.11 | 0.54 | 0.57 | 0.59 | 0.69 | 0.77 | 0.74 | 0.70 | 0.77 | 0.78 | 0.72 | 0.75 | 0.79 | 0.74 |
| Diff-MoE | 0.16 | 0.43 | 0.67 | 0.59 | 0.62 | 0.65 | 0.70 | 0.74 | 0.81 | 0.75 | 0.83 | 0.84 | 0.74 | 0.77 |

The two runs converged on nearly the same λ profile (**r = 0.91**) despite entirely different feed-forwards. λ controls negative mass at **r = +0.975 / +0.993**.

### 3.8 BLiMP — construct validity closed

| | Dense | Diff-Dense | MoE | Diff-MoE |
|---|---|---|---|---|
| **overall** | 70.50% | 70.62% | **71.39%** | 71.14% |
| semantics | 63.90% | **67.03%** | 64.21% | **67.40%** |
| syntax | 66.42% | 65.07% | 66.65% | 66.94% |
| morphology | 82.59% | 82.68% | 82.86% | 82.50% |

The perplexity ranking transfers exactly (r = −0.99). **Differential attention's overall +0.1pp is noise; its semantics effect is not, and it replicated:** +3.13pp on a dense feed-forward, +3.19pp on an MoE, with both standard-attention baselines near 64%.

---

## 4. What we can infer — ranked by confidence

### 4.1 High confidence — "MoE beats dense at matched active compute"

−0.041 nats on validation at 12 of 12 checkpoints, **−0.129 on the full test split**, better in **all six domains**, **96% of individual windows**, and it survives domain-clustered resampling. It costs 24% throughput and 1.8× checkpoint size.

Framing: **MoE converts memory into quality at fixed FLOPs/token.** This is the cleanest signal in the grid. Still n=1, but the margin is 6–13× plausible seed noise and it is consistent across every slicing tried.

### 4.2 High confidence — "differential attention is nearly free"

2–4% throughput across both feed-forwards, measured on healthy sessions. **Diff-Dense reaches the lowest val NLL of any cell inside the dense run's own 7.6 GPU-hours.** This rests on a hardware ratio measured across thousands of logged steps, not on a small NLL difference.

### 4.3 Probable, and the most interesting claim in the project — "differential attention buys long-range referential binding, not general LM quality"

Three independent measurement modalities agree, and one of them replicates across feed-forwards:

| Modality | Evidence |
|---|---|
| **Per-position** | advantage ≈ 0 at position 0, grows monotonically to −0.048 by position 511; **r = −0.88** with position. MoE, a capacity mechanism, shows r = −0.40 and is near its full advantage by position 32. |
| **Per-domain** | gain concentrated on simple_wiki (2.28× its own mean) and open_subtitles (1.48×) — topic-switching, referentially dense text. Least on switchboard (0.21×) and childes (0.56×). |
| **Per-linguistic-field** | BLiMP semantics **+3.13pp** (dense) and **+3.19pp** (MoE), against ~0 overall. Two models sharing nothing but their attention agree to 0.06pp. |

And the causal chain is closed link by link: λ learned per layer → controls negative attention mass (r ≈ 0.98) → attention sharpens (−18% effective support) → advantage grows with context position (r = −0.88).

**This is the project's strongest scientific content and its best publication claim.** It is also the claim most exposed by §5.2 and §5.3.

### 4.4 Not established — "plain MoE beats Diff-MoE"

Point estimate +0.041 nats, iid interval 19σ from zero — and it **crosses zero under domain-clustered resampling** ([−0.016, +0.055]), with a **47.7%** window win rate. The entire gap is CHILDES, 40% of the test split.

The honest statement: Diff-MoE loses **on this corpus mixture**, because this corpus is 40% child-directed speech and that is precisely the text where a long-range mechanism has nothing to offer. Reweight the domains and the sign is not safe. This is a claim about the benchmark, not about the architecture.

### 4.5 Established — the mechanisms are sub-additive, and one domain explains it

Stacking recovers **82%** of the two individual gains on validation but only **56%** on held-out test. The gap between those two views is itself the finding: validation is BNC-only, and the entire test shortfall lives in CHILDES — a domain validation never sees.

### 4.6 Established as a property of the setup, not the architecture — no expert specialization

Six domains, six experts, max TV from uniform 0.052. The load-balance loss did exactly what it was told; balance and specialization pull against each other and balance won outright. `aux_loss_coef` sets that trade directly and **has never been swept**.

---

## 5. Confounds and limitations

### 5.1 n = 1 per cell — still the binding constraint

Every run is `seed=42`, once. Sampling error is solved to 19–63σ; **seed variance is unmeasured.** The differential-attention effects (−0.028 test, +3.1pp semantics) sit exactly in the range where seed noise could plausibly explain them. No amount of further analysis on these four checkpoints changes this.

### 5.2 A long-context mechanism tested at 512 tokens

§4.3's central claim is that differential attention pays off with context length — and the measured slope is **still descending at position 500 with no flattening.** The project therefore tested the mechanism in the regime where it should help *least*. This is the largest external-validity gap, and it is now a *falsifiable prediction* rather than just a caveat.

### 5.3 The head-count confound

Parity buys λ by halving heads, so "differential attention" here means **"the subtraction, minus half the attention patterns."** Every result in §4.3 is attributable to either. One deliberately parity-breaking run separates them. This is the first thing a reviewer will attack and it costs one config change.

### 5.4 One learning rate for four architectures

`lr 2e-4` throughout. λ has entirely different gradient scales from the projections around it and never got its own schedule — so part of the early differential penalty may be an optimizer artifact rather than an architectural cost.

### 5.5 The validation slice was one domain

`eval_batches` reads contiguously from offset 0 — the first 51,200 tokens of `val.bin`, which is entirely `bnc_spoken`. All four runs read identical windows so the comparison stayed fair, but checkpoint selection saw one domain, and BNC is where MoE helps *least* (−0.044 vs −0.129 overall).

### 5.6 Undertrained on purpose

~1 token per active parameter, ~15× below Chinchilla-optimal. That is BabyLM's premise, not an oversight — but no run reached its own ceiling and **none ever overfit.** These are early-training rankings and need not hold at 5 epochs.

### 5.7 A verdict already reversed once

Part 1 concluded differential attention costs 27% throughput and does not pay for itself. That rested on a single measurement from a contended Kaggle session; a rerun at identical memory returned 5,139 tok/s and the conclusion inverted. **A cost measured once is a property of that setup.** Treat every single-measurement claim in this document accordingly.

### 5.8 Other standing limitations

- **One data order.** Seed 42 fixes both init and data order.
- **Single scale** (209M active). Nothing here licenses claims at 7B.
- **GLUE-style probes not run.** BLiMP is done; the rest of BabyLM's suite is not.

---

## 6. Grading

| Axis | Grade | Basis | Change since 07-25 |
|---|---|---|---|
| Internal validity | **A−** | byte-identical configs, parity enforced in code *and* tests, identical eval windows, geometry reconstructed from weights | — |
| Measurement precision | **A** | 1.64M test tokens, paired design, bootstrap CIs under two resampling units | ↑ clustered CIs added |
| Statistical inference | **D** | n=1 per cell, no variance estimate | — |
| External validity | **C** | one scale, one corpus, one data order, one LR, 512 context | — |
| Construct validity | **B−** | BLiMP closed; GLUE-style probes still absent | ↑ from C |
| Mechanism evidence | **A−** | three converging modalities, one replicated across feed-forwards, causal chain closed link by link | ↑ new axis |

**Overall: a well-built instrument that has now taken good readings, on one sample.** The mechanism evidence is genuinely strong — stronger than most architecture papers offer. The statistical inference is still gated on a single seed.

**"Small model / small dataset" is not the weakness.** Small scale limits *external* validity, not internal validity. The weakness is **n=1**, which would be equally fatal at 7B.

---

## 7. Artifacts

| Path | What |
|---|---|
| `scripts/eval_all.py` | all checkpoints, one identical protocol, per-domain + routing + generations |
| `scripts/eval_uncertainty.py` | per-window NLLs + paired bootstrap CIs |
| `scripts/eval_bootstrap_clustered.py` | domain-clustered CIs (CPU only) |
| `scripts/eval_deep.py` | NLL by position, calibration, frequency deciles, truncation probe |
| `scripts/eval_attention.py` | attention matrices recomputed: entropy, negative mass |
| `scripts/eval_blimp.py` | BLiMP, 67 paradigms |
| `scripts/make_figures_part1.py` | every figure in the README and both blogs |
| `docs/blog/runs_export/*.json` | every number in this document |
| `docs/blog/runs_export/per_window_nll.npz` | raw per-window NLLs — re-analysis needs no GPU |
| `docs/blog/part1-*.md`, `part2-*.md` | the two narrative writeups |
| `README.md` | results-first project overview |

Reproduce end to end (GPU needed for the first, fourth, fifth, sixth):

```bash
python scripts/eval_all.py                  # ~35 min, RTX 3060
python scripts/eval_uncertainty.py          # ~12 min, writes CIs + cached per-window NLLs
python scripts/eval_bootstrap_clustered.py  # CPU only, seconds
python scripts/eval_deep.py                 # ~25 min — memory-heavy, see note
python scripts/eval_attention.py            # ~4 min
python scripts/eval_blimp.py                # ~13 min, downloads nyu-mll/blimp
python scripts/make_figures_part1.py
```

> `eval_deep.py` materialises `logits`, `logp`, `p`, and `ent` at (16, 512, 100352) fp32 — ~3.3 GB each. On a 12 GB card it runs at ~6 min/model. Chunking the entropy computation over the vocabulary axis would cut this substantially and is the obvious optimisation if this script gets rerun often.

**⚠️ `docs/` and `PLAN_AHEAD/` are in `.gitignore`.** Both blog posts, all eval JSONs, and these planning documents are untracked — they exist on disk only. Fix before treating any of it as published.
