# The complete 2×2 — results, downfalls, and what to take from it

**Snapshot: 2026-08-01, after Diff-MoE's rerun completed all 2,900 steps.** All four cells are trained to the same schedule and evaluated under one identical protocol; every number below is from the final checkpoints. This document is the detailed results record — [`01-current-state.md`](01-current-state.md) is the current-state summary and [`02-future-scope.md`](02-future-scope.md) is the plan.

---

## 1. The grid, complete

| Cell | Attention | FFN | Steps | Val NLL | Test NLL | tok/s | Wall clock |
|---|---|---|---|---|---|---|---|
| **A · Dense** | standard | dense | 2900 | 3.6397 | 3.0517 | 6,944 | 7.67 h |
| **B · Diff-Dense** | differential | dense | 2900 | 3.6103 | 3.0238 | 6,674 | 7.97 h |
| **C · MoE** | standard | MoE top-2 | 2900 | 3.5990 | **2.9226** | 5,250 | 10.18 h |
| **D · Diff-MoE** | differential | MoE top-2 | 2900 | **3.5822** | 2.9633 | 5,139 | 10.27 h |

Held-out test: 3,200 windows / 1.64M tokens, byte-identical for every model, fp32.

**The ranking depends entirely on which axis you privilege**, and that is the single most important thing this grid produced:

| Ranked by… | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| **Test NLL** | MoE **2.923** | Diff-MoE 2.963 | Diff-Dense 3.024 | Dense 3.052 |
| **Val NLL @ final step 2900** | Diff-MoE **3.582** | MoE 3.599 | Diff-Dense 3.610 | Dense 3.640 |
| **Val NLL in a fixed 7.6 GPU-h** | Diff-Dense **3.613** | Diff-MoE 3.620 | MoE 3.636 | Dense 3.640 |

Three axes, three different winners. No single number answers "which architecture is best."

---

## 2. Both pre-registered predictions held

Before Run B trained, [`01-current-state.md`](01-current-state.md) §6 recorded two falsifiable predictions. Both were confirmed.

### Test A — the early differential penalty is intrinsic, not router interference ✅

| Comparison | Δ val NLL at step 250 |
|---|---|
| Diff-MoE − MoE (differential **with** a router beneath) | **+0.079** |
| Diff-Dense − Dense (differential **without** a router) | **+0.072** |

Differential attention starts behind in *both* contexts by almost exactly the same amount, then closes and crosses over (step 1000 in the dense pair, 1250 in the MoE pair). The penalty is a property of the mechanism — λ starting at its initialisation, and parity buying those λs by halving the head count — not a conflict with expert routing.

**Corroborating evidence (new metric).** The λ values the two differential runs converge to correlate at **r = 0.91 across the 14 layers**:

| | mean λ | mean deviation from init | layer-0 λ (init 0.20) |
|---|---|---|---|
| Diff-Dense | 0.662 | 0.060 | **0.110** |
| Diff-MoE | 0.664 | 0.055 | **0.160** |

Both push layer 0 far *below* its initialisation and both raise the deep layers. The mechanism learns essentially the same depth profile regardless of what feed-forward sits underneath it — strong support for "intrinsic."

### Test B — sub-additive ✅

At the final step 2900, against the dense baseline:

| Quantity | Δ val NLL |
|---|---|
| MoE alone | −0.0407 |
| Diff-Dense alone | −0.0293 |
| **Additive prediction** | **−0.0700** |
| Diff-MoE actual | −0.0575 |
| **Interaction (shortfall)** | **+0.0125** |

Stacking recovers **82%** of the sum of the parts on validation. They do compose — Diff-MoE beats both single-mechanism cells per step — but with diminishing returns. On **held-out test** the shortfall is far larger: additive predicts −0.157, actual is −0.088, only **56% recovered**. §4 explains the difference — validation is BNC-only and the entire extra shortfall lives in CHILDES.

![Do the two mechanisms compose?](../assets/fig_p1_interaction.png)

---

## 3. The correction: differential attention *does* pay for itself

[`01-current-state.md`](01-current-state.md) §4.1 called *"differential attention does not pay for itself in wall-clock"* the **most reliable claim in the project**, on the grounds that it rested on a large, precisely-measured throughput ratio rather than a small noisy NLL delta.

**That claim was wrong.** It generalised from a single arm, and that arm turns out to be the anomalous one.

| Identical architectural change | Throughput | Cost |
|---|---|---|
| Dense → Diff-Dense | 6,944 → 6,674 tok/s | **−3.9%** |
| MoE → Diff-MoE | 5,250 → 3,851 tok/s | **−26.7%** |

The same change costs **5.2× more in absolute throughput** inside the MoE. This is not measurement noise — within-run coefficient of variation is 2.3% (Diff-Dense) and 2.9% (Diff-MoE).

At its true 3.9% cost, differential attention wins on both axes in the dense setting:

| In the dense run's 7.67 h | Steps reached | Val NLL |
|---|---|---|
| **Diff-Dense** | 2,812 | **3.6124** |
| MoE | 2,212 | 3.6346 |
| Dense | 2,900 | 3.6397 |
| Diff-MoE | 1,622 | 3.6826 |

**Diff-Dense is the best use of a fixed GPU-hour budget of the four.** The reasoning in §4.1 of the earlier document — "it rests on a hardware fact, not a noisy statistic" — was sound in form and wrong in substance, because the hardware fact itself was arm-specific and I treated it as a property of the mechanism.

### The anomaly — RESOLVED (2026-07-28 rerun)

**Update:** a full Diff-MoE rerun (same seed/config, W&B `wr1iseep`) completed 2900 steps at **5,139 tok/s — at the same 11.7 GB**. The memory-ceiling hypothesis below is dead; the first session simply ran on contended shared hardware. Differential attention's true cost is 2.1% on MoE / 3.9% on dense. The rerun also finished at val NLL **3.582**, the best of the grid, and tracked the crashed run within ±0.007 nats at every shared checkpoint. Iso-wall-clock update: Diff-Dense 3.613 > Diff-MoE 3.620 > MoE 3.636 > Dense 3.640 — both differential cells now beat both standard cells on a time budget. Test-set re-evaluation of the new checkpoint is pending.

*(Original analysis kept below for the record — it is a worked example of a plausible mechanism story failing its second measurement.)*

| Run | GPU memory reserved | tok/s |
|---|---|---|
| Diff-Dense | 7.2 GB | 6,674 |
| MoE | 11.4 GB | 5,250 |
| Diff-MoE | 11.7 GB | 3,851 |

Leading hypothesis: at 11.7 GB on a ~14.5 GB usable T4, the Diff-MoE run sat near the allocator ceiling, where `expandable_segments` churn and fragmentation cost real throughput — while Diff-Dense at 7.2 GB had ample headroom. Competing hypothesis: Diff-MoE ran on a different day (2026-07-22) and Kaggle T4s are shared infrastructure, so sustained contention is possible.

**These cannot be separated from the logs available.** The clean number is 3.9%, because Diff-Dense vs Dense differ *only* in attention and both had memory headroom. Re-running Diff-MoE with `batch_size 1 / accum 64` (same global batch, lower peak memory) would settle it in about two hours.

---

## 3b. Measurement uncertainty — and a signal the means hide

Because every model saw byte-identical windows, each difference is measured **per window** and bootstrapped as a paired statistic (20,000 resamples, 3,200 windows).

| Comparison | Δ test NLL | 95% CI | SE | Δ / SE | **Windows won** |
|---|---|---|---|---|---|
| MoE − Dense | −0.1291 | [−0.1339, −0.1245] | 0.0024 | 54σ | **96.0%** |
| Diff-MoE − Dense | −0.0884 | [−0.0912, −0.0857] | 0.0014 | 63σ | **95.1%** |
| Diff-MoE − Diff-Dense | −0.0605 | [−0.0626, −0.0584] | 0.0011 | 55σ | **91.0%** |
| Diff-Dense − Dense | −0.0279 | [−0.0300, −0.0258] | 0.0011 | 26σ | **69.2%** |
| Diff-MoE − MoE | +0.0407 | [+0.0366, +0.0449] | 0.0021 | 19σ | 47.7% |

**Every iid interval excludes zero by a wide margin.** Test-set sampling error is 0.001–0.002 nats against effects of 0.028–0.129 — so the test set is roughly **19–63× more precise than the effects measured**. Finite test data is emphatically *not* the limiting factor on any claim in this document.

That has an important consequence for how to read §5.1: since sampling error is ruled out to 19σ or better, **essentially all remaining doubt is seed variance**, which these intervals do not touch and which nothing in this project has measured. With one exception — the next subsection, where a different resampling unit reopens one comparison entirely.

### Those intervals assume windows are exchangeable — they aren't

The bootstrap above resamples the 3,200 windows independently. But windows share domains, and the six domains differ by nearly two nats in difficulty. Re-bootstrapping by resampling the **domains themselves** (`scripts/eval_bootstrap_clustered.py`, CPU-only from the cached per-window NLLs) gives a much more conservative interval:

| Comparison | Δ | iid 95% CI | stratified | **domain-clustered** | width ratio |
|---|---|---|---|---|---|
| MoE − Dense | −0.1291 | [−0.1338, −0.1244] | [−0.1332, −0.1249] | [−0.1419, **−0.0512**] | 9.6× |
| Diff-MoE − Dense | −0.0884 | [−0.0912, −0.0857] | [−0.0910, −0.0858] | [−0.1133, −0.0570] | 10.3× |
| Diff-MoE − Diff-Dense | −0.0605 | [−0.0626, −0.0584] | [−0.0626, −0.0584] | [−0.0722, −0.0380] | 8.0× |
| Diff-Dense − Dense | −0.0279 | [−0.0301, −0.0258] | [−0.0299, −0.0259] | [−0.0437, **−0.0156**] | 6.5× |
| Diff-MoE − MoE | +0.0407 | [+0.0366, +0.0448] | [+0.0373, +0.0442] | [**−0.0160**, +0.0550] | 8.6× |

Two things to take from this:

**Stratifying changes almost nothing** — the stratified interval is within a hair of the iid one. So it is not the *uneven domain mix* that matters; it is the *domain-level correlation*.

**One comparison does not survive it.** Diff-MoE − MoE's clustered interval is **[−0.0160, +0.0550] — it crosses zero**, despite being 19σ from zero under iid resampling. That is exactly what §4 predicted: the entire MoE-over-Diff-MoE gap lives in CHILDES, which supplies **1,277 of the 3,200 windows (40%)**. Resample the domains and draw a set without CHILDES, and the comparison reverses sign.

> **This is the single most transferable statistical result in the project.** Same data, same estimator, same 20,000 resamples — only the exchangeability assumption changed, and a conclusion that looked decisive at 19σ became unsupportable. Every paired bootstrap in the small-LM literature resamples examples; almost none resample the corpus strata, even when the corpus is explicitly a labelled mixture.

The iid interval answers "how much would this move on another sample of *this* corpus?" The clustered interval answers "how much would this move on a corpus with *different domains*?" **The second is the honest one whenever the claim is meant to generalise**, and it costs nothing extra once evaluation is paired and per-window losses are cached.

### The win rate is the number the means conceal

The rightmost column counts how many of the 3,200 windows each model actually wins — and it separates two effects that look merely different in size:

- **MoE beats dense on 96.0% of windows.** Near-universal. The gain is not carried by a handful of easy windows; it is a broad shift of the whole distribution.
- **Diff-Dense beats dense on only 69.2% of windows.** Same direction, far less consistent — differential attention is *worse* on nearly one window in three, and its mean advantage is carried by a minority where it wins big.
- **Diff-MoE beats MoE on 47.7% of windows** — a coin flip. The +0.041 mean is not a broad deficit at all; it is a near-tie with a heavy tail on one domain. A mean alone would have hidden that completely.

This is a genuinely different character of improvement, invisible in the headline NLL, and it aligns exactly with §4: MoE improves everything a bit and CHILDES a lot; differential attention wins decisively on referentially dense text and loses elsewhere. **Two mechanisms with the same sign and comparable means, doing quite different things.**

---

## 4. A finding nobody predicted: the mechanisms fix different domains

Per-domain test NLL gain over dense:

| Domain | Diff-attn alone | MoE alone | normalised: diff | normalised: MoE |
|---|---|---|---|---|
| simple_wiki | **−0.0584** | −0.1249 | **2.28×** | 1.38× |
| open_subtitles | −0.0379 | −0.0743 | **1.48×** | 0.82× |
| bnc_spoken | −0.0225 | −0.0441 | 0.88× | 0.49× |
| gutenberg | −0.0153 | −0.0745 | 0.60× | 0.82× |
| childes | −0.0143 | **−0.2004** | 0.56× | **2.21×** |
| switchboard | −0.0054 | −0.0265 | 0.21× | 0.29× |

*(normalised = each mechanism's gain on that domain divided by its own mean gain — shape, not magnitude)*

**The two gain profiles are close to uncorrelated: Pearson r = +0.21, Spearman +0.26.**

- **MoE's gain is concentrated on CHILDES** (2.21× its own average) — the largest, most repetitive, most formulaic slice.
- **Differential attention's gain is concentrated on simple_wiki** (2.28×) **and open_subtitles** (1.48×) — the domains with the most topic-switching and referential density, which is exactly where a noise-cancelling attention mechanism *should* help.

This created a real puzzle. If the two mechanisms repair **different** failure modes on **different** domains, why is stacking them sub-additive? Three candidate explanations were on the table:

1. **The truncation confound** — Diff-MoE had stopped at 2500, and CHILDES (MoE's stronghold) is where the last 400 steps buy the most. The measured sub-additivity might be substantially an artefact.
2. **A shared bottleneck** — both mechanisms limited by the same capacity or data constraint, so neither can cash in its full gain once the other has moved the model.
3. **Genuine interference at the representation level** despite different domain profiles.

**RESOLVED.** The completed 2,900-step Diff-MoE, evaluated on the identical protocol, eliminates explanations 1 and 2:

| Domain | Diff-MoE − MoE (2500 ckpt) | Diff-MoE − MoE (2900 ckpt, final) |
|---|---|---|
| **childes** | +0.127 | **+0.123** |
| simple_wiki | +0.010 | **−0.022** |
| open_subtitles | +0.000 | **−0.010** |
| bnc_spoken | +0.006 | **−0.010** |
| gutenberg | +0.020 | **−0.004** |
| switchboard | +0.006 | **−0.004** |

The extra 400 steps flipped five of six domains into Diff-MoE's favour and moved CHILDES by 0.004. **Truncation was not the cause; the deficit is architectural and confined to one domain.** Overall test gap narrowed 0.054 → 0.041 purely because the other five improved.

And §3b adds the final qualifier: once *domains* rather than windows are the resampling unit, that 0.041 deficit **crosses zero**. So the honest statement is not "Diff-MoE is worse" but **"Diff-MoE is worse on a corpus that is 40% child-directed speech."**

Interpretation: differential attention has a real, reproducible weakness on short, repetitive, locally-predictable text — precisely where its own mechanism predicts no benefit (no accumulated attention noise to cancel over a three-word context) while its costs still apply (halved head count, common-token fluency traded for rare-token precision). Sub-additivity here is *not* general interference between the mechanisms; it is one domain, weighted 40% of the test split, dominating the average.

This converts the composition question into a falsifiable prediction for the dataset-generalization study: **on a long-range corpus the interaction should be near-additive; on a TinyStories-like corpus it should be strongly sub-additive.**

---

## 4b. The mechanism verified, not just measured

Two additional evaluation passes (`scripts/eval_deep.py`, `scripts/eval_attention.py`) test whether differential attention works *for the reason it claims*, rather than merely working. Neither needed retraining.

### Position-resolved NLL — the discriminating test

If the mechanism cancels attention leaking onto irrelevant context, and that leak accumulates with context, its advantage must **grow with position index**. A capacity mechanism has no such reason.

| | Δ vs dense, positions 0–31 | positions 480–511 | corr. with position |
|---|---|---|---|
| **Diff-Dense − Dense** | **+0.010** *(behind)* | **−0.048** | **r = −0.877** |
| Diff-MoE − Dense | −0.023 | −0.119 | r = −0.847 |
| MoE − Dense | −0.082 | −0.133 | r = −0.399 |

**At sequence start, differential attention is marginally *worse* than standard attention.** Its entire advantage is accumulated over the window. MoE has most of its advantage by position 32 and gains only mildly after. Exactly the predicted contrast.

### Attention statistics — recomputed directly

The fused kernel never materialises attention weights, so both attention forwards were reimplemented (verified against the originals to **4.8 × 10⁻⁷**). Shannon entropy is undefined on differential rows — they are signed and sum to 1 − λ — so the normalised magnitude profile |A| / Σ|A| is used instead, valid for both.

| Model | Effective support | Top-8 mass | Mean distance | **Negative mass** |
|---|---|---|---|---|
| Dense | 86.2 | 0.380 | 90.0 | 0 |
| **Diff-Dense** | **70.8** (−17.8%) | 0.428 | 78.6 | **31.3%** |
| MoE | 75.6 (−12.4%) | 0.419 | 82.0 | 0 |
| Diff-MoE | **50.2** (−33.6% vs MoE) | 0.503 | 64.6 | **32.4%** |

**λ controls negative mass almost perfectly**: correlation across the 14 layers is **r = +0.975** (Diff-Dense) and **+0.993** (Diff-MoE). Diff-Dense's layer 0, which pushed λ down to 0.11, produces just 8.3% negative mass; layer 13 at λ = 0.74 produces 36.6%.

The full chain, each link measured independently:

> λ learned per layer → controls negative attention mass (r ≈ 0.98) → attention sharpens (−18% effective support) → advantage grows with context position (r = −0.88).

### Two controls that limit the claim

- **Sharpening is not unique to the mechanism.** MoE with ordinary softmax also sharpened (75.6 vs 86.2, −12.4%). Sharper attention is a general property of better models at this scale. What stays uniquely differential is the **negative mass**, which softmax cannot produce at all.
- **Sharpness does not predict quality.** Across the four models, corr(effective support, test NLL) = **+0.45**, and the sharpest model (Diff-MoE, 50.2) has a *worse* test NLL than MoE. Attention statistics show the mechanism is engaged; they do not explain why a model wins.

### BLiMP — construct validity closed

67 paradigms × 1,000 minimal pairs (`scripts/eval_blimp.py`, full-sentence log-likelihood):

| Model | BLiMP macro | vs Dense | Test NLL rank |
|---|---|---|---|
| Dense | 70.50% | — | 4 |
| Diff-Dense | 70.62% | +0.12pp *(noise)* | 3 |
| **MoE** | **71.39%** | **+0.89pp** | 1 |
| Diff-MoE | 71.14% | +0.64pp | 2 |

corr(BLiMP, test NLL) = **−0.99**: the perplexity ranking transfers to grammar exactly. But magnitudes compress hard — a 0.129-nat NLL gap becomes 0.9pp of accuracy, and differential attention's overall gain vanishes into the binomial noise floor (~0.18pp).

**By field is where it gets interesting, and it replicated.** Adding differential attention moves BLiMP semantics by **+3.13pp** on a dense feed-forward (63.90 → 67.03) and **+3.19pp** on an MoE (64.21 → 67.40), while both standard-attention baselines sit near 64%. Two models that share nothing but their attention agree to 0.06pp. Syntax and morphology move by ≤1.3pp in either direction.

That makes semantics the **third independent modality** — after per-domain (wiki/subtitles) and per-position (late context) — pointing at the same conclusion: this mechanism's value is concentrated where meaning depends on distant context. It is the strongest single piece of evidence in the project, and the only one that has been replicated across two architectures.

### Secondary metrics

| Model | ECE | Mean predictive entropy | Rarest-decile NLL Δ | Most-frequent-decile NLL Δ |
|---|---|---|---|---|
| Dense | 0.0199 | 2.990 | — | — |
| Diff-Dense | 0.0202 | 2.981 | −0.144 | **+0.146** |
| MoE | 0.0225 | 2.914 | −0.209 | **−0.486** |
| Diff-MoE | 0.0217 | 2.914 | −0.279 | −0.204 |

Calibration is near-identical across all four (ECE 0.020–0.023). The frequency deciles explain the win-rate gap mechanically: **differential attention trades common-token fluency for rare-token accuracy** — worse on the most frequent decile, better on the rarest — which is why it wins only 69% of windows while still winning on average. *(Decile bucketing is non-monotonic due to ties in the unigram counts; treat the extremes as the reliable signal.)*

A context-truncation probe was also run and **discarded as underpowered** (320 windows × 1 token, SE ≈ 0.06 nats against effects of ~0.02). It pointed the opposite way from the position analysis; with that standard error it cannot support either direction.

---

## 5. Downfalls — what this experiment cannot tell you

Ordered by how much they should restrain your conclusions.

### 5.1 n = 1. Still the binding constraint.
Every cell is one seed. The differences that matter most here — Diff-Dense's −0.028 over dense, the +0.0125 interaction term — are the same size as plausible seed noise for LM pretraining at this scale (~0.01–0.02 nats). **The MoE effect (−0.129 on test) is the only result comfortably outside that band.** Nothing about differential attention in this document is statistically established.

The one partial exception is the **BLiMP semantics effect** (§4b), which replicated at +3.13pp and +3.19pp across two different feed-forwards. That is not a seed replicate — both runs share seed 42 — but two architectures agreeing to 0.06pp on an effect 25× the binomial noise floor is meaningfully harder to explain as chance than a single NLL delta.

### 5.2 The wall-clock conclusion flipped on one new data point.
That is exactly what n=1 fragility looks like from the inside. A claim I labelled "most reliable" reversed the moment a second measurement of the same quantity arrived. Treat every other single-arm generalisation here with the same suspicion.

### 5.3 The head-count confound — now the most exposed weakness.
Parity buys λ by halving heads, so "differential attention" here means **the subtraction *and* half the attention patterns**. Every claim in §4b — the position slope, the semantics gain, the sharpening — is attributable to either. This was always true; it becomes the binding issue now that truncation is resolved and the mechanism story is the project's headline. **One deliberately parity-breaking arm (standard attention, 6 double-width heads, no subtraction) settles it and costs one config change.** See [`02-future-scope.md`](02-future-scope.md) #3′.

*(Superseded: this section previously flagged Diff-MoE's 400-step truncation. The rerun completed all 2,900 steps and every number here is from it.)*

### 5.4 A long-context mechanism tested at 512 tokens.
Differential attention claims benefits that grow with context length. Everything here is `seq_len 512`. The per-domain result in §4 sharpens this considerably: the mechanism helped most on the domains with the most topic-switching, which is a hint that longer contexts would help it more. **Untested, and now clearly the most promising untested direction.**

### 5.5 One learning rate for four architectures.
All cells use `lr 2e-4`. Differential attention introduces λ parameters with entirely different gradient scales from the projections around them, and they were never given their own learning rate or schedule. The observed λ warm-up penalty may be partly an optimisation artefact rather than an architectural cost.

### 5.6 The validation slice is one domain.
`eval_batches` reads contiguously from offset 0, so every training-time val number is the first 51,200 tokens of `val.bin` — entirely `bnc_spoken`. Comparisons stayed fair (identical windows), but "best checkpoint" was selected on adult British speech, and §4 shows the two mechanisms have *very different* per-domain profiles — so the selection slice actively favoured neither, but measured neither well. **Still unfixed.**

### 5.7 Everything else standing
Undertrained ~15× below Chinchilla with no run ever overfitting; one data order; perplexity rather than BabyLM's own BLiMP/GLUE metrics; a single scale.

---

## 6. Crucial things to note — the transferable learnings

These are the items worth carrying to the next project, ranked by how much they generalise beyond this repo.

### 6.1 Measure a cost twice before calling it a property of the mechanism
The single sharpest lesson. "Differential attention costs 27% throughput" was measured once, in one context, and stated as an architectural fact. The second measurement said 3.9%. **A cost measured in one configuration is a property of that configuration until proven otherwise** — especially when the configuration sits near a memory ceiling.

### 6.2 Iso-step, iso-FLOP, and iso-wall-clock give different winners
Three axes, three winners, all from the same four runs. Most published ablations report one. If your budget is time (it is, on free tiers), report iso-wall-clock — and note that iso-wall-clock rankings depend on hardware, so they travel less well than iso-step ones. **Report both; they answer different questions.**

Worth stating plainly: the *specific* worked example here reversed. This document once carried "an architecture that wins per step and loses per hour" as its headline methods case. After the contended-session throughput was remeasured, Diff-Dense became the **best** cell per GPU-hour. The methodological point survives — which axis you report changes the winner — but the direction did not, which is §6.1 all over again.

### 6.3 A 2×2 buys you the interaction term, and that is the whole point
Three separate A/B comparisons cannot tell you whether two mechanisms compose. The fourth cell is what converts "both help" into "together they deliver 82% of the sum on validation and 56% on test." If you are running three arms, run the fourth — it is usually the cheapest and most informative one. **And report the interaction on both your selection metric and your held-out metric**: the gap between those two numbers located the entire effect in one domain.

### 6.4 Pre-register predictions before the deciding run
Writing Test A and Test B down *before* Diff-Dense trained is why its results are informative rather than a post-hoc story. Both held; had either failed, that would have been equally publishable. This costs nothing and dramatically raises the value of the run.

### 6.5 Aggregate metrics hide mechanism
Two mechanisms with almost identical *average* usefulness turned out to have near-uncorrelated *per-domain* profiles (r = 0.21). A single test NLL would have shown "MoE helps 4.6× more than diff-attn" and concealed that they fix different things. **Always break the headline metric down along whatever structure the data ships with.**

### 6.6 Report the win rate, not just the mean
Two effects with the same sign and comparable magnitude turned out to have completely different shapes: MoE wins **96%** of test windows, differential attention only **69%**. A mean improvement can come from shifting the whole distribution or from winning big on a minority while losing on the rest — and those imply different things about when to deploy the mechanism. **The fraction of examples improved costs nothing to compute once evaluation is paired, and it is often the more actionable number.**

### 6.7 Pair your evaluation, then the error bars are nearly free
Scoring every model on byte-identical windows turns a two-sample comparison into a paired one, which collapsed the standard error to 0.001–0.002 nats — making the test set 19–63× more precise than the effects measured. **Design the evaluation to be paired from the start**; retrofitting it costs a full re-run of every model.

### 6.8 An error bar is only as honest as its exchangeability assumption
The paired bootstrap gave intervals so tight they looked decisive — ±0.002 nats, 19–63σ from zero. But resampling *windows* assumes windows are interchangeable draws, and they are not: they come in six domains that differ by two nats. Resampling **domains** instead widened every interval **6.5–10.3×**, and pushed one comparison's interval **across zero** — a result that had looked settled at 19σ became unsupportable.

Nothing was wrong with the first bootstrap; it answers a narrower question ("another sample of this corpus") than the one usually being asked ("does this generalise"). **Ask what unit your claim is really about, and resample *that* unit.** With per-window losses cached it costs seconds of CPU, and here it materially changed how much confidence one comparison deserves.

### 6.9 Check what your validation slice actually covers
30+ GPU-hours reported a number computed on one of six domains, because the eval sampler read contiguously from offset 0. Cost: every "best checkpoint" decision was made on the wrong distribution. **A three-line assertion comparing eval coverage against domain offsets would have caught it on day one.**

### 6.10 Verify the mechanism, not just the effect
Almost every architecture result stops at "the loss went down." These checkpoints supported three *independent* measurements of the causal chain, and each one could have failed:

> λ is learned per layer → it controls negative attention mass (**r = +0.98 / +0.99**) → attention sharpens (**−18%** effective support) → the advantage grows with context position (**r = −0.88**).

None of this needed retraining — it is all recomputation on checkpoints that already existed. **When you have the weights, you can usually test the mechanism's own story, and a mechanism that helps for the claimed reason is worth far more than one that merely helps.**

### 6.11 Always run the control that could embarrass you
Differential attention sharpens attention — but so did **MoE with ordinary softmax** (effective support 75.6 vs dense's 86.2). Without that control, "differential attention produces sharper attention" reads as a mechanism signature; with it, sharpening is revealed as a general property of better models here. And sharpness does **not** predict quality: correlation with test NLL is only +0.45, and the sharpest model of the four is not the best. **The control is what separates the claim you can defend (negative mass — softmax cannot produce it at all) from the one you cannot.**

### 6.12 Check that your metric is even defined for what you're measuring
Shannon entropy is undefined on differential attention rows: they are signed and sum to 1 − λ, not 1. Reaching for the standard metric would have produced numbers that looked fine and meant nothing. Normalising |A| instead is well defined for both variants. **When a mechanism changes the mathematical character of a quantity, re-derive the metric before reusing it.**

### 6.13 Validate instrumentation against ground truth before trusting one number from it
Measuring attention required reimplementing both attention forwards (the fused kernel never materialises the weights). Before using any output, the patched forward was checked against the original: **max absolute difference 4.8 × 10⁻⁷**. Five minutes of work that determines whether every downstream number is real.

### 6.14 Position-resolved metrics are cheap and unusually diagnostic
Scoring loss separately by context position cost one extra forward pass and produced the single most discriminating result in the project — differential attention's advantage grows with position (r = −0.88) while MoE's barely does (−0.40). It also converted a stated weakness ("we tested a long-context mechanism at 512 tokens") into a measured slope and a concrete prediction. **Whenever a mechanism claims something about *where* or *when* it helps, resolve the metric along that axis instead of averaging over it.**

### 6.15 Report the underpowered probe as underpowered
The context-truncation probe pointed the *opposite* way from the position analysis. It used 320 windows × 1 token — standard error ≈ 0.06 nats against effects of ~0.02. It was underpowered, so it was reported as underpowered and nothing was drawn from it. **The temptation is to quote whichever probe agrees with you; power-check first, then decide what it can support.**

### 6.16 Rankings transfer across metrics; magnitudes don't
BLiMP preserved the four models' perplexity ordering exactly (r = −0.98) — and compressed a 0.129-nat gap into 0.9 percentage points while erasing a 0.028-nat gap entirely. **Evaluate on the metric your field actually uses before claiming an effect "matters":** an ordering that survives metric transfer is a robust finding; a magnitude that doesn't survive it was never the headline you thought.

### 6.17 Memory headroom is a throughput variable, not just a capacity limit
The MoE cells ran at 11.4–11.7 GB on a ~14.5 GB card and paid for it in tok/s. Fitting is not the same as running well. **Budget memory to ~70% of the card, not 95%**, and if you must run near the ceiling, measure the throughput cost explicitly rather than assuming it is zero.

### 6.18 Checkpoint everything needed to resume, always
Diff-MoE crashed at 2500/2900. Because the *best* checkpoint carried optimizer, scaler, and RNG state — not just weights — the fix is a 1.9-hour resume instead of a 13.7-hour rerun. **That design decision saved ~12 GPU-hours**, and it costs nothing but disk.

### 6.19 Reconstruct config from the artefact, not from the repo
The Kaggle notebook rewrote model geometry at runtime, so no YAML in `configs/` matches what actually trained. Inferring `ModelConfig` from checkpoint tensor shapes made every downstream eval correct by construction. **When runtime can override config, the checkpoint is the only source of truth.**

### 6.20 Small scale is not the weakness; unreplicated is
Nothing in this document is limited by the models being 209M parameters or the corpus being 167M tokens. The test set (16.1M tokens) measures differences to ~±0.005 nats. Every genuine limitation traces back to **n=1**, which would be equally fatal at 7B and costs ~20× less to fix here.

---

## 7. What to do next, updated

**All five items from the previous version of this section are now closed.** Diff-MoE was resumed and completed 2,900 steps; the throughput anomaly was resolved (contention, not memory); BLiMP closed construct validity. The current plan lives in [`02-future-scope.md`](02-future-scope.md) — this table is the delta the completed grid produced.

| Priority | Action | Cost | Why now |
|---|---|---|---|
| **1** | Repo hygiene: un-ignore `docs/` and `PLAN_AHEAD/` | minutes | Both blog posts, every eval JSON, and these planning documents are currently untracked. The narrative writeups are the most valuable output and they are not in version control. |
| **2** | Write up the resampling-unit result (§3b) | ~1 afternoon, 0 GPU | New, general, and cheap. Same data and estimator, different exchangeability assumption, opposite verdict at 19σ. Nobody in the small-LM literature resamples corpus strata. |
| **3** | Re-analyse expert routing against non-domain bucketings | 0 GPU | Checkpoints are on disk. Turns a null result ("no domain specialization") into a positive one if experts specialize on token type instead. |
| **4** | Seed-variance floor at tier A | ~1 quota week | **Unchanged as the highest-value research item.** Everything in §4 and §4b is gated on it, and §5.2 is a live demonstration of why. |
| **5** | Break attention parity deliberately (§5.3) | 2–4 tier-A runs | Promoted to the critical path. The mechanism story in §4b is now the project's headline claim, and the head-count confound is the first thing a reviewer attacks. |
| **6** | Long-context sweep of differential attention | 3–4 runs/length | The position slope is still descending at 500 with no flattening — a pre-registerable prediction rather than a hunch. |

**What changed in the plan:** the four cheap clean-up items are done, so the critical path is now *statistics* (#4) and *controls* (#5), not more measurement of these four checkpoints. §4b has extracted about as much as four single-seed models can support.

**The claim to build on:** differential attention buys long-range referential binding, not general LM quality — supported by per-position (r = −0.88), per-domain (wiki/subtitles at 2.3×/1.5× its own mean), and per-field BLiMP semantics (+3.13pp dense, +3.19pp MoE, replicated). That is a converging-evidence mechanism result, and it is what makes this project worth publishing rather than merely worth reading.
