# Future scope — what to run next, and why

**Companion to [`01-current-state.md`](01-current-state.md).** That document says what we know. This one says what to do about it.

**Revised 2026-08-01**, after the completed 2×2 and the full eval battery. Three items on the previous list are now closed or invalidated — see [§0](#0-what-changed-on-2026-08-01).

Ranked by **research value per GPU-hour**, assuming the compute budget stays what it has been: free Kaggle quota (~30 GPU-h/week on 2×T4) plus one local RTX 3060 for evaluation.

Everything is sized against **tier A (~16M params, ~1.5 GPU-h/run)** unless stated. That is the deliberate strategic choice running through this plan: tier S buys you *four* runs per quota week and no error bars; tier A buys you *twenty* and therefore actual statistics. **The scientific bottleneck is n, not scale.**

---

## 0. What changed on 2026-08-01

| Previous item | Status now |
|---|---|
| **#9 — score on BabyLM's actual metrics** | ✅ **Done.** BLiMP run on all four cells. Ranking transfers exactly (r = −0.99). Removed from this list; results in `01` §3.8. |
| **#3 — iso-wall-clock as a methods contribution** | ⚠️ **Worked example reversed.** The old framing was "an architecture that wins per step and loses per hour." That is now false: differential attention costs 2–4%, not 27%, and **Diff-Dense is the best cell per GPU-hour.** The methods point survives in stronger form — see [#3′](#3-iso-wall-clock-reporting-rewritten-worked-example). |
| **#2 — what do the experts specialize on?** | Still open, still zero-GPU, still the cheapest item here. Unchanged. |
| — | 🆕 **New finding to build on:** differential attention's value is *specific*, not general, and it replicated across feed-forwards. See [the thesis](#the-thesis-worth-defending). |
| — | 🆕 **New methods contribution:** the resampling unit changes a conclusion's significance. See [#1b](#1b-the-resampling-unit-is-a-choice-and-it-changes-answers). |

---

## The thesis worth defending

The project's headline question — *"do these two tricks earn their place?"* — is a blog question. The result the data actually supports is sharper, and it is what a paper would be built on:

> **Differential attention does not improve language modeling generally. It buys long-range referential binding specifically.**

Three independent measurement modalities agree, and one of them **replicated across two different feed-forwards**:

| Modality | Evidence |
|---|---|
| per-position | advantage ≈ 0 at position 0, grows to −0.048 by 511, **r = −0.88**; MoE (a capacity mechanism) shows r = −0.40 and saturates by position 32 |
| per-domain | 2.28× its own mean on simple_wiki, 1.48× on subtitles; 0.21× on switchboard |
| per-linguistic-field | BLiMP semantics **+3.13pp** (dense) and **+3.19pp** (MoE) against ~0 overall |

Plus a causal chain closed link by link: λ → negative attention mass (r ≈ 0.98) → sharper attention (−18% support) → advantage grows with position (r = −0.88).

**That is a stronger mechanistic story than most architecture papers ship.** Everything in Tier 1 below exists to make it defensible.

---

## Tier 1 — The critical path. Nothing else matters until these are done.

These three are what stand between "an unusually careful blog post" and "a result people cite."

### 1. Publish the noise floor for small-scale LM ablations
**Cost:** ~20 tier-A runs (4 cells × 5 seeds) ≈ 1 quota week · **Novelty:** high · **Risk:** none — it produces a number either way

Nobody publishes seed-variance baselines for small LM pretraining, yet hundreds of people run exactly these ablations — every BabyLM entrant, every "does trick X help" post, most MSc theses.

Output: *"At 16M params on BabyLM-strict, run-to-run σ of final validation NLL is X nats. Any ablation reporting an effect below 2X is unfalsifiable."*

A **reusable community artifact**, not just a result for this repo. It also retroactively determines whether this project's own differential-attention finding means anything. Vary the seed over *both* init and data order and report them separated if affordable — knowing which source dominates is itself useful.

> **Still the single highest-value experiment available.** Every conclusion in `01` §4 is gated on it.

### 1b. The resampling unit is a choice, and it changes answers
**Cost:** zero GPU — the per-window NLLs are cached in `per_window_nll.npz` · **Novelty:** medium-high · **Risk:** none

🆕 This project measured the same comparison two ways. Resampling **windows** gave Diff-MoE − MoE = +0.041, 19σ from zero, apparently decisive. Resampling **domains** gave [−0.016, +0.055] — crossing zero. Same data, same estimator, different exchangeability assumption, opposite verdict.

Every paired-bootstrap CI in the small-LM literature resamples windows or examples. Almost none resample the corpus strata, even when the corpus is explicitly a mixture. The write-up is short and the point is general: **an error bar is only as honest as its exchangeability assumption.**

Pairs with #1 and #15 into the methods paper. Costs an afternoon.

### 2. Turn the null result into a positive one: what *do* the experts specialize on?
**Cost:** zero GPU-hours for training — analysis on existing checkpoints · **Novelty:** medium-high · **Risk:** low

Domain specialization measured ≈ 0 (max TV 0.052). But domains are only one hypothesis. Bucket held-out tokens by:

- **part of speech** (function vs content words)
- **frequency decile** (does one expert take the rare tail?)
- **position in sequence** (early vs late context)
- **preceding-token identity** (syntactic locality)
- **whitespace / punctuation / numeral** surface classes

Then recompute routing TV against each bucketing. If any lights up while domains don't, that is a genuine mechanistic finding: *experts specialize on token type, not topic.* If none do, you have a far stronger negative result than "domains didn't work."

Cheapest item on this list. The checkpoints are already on disk, and `eval_deep.py` already computes frequency deciles.

### 3′. Break parity deliberately — separate the subtraction from the head count
**Cost:** 2–4 tier-A runs · **Novelty:** medium-high · **Risk:** none · **Was: not on the list**

🆕 Promoted to Tier 1 because it is **the first thing a reviewer attacks.** Right now "differential attention" means "the subtraction *and* half the attention patterns," and every claim in the thesis above is attributable to either.

Arms: (a) differential at 6 heads — current; (b) differential at 12 heads, accepting +λ params; (c) standard at 6 double-width heads, no subtraction; (d) standard at 12 heads — current baseline. Arm (c) is the one nobody runs and the one that settles it.

If the semantics/position effects survive arm (c) being flat, the mechanism claim is clean. If arm (c) reproduces them, the story was head width all along — which is a *more* interesting negative result, and publishable.

---

## Tier 2 — The strongest paper candidates, once Tier 1 lands

### 4. Test differential attention where it actually claims to win
**Cost:** 3–4 tier-A runs per context length · **Novelty:** high · **Risk:** medium (needs RoPE scaling care)

The measured position slope is **still descending at position 500 with no flattening.** That converts the old caveat into a sharp, pre-registerable prediction: **the advantage should widen with context.**

Sweep `seq_len` 512 → 1024 → 2048 → 4096 and add a synthetic long-range retrieval probe. If it widens, that is a clean replication-and-extension of Ye et al. outside their setup. If it doesn't, that is a meaningful failure-to-replicate — and negative replications of popular mechanisms are publishable and useful.

**Pre-register the prediction and the falsification threshold before running.** This project already did that once (the crossover prediction in Part 1) and it paid off.

### 5. The balance ↔ specialization Pareto frontier
**Cost:** 5–6 tier-A runs · **Novelty:** high · **Risk:** low — every outcome is informative

Sweep `aux_loss_coef ∈ {0, 0.001, 0.003, 0.01, 0.03}`; plot domain specialization (TV from uniform) against NLL and routing entropy.

What makes this novel: **BabyLM ships labeled domains.** Nearly all MoE interpretability work infers specialization post-hoc from clustering because no ground truth exists. Here it does. A trade-off curve measured against ground-truth labels is a contribution.

Three possible shapes, all worth publishing: specialization emerges and NLL improves (balance was over-tightened); specialization emerges and NLL degrades (a clean quantification of the trade); nothing emerges at any coefficient (specialization needs scale or heterogeneity beyond this).

Run this **after #2** — if #2 finds experts specializing on token type, this sweep should be re-aimed at that axis rather than domains.

### 6. MoE capacity scaling at fixed active compute
**Cost:** 5 tier-A runs · **Novelty:** medium-high · **Risk:** low

Sweep `n_experts ∈ {2, 4, 8, 16, 32}` holding active parameters constant (shrink `expert_inter_dim` as the bank grows). *The* core MoE question — how quality scales with raw capacity when compute per token is pinned — and it produces a **curve rather than a point**.

Especially interesting on a small fixed corpus: there should be a knee where extra experts stop paying because there isn't enough data to fill them. Finding that knee on BabyLM is a real result.

### 7. Data-constrained MoE: does sparsity help more or less when data is fixed?
**Cost:** 2–4 long tier-A runs (5–10 epochs) · **Novelty:** high · **Risk:** low

Every model here is ~15× under Chinchilla and **none ever overfit** — all curves still descending at cutoff. So the most basic question about MoE on small data is untouched: *which architecture turns first?*

Classic intuition says more parameters → overfit sooner. But sparsity may act as a regularizer (each expert sees a subset of tokens). Train dense and MoE to 5–10 epochs and find each turning point.

This lands directly on BabyLM's founding premise and is the question the challenge exists to ask.

### 8. Does the 100k frontier vocabulary actually pay at 200M params?
**Cost:** 4 runs + tokenizer prep · **Novelty:** medium-high · **Risk:** low

**77M of the 209M active parameters — 37% of the model — is the embedding table.** Compare cl100k against custom 8k/16k BPE at matched *non-embedding* parameters, so a small vocabulary frees ~60M parameters for the body.

Report in bits-per-byte, which is tokenizer-independent and already implemented. Immediately useful to anyone training a small LM.

---

## Tier 3 — Solid additions, lower ceiling

### 10. Shared-expert ablation
**Cost:** 2 runs · **Novelty:** low-medium · **Risk:** none

`n_shared_experts` is 0. DeepSeek's central MoE finding is that one always-on shared expert plus routed experts beats pure routing. One config value, directly testing a published claim at a new scale on a new corpus. Cheap credibility.

### 11. λ initialization and learnability ablation
**Cost:** 4 runs · **Novelty:** medium · **Risk:** low

λ moved at every layer, and both runs independently pushed layer 0 *below* its initialization (0.11 and 0.16 against 0.20) — so the schedule `0.8 − 0.6e^{−0.3ℓ}` is a hyperparameter that has never been questioned. Four arms: frozen λ at init; learned from constant init; learned from the paper's schedule (current); per-head vs per-layer λ.

Separates *"the subtraction helps"* from *"the schedule helps"* from *"learnability helps"* — three claims the current design bundles together. **The r = 0.91 agreement between the two runs' λ profiles is a strong hint that the schedule is doing real work**, which makes this more interesting than it looked before.

### 12. Curriculum ordering — the developmentally plausible question
**Cost:** 2–4 runs · **Novelty:** medium-high · **Risk:** medium

BabyLM's premise is *developmentally plausible* pretraining, but this project packs all six domains into one homogenized stream. Does curriculum order — CHILDES → subtitles → simple-wiki → Gutenberg — beat random ordering?

Interacts intriguingly with MoE: a curriculum might induce the very domain specialization the aux loss suppresses. That interaction connects #5 and #12 into one story.

### 13. Inference-cost accounting — MoE's actual deployment story
**Cost:** evaluation only · **Novelty:** medium · **Risk:** none

Training throughput was measured; inference was not. MoE's real pitch is inference efficiency. Measure tokens/sec at batch 1 / 8 / 64, peak memory, quality-per-GB.

Expect a useful surprise: at batch 1 routing overhead likely dominates and MoE loses; at large batch it should win. Practitioners care about this far more than training-step counts, and almost nobody at this scale reports it.

### 9′. Close the rest of BabyLM's evaluation suite
**Cost:** evaluation only · **Novelty:** low (BLiMP is done) · **Risk:** none

BLiMP is closed. GLUE-style fine-tuning probes and the BabyLM supplement tasks are not. Lower value than it was — the interesting question ("does the perplexity win transfer?") is already answered *yes* — but it makes the numbers leaderboard-comparable.

---

## Tier 4 — Repository traction, not research

### 14. Ship the ablation harness
**Cost:** engineering only · **Novelty:** low as research, high as a tool

Generalize the repo into a reusable instrument: config-driven N×M grids, parity assertions enforced at construction, iso-wall-clock reporting built into the report, eval-coverage linting, per-window **and per-stratum** paired bootstrap CIs emitted automatically.

Most repos publish results. Few publish **a harness that makes small ablations honest by default.** That is the stars-and-forks play, and this project has already paid the cost of learning what such a harness needs.

### 15. The eval-coverage linter, packaged standalone
**Cost:** an afternoon · **Novelty:** low · **Risk:** none

This project spent 30+ GPU-hours reporting a validation number computed entirely on `bnc_spoken`, because `eval_batches` reads contiguously from offset 0. Ship a utility that cross-references eval window offsets against `domain_offsets.json` and **warns when a validation set is secretly single-domain**.

That bug class near-certainly affects other repos. Small sharp tools earn attention disproportionate to their size — and it comes with a real war story.

### 16. Fix the repository's own hygiene
**Cost:** minutes · **Risk:** none

🆕 `docs/` and `PLAN_AHEAD/` are in `.gitignore`. **Both blog posts, every eval JSON, and these planning documents are untracked.** The narrative writeups — arguably the most valuable output — are not in version control. Un-ignore `docs/**/*.md`, `docs/blog/runs_export/*.json`, and `PLAN_AHEAD/`, keeping the `*.bin` / `*.pt` exclusions.

Also worth doing: chunk the vocabulary axis in `eval_deep.py`'s entropy computation. It currently allocates ~10 GB per batch and runs at ~6 min/model on a 12 GB card.

---

## Packaging: what these combine into

| Bundle | Items | Deliverable |
|---|---|---|
| **The honest-ablation paper** | 1 + 1b + 3′ + 15 | *"How large must a small-scale LM ablation be to mean anything?"* — no new architecture, no large GPUs. Now ~75% done, and **1b makes it sharper than it was**. Best paper-to-effort ratio on this list. |
| **The differential-attention note** | 3′ + 4 + 11 + the thesis | *"Differential attention buys referential binding, not perplexity."* Converging-evidence mechanism paper. The most *citable* bundle — but it is only defensible after 3′ and #1. |
| **The MoE-on-small-data paper** | 2 + 5 + 6 + 7 | *"What a mixture of experts learns when data, not compute, is the constraint."* Most scientifically interesting; needs the most compute. |
| **The public repo** | 14 + 15 + 16 + current README/blogs | The artifact people actually use. |

---

## Suggested order

1. **#16, #1b, #2 immediately** — zero GPU. Repo hygiene is minutes; the resampling write-up is an afternoon; the specialization re-analysis needs only checkpoints already on disk.
2. **#1 next** — one quota week, and it retroactively validates or invalidates everything else.
3. **#3′ straight after #1** — cheap, and it is the load-bearing control under the project's best claim. Do not publish the referential-binding thesis without it.
4. **Then pick a Tier-2 bundle based on what #1 reveals.** If the noise floor is large (σ > 0.02), pivot to #5/#6/#7 where effects are big enough to survive it. If small (σ < 0.01), **#4 becomes the headline experiment** and the differential-attention question is wide open in the good way.

## What not to do next

- **Do not scale up.** Scaling amplifies a result that cannot yet be defended, and tier S consumes a quota week for four runs with no error bars. Scale only after #1.
- **Do not switch corpora.** BabyLM's six labeled domains are this project's single most exploitable asset — they are what make #2, #5, #12, and the entire per-domain analysis possible. Change datasets only as a late generalization check.
- **Do not add architectures to the grid.** The 2×2 is not resolved at n=1. Widening before deepening makes the statistics worse.
- **Do not lead with "Diff-MoE loses to MoE."** It crosses zero under domain-clustered resampling (`01` §4.4). It is a fact about a corpus that is 40% child-directed speech, not about the architecture — and stating it as an architecture result is the kind of overclaim this project is otherwise careful to avoid.
