# PLAN_AHEAD

Three documents: where the project stands, the detailed results record, and where it should go.

| Document | Answers |
|---|---|
| [`01-current-state.md`](01-current-state.md) | What has been run and measured, what those numbers do and do not support, the confounds, and a graded assessment of experimental validity |
| [`03-full-2x2-results.md`](03-full-2x2-results.md) | The complete results record for the 2×2, plus 20 transferable learnings worth carrying to the next project |
| [`02-future-scope.md`](02-future-scope.md) | What to run next, ranked by research value per GPU-hour, with cost estimates and what the items combine into as papers or a public repo |

**Snapshot: 2026-08-01.** All four cells complete at 2,900 steps; the full eval battery (test, paired + domain-clustered bootstrap, position-resolved NLL, attention matrices, BLiMP) has been rerun against the final checkpoints.

**The one-line summary:** the experiment is well-controlled, mechanistically well-evidenced, and under-replicated. Internal validity is strong (byte-identical configs, enforced parameter parity, identical eval windows); the mechanism evidence is unusually good for an ablation (three converging modalities, one replicated across architectures); statistical inference is still weak (n=1 per cell, no seed variance measured).

**The claim worth defending:** *differential attention does not improve language modeling generally — it buys long-range referential binding specifically.* See `01` §4.3.

**The next experiment worth running is not a bigger model.** It is the same small models several more times, plus the one control that separates the subtraction from the halved head count. See `02` Tier 1.

**Read `01` §5 before quoting any number from this project.**
