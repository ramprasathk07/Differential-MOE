# runs_export — data for the Part 2 figures

`make_figures_part2.py` reads this folder. One subdirectory per training run,
each holding up to three files the training/eval already writes (all small, no
wandb key needed):

```
runs_export/
  s_dense/     metrics.csv  report.json  final_test_eval.json
  s_diff/      metrics.csv  report.json  final_test_eval.json
  s_moe/       metrics.csv  report.json  final_test_eval.json
  s_diffmoe/   metrics.csv  report.json  final_test_eval.json
```

- `metrics.csv` / `report.json` come from `/kaggle/working/checkpoints/<run>/` (written by `src/train.py`).
- `final_test_eval.json` comes from `src/eval.py` (the eval run). Optional — the
  λ-by-depth and expert-entropy figures are skipped for any run missing it; the
  loss / val-NLL / aux figures only need `metrics.csv`.

Generate:

```
python docs/blog/make_figures_part2.py
```

Figures land in `docs/blog/assets/fig_s0*.png`. Partial data is fine — each
figure with no data is skipped with a printed note instead of failing.
