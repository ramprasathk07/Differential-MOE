"""Pull the full logged history of the s_diffmoe run from Weights & Biases.

Writes the complete per-step history to docs/blog/runs_export/s_diffmoe/ as a CSV
(one row per logged step, sparse across metrics that log at different cadences),
plus the run config + summary. make_figures_wandb.py turns it into the figures.
"""

import json
import os

import pandas as pd
import wandb

RUN = "New_103/diff-moe-kaggle/p9ut9628"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs/blog/runs_export/s_diffmoe")
os.makedirs(OUT, exist_ok=True)

api = wandb.Api()
run = api.run(RUN)
print("run:", run.name, run.id, run.state)

rows = list(run.scan_history())          # every logged row, no sampling
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "wandb_history.csv"), index=False)
print("rows:", len(df))
print("columns:", sorted(df.columns))

with open(os.path.join(OUT, "wandb_config.json"), "w") as f:
    json.dump(dict(run.config), f, indent=2, default=str)
with open(os.path.join(OUT, "wandb_summary.json"), "w") as f:
    json.dump({k: v for k, v in run.summary.items()
               if isinstance(v, (int, float, str, bool))}, f, indent=2, default=str)
print("wrote", OUT)
