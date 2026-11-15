# Stable LatentMoE on one RTX 6000 Ada (48 GB)

Use [rtx6000ada_stable_latentmoe.yaml](rtx6000ada_stable_latentmoe.yaml) for a
from-scratch English base language model. This is a starting pretraining run;
chat behavior would need a later instruction tuning stage.

## Model and memory

| Setting | Value |
|---|---|
| Total / active parameters | 914,704,384 / 405,096,448, including tied embeddings |
| Width / layers / context | 1,024 / 20 / 1,024 tokens |
| Attention | Standard SDPA, 16 heads |
| FFN | 2 dense layers, then 18 Stable LatentMoE layers |
| Routed experts | 16 per layer, top-4, latent width 512, hidden width 1,536 |
| Shared experts | 2 per MoE layer, full model width, hidden width 512 each |
| Microbatch / accumulation | 2 sequences / 32 microbatches |
| Tokens per optimizer step | 65,536 on one GPU |
| Run length | 32,768 steps; 2,147,483,648 sampled tokens |
| Precision | Existing FP16 AMP + GradScaler, FP32 model and AdamW state |

Parameter counts come from `Transformer` and `count_params`, instantiated on
PyTorch's `meta` device. All experts reside in memory even though only four
are active per token. Parameters, gradients, and AdamW's two moments alone
need **13.63 GiB** (16 bytes per total parameter). Activations, autocast weight
copies, logits, optimizer temporaries, and CUDA allocations add to that.
The conservative microbatch leaves substantial room within 48 GB, but peak
VRAM has **not been measured on an RTX 6000 Ada**. The recipe uses the existing
training path without activation checkpointing or CPU offload.

This recipe keeps FP16 (`train.precision` defaults to `fp16`); set
`precision: bf16` to drop the GradScaler on Ada, or see `l_tier.md` for the
multi-GPU FSDP2 path. Evaluation uses FP32 with the same microbatch. Keep standard attention
for this initial run so it uses the existing SDPA path.

Allow at least **40 GB of free disk** for prepared data, checkpoints, and a
temporary checkpoint during replacement; Hugging Face caches need additional
space. Each model + AdamW checkpoint is approximately 10.22 GiB. The trainer
keeps `last.pt` and one best checkpoint. Host RAM must also accommodate loading
the FP32 model before the trainer moves it to CUDA.

## Data

Use [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
subset **`sample-10BT`**, revision **`v1.4.0`**. It is English web text filtered
for educational quality, with a randomly sampled 10B-token subset. It is a
useful starting corpus for general English and knowledge pretraining.

The recipe uses the
[SmolLM2 tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/tokenizer_config.json)
with 49,152 vocabulary entries. Only its tokenizer is loaded; all model
weights are initialized by this repository. The smaller vocabulary reduces
the output projection cost and stores IDs as `uint16`.

`src.data.prepare_hf` streams and shuffles documents, then writes **2.2B train
tokens + 1M validation + 1M test tokens**, about **4.40 GB** total. These counts
use the selected tokenizer and include document EOS markers; the upstream
10BT name uses a different tokenizer. Streaming avoids downloading the entire
subset before preparation, but it still transfers the text needed for this run.

Splits are assigned by a seeded hash of stripped document text before packing.
Identical text stays in one split; near-duplicate detection is not performed.
Each split has a fixed token cap and EOS between documents. When a split fills,
later documents assigned to it are skipped. The last document may be truncated.
The metadata records the source, revision, seed, counts, dtype, and tokenizer.
The preparer requires an empty output directory and restarts after interruption.

The trainer samples random token windows with replacement, so 2.15B sampled
tokens does not mean every prepared token is seen exactly once. This is an
initial compute budget, not a claim of converged training. Choose any longer
schedule before launching so its cosine decay covers the intended run.

## Prepare and train

Run from the repository root in an environment with CUDA-enabled PyTorch and
the dependencies from `requirements.txt`. Confirm
`python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

```bash
python -m src.data.prepare_hf --config configs/rtx6000ada_stable_latentmoe.yaml
python -m src.data.verify --data_dir data_fineweb_edu --config configs/rtx6000ada_stable_latentmoe.yaml
python -m src.train --config configs/rtx6000ada_stable_latentmoe.yaml --device cuda
```

Use `python -m src.train` on one GPU. Running the training command again
automatically resumes `checkpoints/rtx6000ada_stable_latentmoe_fineweb_edu/last.pt`.

### W&B login and metrics

Copy [.env.example](../.env.example) to `.env` in the repository root, then fill
in `WANDB_API_KEY` from [W&B](https://wandb.ai/authorize) and, optionally, your
`WANDB_ENTITY` team slug. The template sets `WANDB_ENABLED=true` and
`WANDB_PROJECT=latentmoe-fineweb-edu`. `.env` is ignored by Git.

The same plain `python -m src.train --config ... --device cuda` command above
loads `.env` automatically and authenticates using the key. No separate login
command or `--wandb` flag is needed. Existing shell variables override `.env`;
CLI flags such as `--wandb_project` override those defaults. `--env_file PATH`
selects another file. For an existing SDK login, the API key may remain blank.
These use W&B's documented
[environment settings](https://docs.wandb.ai/models/track/environment-variables).

Use `--no-wandb` to train without logging or `--wandb_mode offline` for local
logging without authentication or cloud uploads. The GPU smoke test below uses
offline mode. Install the new dependency with `python -m pip install -r requirements.txt`.

The dashboard includes:

- Training CE loss, total objective, perplexity, auxiliary and router losses,
  learning rate, gradient norm, tokens, training throughput, AMP scale, and
  skipped optimizer updates in the current process.
- Validation NLL, perplexity, top-1 accuracy, bits/token, and bits/byte when the
  tokenizer is available.
- Per-layer expert entropy, imbalance, assignment histograms and individual
  expert counts, router bias histograms and individual biases. Training routing
  describes the latest microbatch; validation routing covers all eval batches.
- Differential-attention lambda means and histograms when that attention is used.
- Current and peak allocated/reserved CUDA memory; CPU runs omit GPU metrics.
- Model geometry, parameter counts, training settings, and effective batch size
  in the run config; best validation score and final totals in the run summary.

Train and validation metrics share one history entry when they occur on the same
step. Charts use `optimizer_step` as the x-axis. The run ID is stored alongside
checkpoints in `wandb_run.json`; online restarts in the same output directory
resume that W&B run. Offline restarts produce separate local logging segments.
For a fresh experiment, choose a new `--out_dir` or YAML `run_name`. Environment
contents and API keys are never included in the logged model/training config.

Before the full run, use a separate output directory for a two-step GPU smoke
test, including gradient accumulation, AdamW state allocation, evaluation, and
checkpoint saving:

```bash
python -m src.train --config configs/rtx6000ada_stable_latentmoe.yaml --device cuda --max_steps 2 --out_dir checkpoints/smoke_rtx6000ada --wandb --wandb_mode offline
```

This uses the prepared corpus. To test preparation with less data first, use
`--out_dir data/hf_smoke --train_tokens 100000 --val_tokens 10000 --test_tokens 10000`
on the preparation command and `--data_dir data/hf_smoke` on the smoke training
command. Keep the full training command pointed at `data_fineweb_edu`.

If VRAM is tight, set `batch_size: 1` and `accum_steps: 64`; the effective batch
and token budget stay unchanged. After measuring peak VRAM through training
and evaluation, `batch_size: 4` with `accum_steps: 16` is an optional throughput
experiment. This eager implementation loops over experts, so throughput must
be measured rather than inferred from the active parameter count. Estimate
training hours as `2,147,483,648 / measured_tokens_per_second / 3600`, with
extra time for evaluation and checkpoints.

Use validation to select the checkpoint. Keep `test.bin` for final evaluation.
