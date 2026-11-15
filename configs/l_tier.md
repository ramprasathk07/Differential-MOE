# L tier: run guide (48 GB GPUs, 1 to 4 of them)

Six cells share [l_base.yaml](l_base.yaml). The 2×2 is {standard, differential}
attention × {dense, MoE}, and the earlier Stable LatentMoE design is kept as a
third FFN column.

| Cell | Attention | FFN | Total params | Active | fp32 train state |
|---|---|---|---|---|---|
| `l_dense` | 24 × 64 | SwiGLU 4096 | 0.780B | 780.4M | 11.6 GiB |
| `l_diff` | 12 × 128, differential | SwiGLU 4096 | 0.780B | 780.4M | 11.6 GiB |
| `l_moe` | 24 × 64 | 8 experts, top-2, width 2048 | 2.026B | 780.6M | 30.2 GiB |
| `l_diffmoe` | differential | 8 experts, top-2, width 2048 | 2.026B | 780.7M | 30.2 GiB |
| `l_latentmoe` | 24 × 64 | Stable LatentMoE: 20 experts, top-4, latent 768 | 2.027B | 781.1M | 30.2 GiB |
| `l_diff_latentmoe` | differential | Stable LatentMoE | 2.027B | 781.1M | 30.2 GiB |

Counts come from `count_params` on the meta device. `tests/test_arch_options.py`
fails if any cell drifts more than 0.2% from the others or outgrows one 48 GB GPU.

Every cell shares: d=1536 with 24 layers, a 4096-token context, a 32K
byte-BPE vocabulary, untied embeddings, Derf norms, Block AttnRes (8 blocks),
XSA and document masking. Each of these is a flag, and the defaults in
`ModelConfig` reproduce the S tier exactly.

## 1 · Tokenizer and data (CPU, once)

```bash
# Score candidate vocab sizes per source, then train the chosen one (~1.4 GB text sample).
python -m src.data.train_tokenizer --hf_config configs/l_base.yaml --sweep \
    --candidates 16384 32768 49152 --pretrained hf:HuggingFaceTB/SmolLM2-135M
python -m src.data.train_tokenizer --hf_config configs/l_base.yaml --vocab_size 32768 \
    --out tokenizers/slimpajama_bpe32k.json
git add tokenizers/slimpajama_bpe32k.json        # the eval PC needs the same file

# Write 12B train tokens (24 GB), plus per-source val/test sets and the doc index.
python -m src.data.prepare_hf --config configs/l_base.yaml
```

Set `HF_TOKEN` in `.env` (a read token) before streaming. The corpus is
`gmongaras/SlimPajama-627B_Reupload` at a pinned commit, because the original
`cerebras/SlimPajama-627B` returns 404 on the Hub.

If `<|endoftext|>` is not id 0, `prepare_hf` stops with an error; set
`model.eos_id` to match. The trainer re-checks it against `data_l/meta.json`.

## 2 · Train

```bash
# 4 GPUs: FSDP2, accumulation 16
python -m src.train --config configs/l_diffmoe.yaml --gpus 0,1,2,3
# 1 GPU: same optimizer batch, accumulation 64
python -m src.train --config configs/l_diffmoe.yaml --gpus 0
```

- **Background runs:** `nohup` alone does not survive closing the terminal, because
  torchrun installs its own SIGHUP handler and stops the run. Start it in its own
  session instead: `setsid nohup python -m src.train ... > log 2>&1 < /dev/null &`
  (or run it inside tmux).
- **Resume:** run the same command again. It picks up `last.pt` even if the GPU
  count changed (for example, 4 GPUs one week and 1 GPU the next).
- **Before committing days:** run `--max_steps 200 --no-mlflow` once per cell
  and read `tok/s` and `train/gpu_peak_allocated_mb`.
- **Incremental ablations** need no new YAML. Each feature turns off with an
  override, for example:

  ```bash
  python -m src.train --config configs/l_diffmoe.yaml --gpus 0,1,2,3 \
      --override run_name=l_diffmoe_no_xsa --override model.xsa=false
  # other switches: model.residual=standard | mhc   model.norm=rmsnorm | dyt
  #                 model.doc_mask=false   train.compile=false (eager, ~20% slower)
  ```

**Memory.** On a single GPU the MoE cells need 30.2 GiB of fp32 weights, grads
and AdamW state. With activation checkpointing that leaves room for 2 × 4096
tokens per micro-batch. Four GPUs shard that state four ways. If you have the
headroom, `--override train.batch_size=4` halves accumulation. Spare memory can
also skip recompute: `--override train.ckpt_skip_last=N` keeps the last N blocks'
activations (each saves ~1/24 of the backward recompute). Find the largest N that
fits with `scripts/profile_step.py --fsdp --compile --override train.ckpt_skip_last=N`,
whose peak plus the printed AdamW share (divided by the GPU count) must stay under ~44 GB.

**Time (estimates until your 200-step timing).** One cell is about 5.3e19
FLOPs for 10B tokens. At 80 to 120 TFLOPS achieved per GPU:

| | 1 GPU | 4 GPUs |
|---|---|---|
| Dense cells | 5 to 8 days | 1.4 to 2.1 days |
| MoE cells (eager expert loop) | about 1.3× dense | about 1.3× dense |

### Before the next full run: the Derf A/B

**Why.** The first `l_diffmoe` run (steps 0 to 1,560) had two linked defects:

- **Dead attention in layers 0 and 1.** It was uniform: entropy 7.384 nats, equal to the uniform value. The std-0.02 embedding reaches Derf at RMS 0.02, and Derf, unlike RMSNorm, does not rescale it.
- **Gradient spikes.** A single Derf shift carried up to 66% of the squared gradient norm, and the pre-clip norm exceeded 10⁶ on 12% of logged steps.

PRs #8 to #11 describe the evidence.

**The A/B.** Three arms of 300 steps each, from scratch, with identical data and seed. Warmup is 2,000 steps, so the learning rate over these 300 steps matches the full schedule exactly. Each arm takes about 3.3 h on 2 GPUs.

```bash
COMMON="--config configs/l_diffmoe.yaml --gpus 0,1 --max_steps 300"
# A: current
python -m src.train $COMMON --override run_name=l_diffmoe_ab_base
# B: fixed
python -m src.train $COMMON --override run_name=l_diffmoe_ab_fix \
    --override model.embed_scale=true \
    --override model.norm_alpha_init_attn=0.8 --override model.norm_alpha_init=0.2 \
    --override train.clip_scalars_separately=true
# C: RMSNorm control
python -m src.train $COMMON --override run_name=l_diffmoe_ab_rms --override model.norm=rmsnorm

# then, on the eval machine
python scripts/eval_l.py --runs checkpoints/l_diffmoe_ab_base checkpoints/l_diffmoe_ab_fix \
    checkpoints/l_diffmoe_ab_rms --suite core attention uncertainty --out_dir eval_results/ab
```

**Pass criteria for arm B.**

1. Layer 0 and 1 attention entropy is clearly below the uniform 7.38 (`attention.per_layer[0:2]`).
2. Fewer than 5% of steps after step 100 have a pre-clip `train/grad_norm` above 10. Every spike that remains is attributed in `grad_spikes.jsonl`.
3. Step-300 loss is no worse than arm A's.

**If it passes:** set `embed_scale`, the two α values and `clip_scalars_separately` in `l_base.yaml`, and restart every L cell.

**If B loses to C:** consider replacing Derf with RMSNorm for the L tier.

## 3 · What gets logged

- **W&B** (`.env`: `WANDB_*`):
  - **Training:** loss, ppl, lr, grad norm, tok/s total and per GPU, step time,
    MFU (from `train.peak_tflops`), epoch and tokens.
  - **Per source:** train loss and sample fraction.
  - **Validation:** NLL, ppl, bits per byte and top-1, globally and per source,
    plus NLL by position (16 buckets).
  - **Routing:** counts, entropy, imbalance and router bias for each MoE layer.
  - **Architecture:** λ for each layer and head, AttnRes depth-weight entropy,
    Derf α/shift and mHC mixing diagonal.
  - **System:** GPU memory and checkpoint events.
  - **Config:** world size, accumulation, git commit and data metadata.
- **MLflow** (`.env`: `MLFLOW_*`): the run itself (flattened config as params,
  tags, scalar metrics) plus a checkpoint lineage. `checkpoints/index.json`
  lists every best, snapshot and last checkpoint with its step, tokens, epoch
  and val NLL. A shared server (`mlflow server --host 0.0.0.0`) lets the
  training box and the eval PC write into the same run.
  `MLFLOW_LOG_CHECKPOINTS=true` also uploads the weights, about 4 GB each.

## 4 · Checkpoints → RTX 3060 eval

In `checkpoints/<cell>/`:

| File | Contents |
|---|---|
| `last.pt` | Full state for resuming (about 32 GB for MoE cells) |
| `best/step*_nll*.pt` | Weights only, about 4 GB; best two by val NLL |
| `snapshots/step*.pt` | Weights only, every 2000 steps, never pruned |

Copy `best/` and `best_index.json`, plus `snapshots/` if you want training
dynamics, and `data_l/{test.bin,val.bin,domain_offsets.json,meta.json}`. The
3060 never needs `last.pt`.

```bash
python scripts/eval_l.py --runs checkpoints/l_dense checkpoints/l_diff \
    checkpoints/l_moe checkpoints/l_diffmoe --data_dir data_l --mlflow
python scripts/eval_l.py --runs checkpoints/l_diffmoe --kind snapshot --suite core position
python scripts/eval_l.py --from_mlflow <run_id> --data_dir data_l      # weights logged to MLflow
```

Each model takes about 4 GB in bf16, with batch 1 × 4096. On Windows without
Triton, document masking uses a dense SDPA mask, so no compiler is needed.
`--suite blimp` adds BLiMP. The `uncertainty` suite needs at least two runs
evaluated in one call.
