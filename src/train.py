"""Train one config. Single GPU, DDP, or FSDP2 across 1-4+ GPUs.

Pick GPUs by id (relaunches itself under torchrun; FSDP2 + bf16 by default):
    python -m src.train --config configs/l_diffmoe.yaml --gpus 0,1,2,3
    python -m src.train --config configs/l_diffmoe.yaml --gpus 1          # one GPU, same code path

Incremental ablations without new YAML files:
    python -m src.train --config configs/l_diffmoe.yaml --gpus 0,1 \
        --override model.xsa=false --override model.residual=standard

Legacy paths are unchanged:
    python -m src.train --config configs/a_diffmoe.yaml                   # one device, fp16 + GradScaler
    torchrun --standalone --nproc_per_node=2 -m src.train --config ... --ddp

Parallelism: --parallel fsdp (FSDP2 fully_shard per block, bf16 params, fp32
grad reduce and fp32 AdamW state) or ddp. FSDP needs train.precision=bf16.
With train.global_batch_tokens set, accumulation is derived from the GPU count,
so 1 GPU and 4 GPUs run the identical optimizer batch and LR schedule.

Checkpoints in <out_dir>/<run_name>/:
    last.pt                 full training state (resume); portable across GPU counts under FSDP
    best/step*_nll*.pt      weights only (bf16 matrices), best-k by val NLL -- copy these to the eval PC
    snapshots/step*.pt      weights only, every train.snapshot_freq steps, never pruned
"""

import argparse
import contextlib
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.data import SourceSampler, eval_windows, get_batch, load_tokens, window_batches
from src.eval import bytes_per_token_estimate, evaluate, expert_stats, full_tensor, lambda_of
from src.model import Config, Transformer, count_params
from src.model.attention import PointwiseNorm
from src.model.moe import RoutedMoE
from src.tracking import (MLflowTracker, arch_metrics, env_flag, init_wandb,
                          load_training_env, routing_metrics)

POSITION_BUCKETS = 16


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step: int, cfg, peak_lr: float) -> float:
    if step < cfg.warmup_steps:
        return peak_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    min_lr = peak_lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def configure_optimizer(model: torch.nn.Module, train_cfg, fused: bool = False) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "embed" not in name and "gate.weight" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused avoids foreach's full-size temporaries, which matter for a 2B model on one GPU
    extra = {"fused": True} if fused else {}
    return torch.optim.AdamW(groups, lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), **extra)


def resolve_accum(train_cfg, seq_len: int, world_size: int) -> int:
    """Per-rank accumulation. global_batch_tokens keeps the optimizer batch fixed
    across GPU counts; otherwise accum_steps is used per rank (legacy)."""
    if not train_cfg.global_batch_tokens:
        return train_cfg.accum_steps
    per = train_cfg.batch_size * seq_len * world_size
    if train_cfg.global_batch_tokens % per:
        raise SystemExit(f"global_batch_tokens={train_cfg.global_batch_tokens} is not divisible by "
                         f"batch_size*seq_len*world_size={per}; change batch_size or the GPU count")
    return train_cfg.global_batch_tokens // per


def export_weights(model_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Weights-only payload for evaluation: big matrices in bf16, small tensors
    (norms, lambdas, router, biases) kept exact."""
    return {k: v.to(torch.bfloat16) if v.is_floating_point() and v.numel() >= 65536 else v
            for k, v in model_sd.items()}


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a crash mid-write never corrupts the previous file


def save_checkpoint(path, model_sd, optim_sd, scaler, step, rng_state, best_val, model_config=None, **extra):
    atomic_save(
        {
            "model": model_sd,
            "optimizer": optim_sd,
            "scaler": scaler.state_dict(),
            "step": step,
            "rng_state": rng_state,
            "best_val": best_val,
            "model_config": asdict(model_config) if model_config is not None else None,
            **extra,
        },
        path,
    )


def gather_state(raw_model, optimizer, fsdp: bool):
    """Full (unsharded) model + optimizer state. Collective under FSDP: every rank
    calls it; rank 0 receives the tensors on CPU."""
    if not fsdp:
        return raw_model.state_dict(), optimizer.state_dict()
    from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict,
                                                         get_optimizer_state_dict)
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_model_state_dict(raw_model, options=opts), get_optimizer_state_dict(raw_model, optimizer, options=opts)


def load_state(raw_model, optimizer, ckpt, fsdp: bool, is_main: bool):
    if not fsdp:
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return
    from torch.distributed.checkpoint.state_dict import (StateDictOptions, set_model_state_dict,
                                                         set_optimizer_state_dict)
    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    set_model_state_dict(raw_model, ckpt["model"] if is_main else {}, options=opts)
    set_optimizer_state_dict(raw_model, optimizer, ckpt["optimizer"] if is_main else {}, options=opts)


class BestCheckpoints:
    """Keeps only the best `max_keep` weights-only checkpoints by val NLL,
    pruning the rest as better ones arrive. Separate from last.pt (always kept,
    for plain resume regardless of quality)."""

    def __init__(self, run_dir: str, max_keep: int = 2, model_config=None):
        self.dir = os.path.join(run_dir, "best")
        self.index_path = os.path.join(run_dir, "best_index.json")
        self.max_keep = max_keep
        self.model_config = model_config
        os.makedirs(self.dir, exist_ok=True)
        self.entries = []
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.entries = json.load(f)

    def offer(self, step: int, nll: float, weights: Dict[str, torch.Tensor], info: Dict[str, Any]):
        """Returns the saved path, or None when nll does not make the top-k."""
        if len(self.entries) >= self.max_keep and nll >= self.entries[-1]["nll"]:
            return None
        path = os.path.join(self.dir, f"step{step}_nll{nll:.4f}.pt")
        atomic_save({"model": weights, "step": step, "best_val": nll, "format": "weights",
                     "model_config": asdict(self.model_config) if self.model_config else None, **info}, path)
        self.entries.append({"step": step, "nll": nll, "path": path})
        self.entries.sort(key=lambda e: e["nll"])
        while len(self.entries) > self.max_keep:
            worst = self.entries.pop()
            if os.path.exists(worst["path"]):
                os.remove(worst["path"])
        with open(self.index_path, "w") as f:
            json.dump(self.entries, f, indent=2)
        return path

    @property
    def best_nll(self) -> float:
        return self.entries[0]["nll"] if self.entries else float("inf")


def train_time_diagnostics(raw_model: Transformer) -> Dict[str, Any]:
    """Cheap, forward-pass-free (lambda, norm scalars) or forward-pass-piggybacked
    (routing, AttnRes depth weights, mHC mixing) diagnostics. Gathers FSDP shards,
    so under FSDP every rank must call it, in the same order."""
    diag: Dict[str, Any] = {
        "lambda_means": {}, "lambda_raw": {}, "expert_entropy": {}, "expert_imbalance": {},
        "expert_counts": {}, "router_bias": {}, "attnres_weights": {}, "norm_params": {},
        "mhc_res_diag": {},
    }
    for i, block in enumerate(raw_model.blocks):
        if hasattr(block.attn, "current_lambda"):
            lam = lambda_of(block.attn)
            diag["lambda_raw"][i] = lam
            diag["lambda_means"][i] = sum(lam) / len(lam)
        if (
            block.is_moe
            and isinstance(block.ffn, RoutedMoE)
            and block.ffn.last_counts is not None
        ):
            entropy, imbalance = expert_stats(block.ffn.last_counts)
            diag["expert_entropy"][i] = entropy
            diag["expert_imbalance"][i] = imbalance
            diag["expert_counts"][i] = block.ffn.last_counts.detach().cpu().tolist()
            if hasattr(block.ffn.gate, "expert_bias"):
                diag["router_bias"][i] = block.ffn.gate.expert_bias.detach().cpu().tolist()
        for part in ("attn", "ffn"):
            norm = getattr(block, f"{part}_norm")
            if isinstance(norm, PointwiseNorm):
                shift = full_tensor(norm.shift.detach()).item() if norm.shift is not None else None
                diag["norm_params"][f"L{i}_{part}"] = (full_tensor(norm.alpha.detach()).item(), shift)
            res = getattr(block, f"{part}_res", None)
            if res is not None and res.last_weights is not None:
                diag["attnres_weights"][f"L{i}_{part}"] = res.last_weights.float().cpu().tolist()
            mhc = getattr(block, f"{part}_mhc", None)
            if mhc is not None and mhc.last_res_diag is not None:
                diag["mhc_res_diag"][f"L{i}_{part}"] = float(mhc.last_res_diag)
    if raw_model.final_res is not None and raw_model.final_res.last_weights is not None:
        diag["attnres_weights"]["final"] = raw_model.final_res.last_weights.float().cpu().tolist()
    return diag


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_report(path, cfg, counts, world_size, tokens_trained, steps_trained,
                  wall_clock_s, final_train_loss, best, accum_steps=None):
    accum_steps = accum_steps or cfg.train.accum_steps
    report = {
        "run_name": cfg.run_name,
        "git_commit": git_commit(),
        "seed": cfg.train.seed,
        "world_size": world_size,
        "model": asdict(cfg.model),
        "params": {k: v for k, v in counts.items()},
        "training": {
            "steps_trained": steps_trained,
            "tokens_trained": tokens_trained,
            "wall_clock_seconds": round(wall_clock_s, 1),
            "final_train_loss": final_train_loss,
            "per_gpu_batch_size": cfg.train.batch_size,
            "accum_steps": accum_steps,
            "effective_global_batch_tokens": (
                cfg.train.batch_size * cfg.model.seq_len * accum_steps * world_size
            ),
        },
        "best_val": best,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args(argv=None):
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env_file", default=None)
    env_args, _ = env_parser.parse_known_args(argv)
    load_training_env(env_args.env_file)
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=None, help="defaults to .env in the repository root")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE",
                     help="repeatable, e.g. --override model.xsa=false --override train.batch_size=4")
    ap.add_argument("--data_dir", type=str, default=None, help="overrides config's data_dir")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                     help="ignored when distributed (device is derived from LOCAL_RANK)")
    ap.add_argument("--gpus", type=str, default=None,
                     help="comma-separated GPU ids, e.g. 0,1,2,3; relaunches under torchrun")
    ap.add_argument("--parallel", choices=["fsdp", "ddp"], default=None,
                     help="multi-process strategy; default fsdp for bf16 configs, ddp otherwise")
    ap.add_argument("--ddp", action="store_true",
                     help="legacy switch: DistributedDataParallel under torchrun (same as --parallel ddp)")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=env_flag("WANDB_ENABLED"),
                     help="enable W&B; defaults to WANDB_ENABLED from the environment/.env")
    ap.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT") or "diff-moe",
                     help="wandb project name -- change per Kaggle run without editing code")
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY") or None)
    ap.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                     default=os.environ.get("WANDB_MODE") or "online")
    ap.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=env_flag("MLFLOW_ENABLED"),
                     help="enable MLflow run/checkpoint tracking; defaults to MLFLOW_ENABLED")
    ap.add_argument("--mlflow_experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME") or "diff-moe")
    ap.add_argument("--mlflow_log_checkpoints", action=argparse.BooleanOptionalAction,
                     default=env_flag("MLFLOW_LOG_CHECKPOINTS"),
                     help="also upload best/snapshot weights as MLflow artifacts (~4 GB each at L size)")
    ap.add_argument("--max_steps", type=int, default=None, help="override, e.g. for smoke tests")
    args = ap.parse_args(argv)
    if args.wandb_mode not in ("online", "offline", "disabled"):
        ap.error("WANDB_MODE must be online, offline, or disabled")
    return args


def launch_workers(gpus: str, argv) -> None:
    """Re-run this module under torchrun on exactly the requested GPUs."""
    ids = [g.strip() for g in gpus.split(",") if g.strip()]
    if not ids:
        raise SystemExit("--gpus needs at least one id, e.g. --gpus 0 or --gpus 0,1,2,3")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    from torch.distributed.run import main as torchrun

    torchrun(["--standalone", f"--nproc_per_node={len(ids)}", "-m", "src.train", *argv])


def setup_distributed(args):
    """Returns (distributed, rank, world_size, local_rank, device)."""
    if not (args.ddp or "LOCAL_RANK" in os.environ):
        return False, 0, 1, None, args.device
    local_rank = None
    if torch.cuda.is_available():
        # set the device BEFORE init and tell the process group which device this
        # rank owns -- avoids NCCL's "Guessing device ID based on global rank"
        # warning (and the hang it warns about on heterogeneous mappings)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        try:
            dist.init_process_group(backend="nccl", device_id=torch.device(device))
        except TypeError:  # torch < 2.3 has no device_id kwarg
            dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")  # CPU-only dev/test fallback
        device = "cpu"
    return True, dist.get_rank(), dist.get_world_size(), local_rank, device


def main():
    args = parse_args()
    if args.gpus and "LOCAL_RANK" not in os.environ:
        launch_workers(args.gpus, sys.argv[1:])
        return

    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    is_main = rank == 0
    device_type = device.split(":")[0]

    cfg = Config.from_yaml(args.config, args.override)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    parallel = None
    if distributed:
        parallel = "ddp" if args.ddp else args.parallel or ("fsdp" if cfg.train.precision == "bf16" else "ddp")
    fsdp = parallel == "fsdp"
    if fsdp and not (cfg.train.amp and cfg.train.precision == "bf16"):
        raise SystemExit("FSDP runs need train.amp=true and train.precision=bf16 (or use --parallel ddp)")
    accum_steps = resolve_accum(cfg.train, cfg.model.seq_len, world_size)
    set_seed(cfg.train.seed)  # same init on every rank; DDP/FSDP also start from identical weights

    run_dir = os.path.join(args.out_dir, cfg.run_name)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    if distributed:
        dist.barrier()
    ckpt_path = os.path.join(run_dir, "last.pt")
    report_path = os.path.join(run_dir, "report.json")
    csv_path = os.path.join(run_dir, "metrics.csv")

    # Build directly on the GPU: identical seeded init on every rank, and FSDP
    # frees the non-local shards right after wrapping.
    with torch.device(device):
        raw_model = Transformer(cfg.model)
    counts = count_params(raw_model)
    raw_model.gradient_checkpointing = cfg.train.activation_checkpointing
    if cfg.train.compile:
        raw_model.compile_blocks()
    if is_main:
        print(f"[{cfg.run_name}] raw={counts['total']/1e6:.2f}M active={counts['active']/1e6:.2f}M "
              f"world_size={world_size} parallel={parallel or 'none'} accum={accum_steps}")

    if fsdp:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in raw_model.blocks:
            fully_shard(block, mp_policy=mp)
        fully_shard(raw_model, mp_policy=mp)
        model = raw_model
    elif distributed:
        ddp_kwargs: Dict[str, Any] = {
            "find_unused_parameters": cfg.model.ffn in ("moe", "stable_latent_moe")
        }
        if local_rank is not None:
            ddp_kwargs["device_ids"] = [local_rank]
        model = DistributedDataParallel(raw_model, **ddp_kwargs)
    else:
        model = raw_model

    optimizer = configure_optimizer(raw_model, cfg.train, fused=device_type == "cuda")
    use_fp16 = cfg.train.amp and cfg.train.precision == "fp16"
    scaler = torch.amp.GradScaler(device_type, enabled=(use_fp16 and device_type == "cuda"))
    amp_dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float16
    # FSDP's MixedPrecisionPolicy already runs the forward in bf16.
    autocast_on = cfg.train.amp and not fsdp
    best_ckpts = BestCheckpoints(
        run_dir,
        max_keep=cfg.train.max_best_checkpoints,
        model_config=cfg.model,
    )

    start_step = 0
    best_val = best_ckpts.best_nll
    # per-rank data stream must differ or every GPU trains on identical batches
    gen = torch.Generator().manual_seed(cfg.train.seed + rank)

    resume_path = args.resume or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_path:
        # Under FSDP only rank 0 reads the file; weights and optimizer state are broadcast.
        ckpt = torch.load(resume_path, map_location="cpu" if fsdp else device, mmap=fsdp,
                          weights_only=False) if (is_main or not fsdp) else None
        load_state(raw_model, optimizer, ckpt, fsdp, is_main)
        meta = [None]
        if ckpt is not None:
            scaler.load_state_dict(ckpt["scaler"])
            meta = [{"step": ckpt["step"], "best_val": ckpt["best_val"], "rng_state": ckpt["rng_state"]}]
        if fsdp:
            dist.broadcast_object_list(meta, src=0)
        start_step = meta[0]["step"]
        best_val = min(best_val, meta[0]["best_val"])
        if not distributed:
            gen.set_state(meta[0]["rng_state"])  # bit-exact resume only guaranteed single-process
        else:
            # multi-rank exact rng state isn't preserved per-rank; reseed decorrelated
            # from the pre-resume stream instead of replaying it (see docs/plan.md).
            gen = torch.Generator().manual_seed(cfg.train.seed + rank + start_step)
        del ckpt
        if is_main:
            print(f"resumed from {resume_path} at step {start_step}")

    train_path = os.path.join(cfg.data_dir, "train.bin")
    val_path = os.path.join(cfg.data_dir, "val.bin")
    missing = [p for p in (train_path, val_path) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"[{cfg.run_name}] tokenized data not found: {missing}\n"
            f"data_dir is '{cfg.data_dir}'. Run the data pipeline first:\n"
            f"  python -m src.data.train_tokenizer --vocab_size {cfg.model.vocab_size} "
            f"--out {cfg.data_dir}/tokenizer.json --track <strict-small|strict>\n"
            f"  python -m src.data.prepare --tokenizer {cfg.data_dir}/tokenizer.json "
            f"--out_dir {cfg.data_dir} --track <strict-small|strict>\n"
            f"  (L tier: python -m src.data.prepare_hf --config {args.config})"
        )
    # Config vocab must cover every id in the data. A mismatch here is the
    # original repo's fatal bug (tokenizer vocab != config vocab -> embedding
    # index out of range); fail now rather than partway through a paid run.
    meta_path = os.path.join(cfg.data_dir, "meta.json")
    data_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data_meta = json.load(f)
        max_id = data_meta.get("max_token_id")
        if max_id is not None and max_id >= cfg.model.vocab_size:
            raise SystemExit(
                f"[{cfg.run_name}] vocab_size mismatch: data in '{cfg.data_dir}' contains "
                f"token id {max_id}, but the config declares vocab_size={cfg.model.vocab_size}.\n"
                f"Set vocab_size to at least {max_id + 1} (data was tokenized with "
                f"{data_meta.get('tokenizer_spec')}, vocab {data_meta.get('vocab_size')})."
            )
    if cfg.model.doc_mask and data_meta.get("eot_id", cfg.model.eos_id) != cfg.model.eos_id:
        raise SystemExit(f"[{cfg.run_name}] model.eos_id={cfg.model.eos_id} but the data's EOS id is "
                         f"{data_meta['eot_id']}; document masking would split on the wrong token")

    train_data = load_tokens(train_path)
    val_data = load_tokens(val_path)
    bytes_per_token = bytes_per_token_estimate(cfg.data_dir, val_path)
    sampler = None
    docs_path = os.path.join(cfg.data_dir, "train_docs.npz")
    if cfg.train.source_weights:
        if not os.path.exists(docs_path):
            raise SystemExit(f"train.source_weights needs {docs_path}; prepare data with src.data.prepare_hf")
        sampler = SourceSampler(docs_path, len(train_data), cfg.model.seq_len, cfg.train.source_weights)
    source_names = sampler.names if sampler else []

    # Fixed val windows (per source when the data has spans), equal count per rank.
    spans = None
    offsets_path = os.path.join(cfg.data_dir, "domain_offsets.json")
    if os.path.exists(offsets_path):
        with open(offsets_path) as f:
            spans = json.load(f).get("val")
        if spans is not None and len(spans) < 2:
            spans = None  # one span == the whole stream; keep the legacy windows
    val_windows = eval_windows(len(val_data), cfg.model.seq_len, cfg.train.eval_batches * cfg.train.batch_size, spans)
    per_round = cfg.train.batch_size * world_size
    val_windows = val_windows[: len(val_windows) - len(val_windows) % per_round] or val_windows[:per_round]
    val_names = sorted({name for _, name in val_windows}) if spans else []
    my_val_windows = val_windows[rank::world_size]

    tokens_per_step = cfg.train.batch_size * cfg.model.seq_len * accum_steps * world_size
    gpu_name = torch.cuda.get_device_name(device) if device_type == "cuda" else "cpu"
    extra_config = {
        "accum_steps": accum_steps, "tokens_per_step": tokens_per_step,
        "effective_global_batch_tokens": tokens_per_step, "parallel": parallel or "none",
        "gpu": gpu_name, "torch": torch.__version__, "git_commit": git_commit(),
        "data_meta": {k: data_meta.get(k) for k in ("tokens", "sources", "tokenizer_spec", "vocab_size", "eot_id")},
        "train_windows_source_weighted": bool(sampler), "val_windows": len(val_windows),
    }
    # Model FLOPs per token, standard-attention estimate (6N + causal attention).
    head_params = cfg.model.vocab_size * cfg.model.dim
    flops_per_token = 6 * (counts["non_embed_active"] + head_params) + \
        6 * cfg.model.n_layers * cfg.model.seq_len * cfg.model.dim

    wandb_run = None
    mlf = None
    if is_main:
        if args.wandb:
            wandb_run = init_wandb(args, cfg, counts, world_size, run_dir, resuming=bool(resume_path),
                                   extra_config=extra_config)
        if args.mlflow:
            mlf = MLflowTracker(run_dir, args.mlflow_experiment, cfg.run_name, resuming=bool(resume_path),
                                log_checkpoints=args.mlflow_log_checkpoints)
            mlf.log_params({"model": asdict(cfg.model), "train": asdict(cfg.train), "data_dir": cfg.data_dir,
                            "params": counts, **{k: v for k, v in extra_config.items() if k != "git_commit"}})
            mlf.set_tags({"run_name": cfg.run_name, "git_commit": extra_config["git_commit"],
                          "attention": cfg.model.attention, "ffn": cfg.model.ffn,
                          "residual": cfg.model.residual, "norm": cfg.model.norm, "xsa": cfg.model.xsa,
                          "world_size": world_size, "gpu": gpu_name})
            mlf.log_artifact(args.config, "config")
            mlf.log_artifact(meta_path, "data")

    csv_writer = csv_file = None
    if is_main:
        csv_new = not os.path.exists(csv_path)
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if csv_new:
            csv_writer.writerow(
                ["step", "tokens_trained", "train_loss", "train_ppl", "train_aux_loss", "train_router_z_loss",
                 "lr", "grad_norm", "tok_per_sec", "val_nll", "val_ppl", "val_bits_per_byte", "val_top1"]
            )

    def reduce_mean(values):
        t = torch.tensor(values, dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(t)
            t /= world_size
        return t.tolist()

    model.train()
    train_start = time.time()
    running_time = 0.0
    skipped_steps = 0
    running = np.zeros(3)  # loss, aux, z
    running_count = 0
    src_loss = torch.zeros(len(source_names), dtype=torch.float64, device=device)
    src_n = torch.zeros(len(source_names), dtype=torch.float64, device=device)
    last_avg_loss = None
    epoch_tokens = max(1, len(train_data))

    for step in range(start_step, cfg.train.max_steps):
        step_start = time.perf_counter()
        wandb_metrics = {}
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step, cfg.train, cfg.train.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        step_sums = torch.zeros(3, device=device)
        for micro in range(accum_steps):
            last_micro = micro == accum_steps - 1
            starts = src = None
            if sampler:
                starts, src = sampler.sample(cfg.train.batch_size, gen)
            x, y = get_batch(train_data, cfg.train.batch_size, cfg.model.seq_len, gen, device, starts)
            if fsdp:
                model.set_requires_gradient_sync(last_micro)
            sync = model.no_sync() if (parallel == "ddp" and not last_micro) else contextlib.nullcontext()
            with sync:
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_on):
                    logits, aux_loss, z_loss = model(x)
                    tok_ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), reduction="none")
                    ce = tok_ce.mean()
                    loss = ce + cfg.model.aux_loss_coef * aux_loss + cfg.model.router_z_coef * z_loss
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            step_sums += torch.stack([ce.detach(), aux_loss.detach().float(), z_loss.detach().float()]) / accum_steps
            if sampler:
                idx = torch.as_tensor(src, device=device)
                src_loss.index_add_(0, idx, tok_ce.detach().view(len(src), -1).mean(1).double())
                src_n.index_add_(0, idx, torch.ones(len(src), dtype=torch.float64, device=device))

        scaler.unscale_(optimizer)
        grad_norm = full_tensor(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.train.grad_clip))
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped_steps += int(scaler.get_scale() < previous_scale)
        # Biases derived from optimizer step t are committed only after every
        # accumulated micro-batch, so they first affect step t+1.
        raw_model.update_quantile_balancing()

        running += np.asarray(step_sums.tolist())
        running_count += 1
        running_time += time.perf_counter() - step_start
        tokens_trained = tokens_per_step * (step + 1)
        epoch = tokens_trained / epoch_tokens

        if (step + 1) % cfg.train.log_freq == 0 or (step + 1) == cfg.train.max_steps:
            avg_loss, avg_aux, avg_z = reduce_mean((running / running_count).tolist())
            if distributed and sampler:
                dist.all_reduce(src_loss)
                dist.all_reduce(src_n)
            diag = train_time_diagnostics(raw_model)  # collective under FSDP: all ranks
            if is_main:
                last_avg_loss = avg_loss
                tok_per_sec = tokens_per_step * running_count / max(running_time, 1e-9)
                ppl = math.exp(min(avg_loss, 50))
                diag_summary = ""
                if diag["lambda_means"]:
                    mean_lam = sum(diag["lambda_means"].values()) / len(diag["lambda_means"])
                    diag_summary += f" lambda_avg {mean_lam:.3f}"
                if diag["expert_entropy"]:
                    mean_ent = sum(diag["expert_entropy"].values()) / len(diag["expert_entropy"])
                    diag_summary += f" expert_entropy_avg {mean_ent:.3f}"
                print(
                    f"step {step+1}/{cfg.train.max_steps} tokens {tokens_trained/1e6:.1f}M epoch {epoch:.3f} "
                    f"loss {avg_loss:.4f} ppl {ppl:.2f} aux {avg_aux:.4f} z {avg_z:.4f} "
                    f"lr {lr:.2e} grad_norm {grad_norm:.2f} tok/s {tok_per_sec:.0f}{diag_summary}"
                )
                csv_writer.writerow(
                    [step + 1, tokens_trained, avg_loss, ppl, avg_aux, avg_z, lr, grad_norm.item(), tok_per_sec,
                     "", "", "", ""]
                )
                csv_file.flush()
                log = {"train/loss": avg_loss, "train/ppl": ppl, "train/aux_loss": avg_aux,
                       "train/router_z_loss": avg_z, "train/lr": lr,
                       "train/grad_norm": grad_norm.item(), "train/tok_per_sec": tok_per_sec,
                       "train/tok_per_sec_per_gpu": tok_per_sec / world_size,
                       "train/step_time_s": running_time / running_count,
                       "train/tokens_trained": tokens_trained, "train/epoch": epoch,
                       "train/objective": avg_loss + cfg.model.aux_loss_coef * avg_aux + cfg.model.router_z_coef * avg_z,
                       "train/amp_scale": scaler.get_scale(),
                       "train/skipped_optimizer_steps": skipped_steps}
                if cfg.train.peak_tflops:
                    log["train/mfu"] = flops_per_token * tok_per_sec / (cfg.train.peak_tflops * 1e12 * world_size)
                for j, name in enumerate(source_names):
                    if src_n[j] > 0:
                        log[f"train/loss_src/{name}"] = (src_loss[j] / src_n[j]).item()
                        log[f"train/sample_frac_src/{name}"] = (src_n[j] / src_n.sum()).item()
                log.update(arch_metrics("train", diag))
                if wandb_run:
                    log.update(routing_metrics("train", counts=diag["expert_counts"],
                        entropy=diag["expert_entropy"], imbalance=diag["expert_imbalance"],
                        biases=diag["router_bias"], lambdas=diag["lambda_raw"]))
                wandb_metrics.update(log)
            running[:] = 0.0
            running_count = 0
            running_time = 0.0
            src_loss.zero_()
            src_n.zero_()

        is_eval = (step + 1) % cfg.train.eval_freq == 0 or (step + 1) == cfg.train.max_steps
        is_ckpt = (step + 1) % cfg.train.ckpt_freq == 0
        is_snapshot = bool(cfg.train.snapshot_freq) and (step + 1) % cfg.train.snapshot_freq == 0
        result = None
        if is_eval:
            batches = window_batches(val_data, my_val_windows, cfg.train.batch_size, cfg.model.seq_len, device)
            result = evaluate(raw_model, batches, bytes_per_token, source_names=val_names,
                              position_buckets=POSITION_BUCKETS if cfg.model.seq_len % POSITION_BUCKETS == 0 else 0)
            model.train()
        if is_eval or is_ckpt or is_snapshot:
            model_sd, optim_sd = gather_state(raw_model, optimizer, fsdp)  # collective under FSDP
            if is_main:
                info = {"step": step + 1, "tokens": tokens_trained, "epoch": epoch}
                if result is not None:
                    info["val_nll"] = result.nll
                    print(
                        f"[eval] step {step+1} nll {result.nll:.4f} ppl {result.perplexity:.2f} "
                        f"bpb {result.bits_per_byte} top1 {result.top1_acc:.4f} "
                        f"entropy {result.expert_entropy}"
                    )
                    csv_writer.writerow(
                        [step + 1, tokens_trained, "", "", "", "", "", "", "", result.nll, result.perplexity,
                         result.bits_per_byte, result.top1_acc]
                    )
                    csv_file.flush()
                    log = {"val/nll": result.nll, "val/ppl": result.perplexity,
                           "val/top1": result.top1_acc, "val/bits_per_token": result.bits_per_token}
                    if result.bits_per_byte is not None:
                        log["val/bits_per_byte"] = result.bits_per_byte
                    for name, r in result.per_source.items():
                        log[f"val/nll_src/{name}"] = r["nll"]
                        if r["bits_per_byte"] is not None:
                            log[f"val/bits_per_byte_src/{name}"] = r["bits_per_byte"]
                    for b, v in enumerate(result.nll_by_position):
                        log[f"val/nll_pos/bucket{b:02d}"] = v
                    if wandb_run:
                        log.update(routing_metrics("val", counts=result.expert_counts,
                            entropy=result.expert_entropy, imbalance=result.expert_imbalance,
                            biases=result.router_bias, lambdas=result.lambda_values))
                    wandb_metrics.update(log)
                    weights = export_weights(model_sd)
                    best_path = best_ckpts.offer(step + 1, result.nll, weights, info)
                    best_val = best_ckpts.best_nll
                    if best_path:
                        wandb_metrics.update({"ckpt/best_step": step + 1, "ckpt/best_val_nll": result.nll})
                        if mlf:
                            mlf.log_checkpoint("best", best_path, info)
                if is_snapshot:
                    snap = os.path.join(run_dir, "snapshots", f"step{step + 1}.pt")
                    atomic_save({"model": export_weights(model_sd), "format": "weights",
                                 "model_config": asdict(cfg.model), **info}, snap)
                    wandb_metrics["ckpt/snapshot_step"] = step + 1
                    if mlf:
                        mlf.log_checkpoint("snapshot", snap, info)
                save_checkpoint(ckpt_path, model_sd, optim_sd, scaler, step + 1, gen.get_state(),
                                best_val, cfg.model, tokens=tokens_trained, epoch=epoch,
                                parallel=parallel or "none")
                wandb_metrics["ckpt/last_step"] = step + 1
            del model_sd, optim_sd
            if distributed:
                dist.barrier()

        if is_main and wandb_metrics:
            wandb_metrics.update(optimizer_step=step + 1)
            wandb_metrics["train/tokens_trained"] = tokens_trained
            if device_type == "cuda":
                wandb_metrics["train/gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / 1e6
                wandb_metrics["train/gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / 1e6
                wandb_metrics["train/gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
                wandb_metrics["train/gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1e6
            if wandb_run:
                wandb_run.log(wandb_metrics)
            if mlf:
                mlf.log_metrics({k: v for k, v in wandb_metrics.items() if "_L" not in k}, step + 1)

    if is_main:
        wall_clock_s = time.time() - train_start
        total_tokens = tokens_per_step * cfg.train.max_steps
        best = best_ckpts.entries[0] if best_ckpts.entries else {"step": None, "nll": None}
        best_summary = {
            "step": best["step"], "nll": best["nll"],
            "ppl": math.exp(min(best["nll"], 50)) if best["nll"] is not None else None,
        }
        report = write_report(
            report_path, cfg, counts, world_size, total_tokens, cfg.train.max_steps,
            wall_clock_s, last_avg_loss, best_summary, accum_steps,
        )
        print(f"[{cfg.run_name}] training complete. report:")
        print(json.dumps(report, indent=2))
        if csv_file:
            csv_file.close()
        if wandb_run:
            wandb_run.summary.update({
                "best_val/nll": best_summary["nll"], "best_val/ppl": best_summary["ppl"],
                "best_val/step": best_summary["step"], "training/steps": cfg.train.max_steps,
                "training/tokens": total_tokens, "training/session_wall_clock_seconds": wall_clock_s,
                "training/final_loss": last_avg_loss,
                "training/session_skipped_optimizer_steps": skipped_steps,
            })
            wandb_run.finish()
        if mlf:
            mlf.log_checkpoint("last", ckpt_path, {"step": cfg.train.max_steps, "tokens": total_tokens,
                                                    "epoch": total_tokens / epoch_tokens})
            mlf.log_artifact(report_path, "report")
            mlf.log_artifact(csv_path, "report")
            mlf.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
