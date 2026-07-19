"""Train one config on the BabyLM corpus. Kaggle-first: fp16 AMP + GradScaler (T4 has
no bf16), map-style memmap dataloader (no worker duplication), checkpoint/resume
every ckpt_freq steps (survives Kaggle's 12h session limit and preemption).

Single GPU (default):
    python -m src.train --config configs/a_diffmoe.yaml --data_dir data \
        --wandb --wandb_project my-diffmoe-run

Dual GPU / DDP (opt-in -- launch with torchrun, add --ddp):
    torchrun --standalone --nproc_per_node=2 -m src.train \
        --config configs/a_diffmoe.yaml --data_dir data --ddp --wandb

Without --ddp (and without torchrun) this always runs single-process, exactly
as before -- --ddp is the single switch between the two modes.

Note: MoE experts that receive zero tokens in a step get no gradient that step;
vanilla DDP requires every registered parameter to participate every backward
or it errors. We pass find_unused_parameters=True automatically for ffn=="moe"
configs to guard against this.
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.data import eval_batches, get_batch, load_tokens
from src.eval import bytes_per_token_estimate, evaluate, expert_stats
from src.model import Config, Transformer, count_params
from src.model.moe import MoE


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


def configure_optimizer(model: torch.nn.Module, train_cfg) -> torch.optim.Optimizer:
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
    return torch.optim.AdamW(groups, lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2))


def save_checkpoint(path, model, optimizer, scaler, step, rng_state, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "rng_state": rng_state,
            "best_val": best_val,
        },
        path,
    )


class BestCheckpoints:
    """Keeps only the best `max_keep` checkpoints by val NLL on disk, pruning
    the rest as better ones arrive. Separate from last.pt (always kept, for
    plain resume regardless of quality)."""

    def __init__(self, run_dir: str, max_keep: int = 2):
        self.dir = os.path.join(run_dir, "best")
        self.index_path = os.path.join(run_dir, "best_index.json")
        self.max_keep = max_keep
        os.makedirs(self.dir, exist_ok=True)
        self.entries = []
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.entries = json.load(f)

    def offer(self, step: int, nll: float, model, optimizer, scaler, gen) -> bool:
        if len(self.entries) >= self.max_keep and nll >= self.entries[-1]["nll"]:
            return False
        path = os.path.join(self.dir, f"step{step}_nll{nll:.4f}.pt")
        save_checkpoint(path, model, optimizer, scaler, step, gen.get_state(), nll)
        self.entries.append({"step": step, "nll": nll, "path": path})
        self.entries.sort(key=lambda e: e["nll"])
        while len(self.entries) > self.max_keep:
            worst = self.entries.pop()
            if os.path.exists(worst["path"]):
                os.remove(worst["path"])
        with open(self.index_path, "w") as f:
            json.dump(self.entries, f, indent=2)
        return True

    @property
    def best_nll(self) -> float:
        return self.entries[0]["nll"] if self.entries else float("inf")


def train_time_diagnostics(raw_model: Transformer) -> Dict[str, Any]:
    """Cheap, forward-pass-free (lambda) or forward-pass-piggybacked (expert
    routing, from the last training micro-batch) diagnostics -- safe to compute
    every log_freq step, not just at eval_freq."""
    diag: Dict[str, Any] = {"lambda_means": {}, "lambda_raw": {}, "expert_entropy": {},
                             "expert_imbalance": {}, "expert_counts": {}}
    for i, block in enumerate(raw_model.blocks):
        if hasattr(block.attn, "current_lambda"):
            lam = block.attn.current_lambda().detach().cpu().tolist()
            diag["lambda_raw"][i] = lam
            diag["lambda_means"][i] = sum(lam) / len(lam)
        if block.is_moe and isinstance(block.ffn, MoE) and block.ffn.last_counts is not None:
            entropy, imbalance = expert_stats(block.ffn.last_counts)
            diag["expert_entropy"][i] = entropy
            diag["expert_imbalance"][i] = imbalance
            diag["expert_counts"][i] = block.ffn.last_counts.detach().cpu().tolist()
    return diag


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_report(path, cfg, counts, world_size, tokens_trained, steps_trained,
                  wall_clock_s, final_train_loss, best):
    report = {
        "run_name": cfg.run_name,
        "git_commit": git_commit(),
        "seed": cfg.train.seed,
        "world_size": world_size,
        "model": {
            "attention": cfg.model.attention,
            "ffn": cfg.model.ffn,
            "dim": cfg.model.dim,
            "n_layers": cfg.model.n_layers,
            "n_heads": cfg.model.n_heads,
            "vocab_size": cfg.model.vocab_size,
            "seq_len": cfg.model.seq_len,
            "n_experts": cfg.model.n_experts if cfg.model.ffn == "moe" else None,
            "top_k": cfg.model.top_k if cfg.model.ffn == "moe" else None,
        },
        "params": {k: v for k, v in counts.items()},
        "training": {
            "steps_trained": steps_trained,
            "tokens_trained": tokens_trained,
            "wall_clock_seconds": round(wall_clock_s, 1),
            "final_train_loss": final_train_loss,
            "per_gpu_batch_size": cfg.train.batch_size,
            "accum_steps": cfg.train.accum_steps,
            "effective_global_batch_tokens": (
                cfg.train.batch_size * cfg.model.seq_len * cfg.train.accum_steps * world_size
            ),
        },
        "best_val": best,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default=None, help="overrides config's data_dir")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                     help="ignored when --ddp is set (device is derived from LOCAL_RANK)")
    ap.add_argument("--ddp", action="store_true",
                     help="enable multi-GPU DistributedDataParallel; launch via torchrun")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="diff-moe",
                     help="wandb project name -- change per Kaggle run without editing code")
    ap.add_argument("--max_steps", type=int, default=None, help="override, e.g. for smoke tests")
    args = ap.parse_args()

    local_rank = None
    if args.ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"  # gloo: CPU-only dev/test fallback
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    else:
        rank = 0
        world_size = 1
        device = args.device
    is_main = rank == 0
    device_type = device.split(":")[0]

    cfg = Config.from_yaml(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    set_seed(cfg.train.seed)  # same init on every rank; DDP also broadcasts rank-0 weights on wrap

    run_dir = os.path.join(args.out_dir, cfg.run_name)
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    if args.ddp:
        dist.barrier()
    ckpt_path = os.path.join(run_dir, "last.pt")
    report_path = os.path.join(run_dir, "report.json")
    csv_path = os.path.join(run_dir, "metrics.csv")

    raw_model = Transformer(cfg.model).to(device)
    counts = count_params(raw_model)
    if is_main:
        print(f"[{cfg.run_name}] raw={counts['total']/1e6:.2f}M active={counts['active']/1e6:.2f}M "
              f"world_size={world_size}")

    if args.ddp:
        ddp_kwargs: Dict[str, Any] = {"find_unused_parameters": cfg.model.ffn == "moe"}
        if local_rank is not None:
            ddp_kwargs["device_ids"] = [local_rank]
        model = DistributedDataParallel(raw_model, **ddp_kwargs)
    else:
        model = raw_model

    optimizer = configure_optimizer(raw_model, cfg.train)
    scaler = torch.amp.GradScaler(device_type, enabled=(cfg.train.amp and device_type == "cuda"))
    best_ckpts = BestCheckpoints(run_dir, max_keep=cfg.train.max_best_checkpoints)

    start_step = 0
    best_val = best_ckpts.best_nll
    # per-rank data stream must differ or every GPU trains on identical batches
    gen = torch.Generator().manual_seed(cfg.train.seed + rank)

    resume_path = args.resume or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        best_val = min(best_val, ckpt["best_val"])
        if not args.ddp:
            gen.set_state(ckpt["rng_state"])  # bit-exact resume only guaranteed single-process
        else:
            # multi-rank exact rng state isn't preserved per-rank; reseed decorrelated
            # from the pre-resume stream instead of replaying it (see docs/plan.md).
            gen = torch.Generator().manual_seed(cfg.train.seed + rank + start_step)
        if is_main:
            print(f"resumed from {resume_path} at step {start_step}")

    train_data = load_tokens(os.path.join(cfg.data_dir, "train.bin"))
    val_data = load_tokens(os.path.join(cfg.data_dir, "val.bin"))
    tokenizer_path = os.path.join(cfg.data_dir, "tokenizer.json")
    bytes_per_token = (
        bytes_per_token_estimate(tokenizer_path, os.path.join(cfg.data_dir, "val.bin"))
        if os.path.exists(tokenizer_path)
        else None
    )

    wandb_run = None
    if args.wandb and is_main:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=cfg.run_name, config=vars(cfg.train))

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

    tokens_per_step = cfg.train.batch_size * cfg.model.seq_len * cfg.train.accum_steps * world_size

    model.train()
    t0 = time.time()
    train_start = time.time()
    running_loss = 0.0
    running_aux = 0.0
    running_z = 0.0
    running_count = 0
    last_avg_loss = None

    for step in range(start_step, cfg.train.max_steps):
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step, cfg.train, cfg.train.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        step_loss = 0.0
        step_aux = 0.0
        step_z = 0.0
        for _ in range(cfg.train.accum_steps):
            x, y = get_batch(train_data, cfg.train.batch_size, cfg.model.seq_len, gen, device)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=cfg.train.amp):
                logits, aux_loss, z_loss = model(x)
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
                loss = ce + cfg.model.aux_loss_coef * aux_loss + cfg.model.router_z_coef * z_loss
                loss = loss / cfg.train.accum_steps
            scaler.scale(loss).backward()
            step_loss += ce.item() / cfg.train.accum_steps
            step_aux += aux_loss.item() / cfg.train.accum_steps
            step_z += z_loss.item() / cfg.train.accum_steps

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += step_loss
        running_aux += step_aux
        running_z += step_z
        running_count += 1
        tokens_trained = tokens_per_step * (step + 1)

        if is_main and (step + 1) % cfg.train.log_freq == 0:
            avg_loss = running_loss / running_count
            avg_aux = running_aux / running_count
            avg_z = running_z / running_count
            last_avg_loss = avg_loss
            elapsed = time.time() - t0
            tok_per_sec = (
                cfg.train.batch_size * cfg.model.seq_len * cfg.train.accum_steps * world_size * cfg.train.log_freq
            ) / elapsed
            ppl = math.exp(min(avg_loss, 50))
            diag = train_time_diagnostics(raw_model)
            diag_summary = ""
            if diag["lambda_means"]:
                mean_lam = sum(diag["lambda_means"].values()) / len(diag["lambda_means"])
                diag_summary += f" lambda_avg {mean_lam:.3f}"
            if diag["expert_entropy"]:
                mean_ent = sum(diag["expert_entropy"].values()) / len(diag["expert_entropy"])
                diag_summary += f" expert_entropy_avg {mean_ent:.3f}"
            print(
                f"step {step+1}/{cfg.train.max_steps} tokens {tokens_trained/1e6:.1f}M loss {avg_loss:.4f} "
                f"ppl {ppl:.2f} aux {avg_aux:.4f} z {avg_z:.4f} "
                f"lr {lr:.2e} grad_norm {grad_norm:.2f} tok/s {tok_per_sec:.0f}{diag_summary}"
            )
            csv_writer.writerow(
                [step + 1, tokens_trained, avg_loss, ppl, avg_aux, avg_z, lr, grad_norm.item(), tok_per_sec,
                 "", "", "", ""]
            )
            csv_file.flush()
            if wandb_run:
                import wandb as _wandb

                log = {"train/loss": avg_loss, "train/ppl": ppl, "train/aux_loss": avg_aux,
                       "train/router_z_loss": avg_z, "train/lr": lr,
                       "train/grad_norm": grad_norm.item(), "train/tok_per_sec": tok_per_sec,
                       "train/tokens_trained": tokens_trained}
                for i, mean_lam_i in diag["lambda_means"].items():
                    log[f"train/lambda_mean_L{i}"] = mean_lam_i
                    log[f"train/lambda_hist_L{i}"] = _wandb.Histogram(diag["lambda_raw"][i])
                for i, e in diag["expert_entropy"].items():
                    log[f"train/expert_entropy_L{i}"] = e
                    log[f"train/expert_imbalance_L{i}"] = diag["expert_imbalance"][i]
                    log[f"train/expert_counts_L{i}"] = _wandb.Histogram(
                        np.repeat(np.arange(len(diag["expert_counts"][i])), diag["expert_counts"][i])
                        if sum(diag["expert_counts"][i]) > 0 else [0]
                    )
                if device_type == "cuda":
                    log["train/gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / 1e6
                    log["train/gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / 1e6
                wandb_run.log(log, step=step + 1)
            running_loss = 0.0
            running_aux = 0.0
            running_z = 0.0
            running_count = 0
            t0 = time.time()

        if (step + 1) % cfg.train.eval_freq == 0 or (step + 1) == cfg.train.max_steps:
            if is_main:
                batches = eval_batches(val_data, cfg.train.batch_size, cfg.model.seq_len, cfg.train.eval_batches, device)
                result = evaluate(raw_model, batches, bytes_per_token)
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
                if wandb_run:
                    import wandb as _wandb

                    log = {"val/nll": result.nll, "val/ppl": result.perplexity, "val/top1": result.top1_acc}
                    if result.bits_per_byte is not None:
                        log["val/bits_per_byte"] = result.bits_per_byte
                    for i, e in result.expert_entropy.items():
                        log[f"val/expert_entropy_L{i}"] = e
                        log[f"val/expert_imbalance_L{i}"] = result.expert_imbalance[i]
                        ecounts = result.expert_counts[i]
                        log[f"val/expert_counts_L{i}"] = _wandb.Histogram(
                            np.repeat(np.arange(len(ecounts)), ecounts) if sum(ecounts) > 0 else [0]
                        )
                    for i, lam in result.lambda_values.items():
                        log[f"val/lambda_mean_L{i}"] = sum(lam) / len(lam)
                        log[f"val/lambda_hist_L{i}"] = _wandb.Histogram(lam)
                    wandb_run.log(log, step=step + 1)

                best_ckpts.offer(step + 1, result.nll, raw_model, optimizer, scaler, gen)
                best_val = best_ckpts.best_nll
                save_checkpoint(ckpt_path, raw_model, optimizer, scaler, step + 1, gen.get_state(), best_val)
                model.train()
            if args.ddp:
                dist.barrier()
        elif (step + 1) % cfg.train.ckpt_freq == 0 or (step + 1) == cfg.train.max_steps:
            if is_main:
                save_checkpoint(ckpt_path, raw_model, optimizer, scaler, step + 1, gen.get_state(), best_val)
            if args.ddp:
                dist.barrier()

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
            wall_clock_s, last_avg_loss, best_summary,
        )
        print(f"[{cfg.run_name}] training complete. report:")
        print(json.dumps(report, indent=2))
        if csv_file:
            csv_file.close()
        if wandb_run:
            wandb_run.finish()

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
