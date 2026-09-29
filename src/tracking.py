"""Environment loading and W&B setup, with no import-time login or side effects."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid

import numpy as np


def load_training_env(path=None):
    from dotenv import load_dotenv

    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if path and not env_path.is_file():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    # Shell/job settings take precedence; never print or log environment contents.
    load_dotenv(env_path, override=False)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name} must be true or false")


def init_wandb(args, cfg, counts, world_size, run_dir, *, resuming=False):
    """Call only on rank zero. W&B reads WANDB_API_KEY before any login prompt."""
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        raise RuntimeError("W&B is enabled; install dependencies with python -m pip install -r requirements.txt") from None

    state_path = Path(run_dir) / "wandb_run.json"
    run_id = os.environ.get("WANDB_RUN_ID") or None
    entity = args.wandb_entity or None
    if resuming and state_path.exists() and run_id is None:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["project"] != args.wandb_project or (entity and state["entity"] != entity):
            raise ValueError("W&B project/entity differs from the saved run; restore its settings or set WANDB_RUN_ID for a new run")
        run_id = state["id"]
        entity = entity or state["entity"] or None
    run_id = run_id or uuid.uuid4().hex[:8]
    # init authenticates using WANDB_API_KEY or existing SDK credentials; a
    # separate `wandb login` command is unnecessary when the key is in .env.
    run = wandb.init(
        project=args.wandb_project, entity=entity,
        name=os.environ.get("WANDB_NAME") or cfg.run_name,
        id=run_id, resume="allow" if args.wandb_mode == "online" else None,
        mode=args.wandb_mode, dir=str(run_dir),
        config={
            **asdict(cfg.train),  # retain existing flat training config keys
            "model": asdict(cfg.model), "params": counts,
            "run_name": cfg.run_name, "data_dir": cfg.data_dir,
            "world_size": world_size,
            "effective_global_batch_tokens": cfg.train.batch_size * cfg.model.seq_len * cfg.train.accum_steps * world_size,
        },
    )
    state = {"id": run.id, "project": run.project, "entity": run.entity}
    tmp_path = state_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_path, state_path)
    # A custom axis preserves optimizer step numbers across sparse logging and
    # checkpoint replay without writing to an already-committed W&B history row.
    run.define_metric("optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("val/*", step_metric="optimizer_step")
    return run


def routing_metrics(prefix, *, counts, entropy, imbalance, biases, lambdas):
    import wandb

    metrics = {}
    for layer, values in counts.items():
        metrics[f"{prefix}/expert_entropy_L{layer}"] = entropy[layer]
        metrics[f"{prefix}/expert_imbalance_L{layer}"] = imbalance[layer]
        # One bin per expert, including empty experts, without expanding counts
        # into an array of individual token assignments.
        metrics[f"{prefix}/expert_counts_L{layer}"] = wandb.Histogram(
            np_histogram=(np.asarray(values), np.arange(len(values) + 1) - 0.5)
        )
        for expert, count in enumerate(values):
            metrics[f"{prefix}/expert_count_L{layer}_E{expert}"] = int(count)
    for layer, values in biases.items():
        metrics[f"{prefix}/router_bias_L{layer}"] = wandb.Histogram(values)
        for expert, bias in enumerate(values):
            metrics[f"{prefix}/router_bias_L{layer}_E{expert}"] = float(bias)
    for layer, values in lambdas.items():
        metrics[f"{prefix}/lambda_mean_L{layer}"] = sum(values) / len(values)
        metrics[f"{prefix}/lambda_hist_L{layer}"] = wandb.Histogram(values)
    return metrics
