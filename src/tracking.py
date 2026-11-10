"""Environment loading, W&B and MLflow setup, with no import-time login or side effects.

W&B carries every training metric (histograms included). MLflow tracks the run
as a model lineage: config as params, scalar metrics, and every checkpoint
(best / snapshot / final) with its step, tokens, epoch and val NLL.
"""

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


def init_wandb(args, cfg, counts, world_size, run_dir, *, resuming=False, extra_config=None):
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
            **(extra_config or {}),
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
    run.define_metric("ckpt/*", step_metric="optimizer_step")
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


def arch_metrics(prefix, diag):
    """Scalars for the incremental architecture options (AttnRes, DyT/Derf, mHC)."""
    metrics = {}
    for layer, w in diag.get("attnres_weights", {}).items():
        w = np.asarray(w, dtype=np.float64)
        ent = -(w * np.log(np.clip(w, 1e-12, None))).sum()
        metrics[f"{prefix}/attnres_entropy_{layer}"] = float(ent / np.log(len(w))) if len(w) > 1 else 1.0
        metrics[f"{prefix}/attnres_embed_weight_{layer}"] = float(w[0])
        metrics[f"{prefix}/attnres_latest_weight_{layer}"] = float(w[-1])
    for layer, (alpha, shift) in diag.get("norm_params", {}).items():
        metrics[f"{prefix}/norm_alpha_{layer}"] = alpha
        if shift is not None:
            metrics[f"{prefix}/norm_shift_{layer}"] = shift
    for layer, value in diag.get("mhc_res_diag", {}).items():
        metrics[f"{prefix}/mhc_res_diag_{layer}"] = value
    return metrics


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value
    return out


class MLflowTracker:
    """Rank-zero MLflow run, resumable through <run_dir>/mlflow_run.json.

    MLFLOW_TRACKING_URI selects the store (a shared server lets the training box
    and the RTX 3060 eval PC write to the same run); MLflow's default otherwise.
    """

    def __init__(self, run_dir, experiment, run_name, *, resuming, log_checkpoints=False):
        try:
            import mlflow
        except ImportError:
            raise RuntimeError("MLflow is enabled; install dependencies with python -m pip install -r requirements.txt") from None
        self.mlflow = mlflow
        self.log_checkpoints = log_checkpoints
        self.state_path = Path(run_dir) / "mlflow_run.json"
        mlflow.set_experiment(experiment)
        run_id = None
        if resuming and self.state_path.exists():
            run_id = json.loads(self.state_path.read_text(encoding="utf-8"))["run_id"]
        self.run = mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name)
        self.state_path.write_text(json.dumps({"run_id": self.run.info.run_id,
                                               "experiment": experiment}, indent=2), encoding="utf-8")
        self.checkpoints = []

    def log_params(self, params):
        flat = _flatten("", params, {})
        existing = self.mlflow.get_run(self.run.info.run_id).data.params
        # MLflow params are immutable; a resumed run only adds new keys.
        fresh = {k: str(v)[:6000] for k, v in flat.items() if k not in existing}
        items = list(fresh.items())
        for i in range(0, len(items), 100):
            self.mlflow.log_params(dict(items[i:i + 100]))

    def set_tags(self, tags):
        self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_metrics(self, metrics, step):
        scalars = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)}
        if scalars:
            self.mlflow.log_metrics(scalars, step=step)

    def log_checkpoint(self, kind, path, info):
        """kind: best | snapshot | last. info: step, tokens, epoch, val_nll, ..."""
        entry = {"kind": kind, "path": str(path), **info}
        self.checkpoints.append(entry)
        self.mlflow.log_dict({"checkpoints": self.checkpoints}, "checkpoints/index.json")
        self.mlflow.log_metrics({f"ckpt/{kind}_step": float(info["step"])}, step=int(info["step"]))
        if kind == "best":
            self.set_tags({"best_checkpoint": path, "best_step": info["step"], "best_val_nll": info.get("val_nll")})
        if self.log_checkpoints and kind in ("best", "snapshot"):
            self.mlflow.log_artifact(str(path), artifact_path=f"checkpoints/{kind}")

    def log_artifact(self, path, artifact_path=None):
        if os.path.exists(path):
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def finish(self, status="FINISHED"):
        self.mlflow.end_run(status=status)
