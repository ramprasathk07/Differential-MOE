"""W&B configuration, resume identity, and complete per-step training metrics."""

from dataclasses import asdict
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src import train
from src.model.config import Config, ModelConfig, TrainConfig
from src.tracking import init_wandb


@pytest.fixture(autouse=True)
def isolated_wandb_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("WANDB_"):
            monkeypatch.delenv(name)


@pytest.fixture
def sdk(monkeypatch):
    runs, calls = [], []

    def initialize(**kwargs):
        calls.append(kwargs)
        run = SimpleNamespace(
            id=kwargs["id"], project=kwargs["project"], entity=kwargs["entity"] or "test-team",
            rows=[], summary={}, definitions=[], finished=False,
        )
        run.log = lambda values: run.rows.append(dict(values))
        run.define_metric = lambda *args, **kw: run.definitions.append((args, kw))
        run.finish = lambda: setattr(run, "finished", True)
        runs.append(run)
        return run

    def histogram(values=None, np_histogram=None):
        return np.histogram(values) if np_histogram is None else np_histogram

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize, Histogram=histogram))
    return SimpleNamespace(runs=runs, calls=calls)


def test_env_loads_for_python_and_shell_cli_override_it(tmp_path, monkeypatch):
    env_file = tmp_path / "settings.env"
    env_file.write_text("WANDB_ENABLED=true\nWANDB_PROJECT=from-file\nWANDB_API_KEY=fixture-key\nWANDB_MODE=offline\n")
    monkeypatch.setenv("WANDB_PROJECT", "from-shell")
    argv = ["--config", "unused.yaml", "--env_file", str(env_file)]
    args = train.parse_args(argv)
    assert args.wandb is True
    assert args.wandb_project == "from-shell"
    assert args.wandb_mode == "offline"
    assert os.environ["WANDB_API_KEY"] == "fixture-key"
    args = train.parse_args(argv + ["--no-wandb", "--wandb_project", "from-cli"])
    assert args.wandb is False
    assert args.wandb_project == "from-cli"


def test_resume_keeps_wandb_identity_without_logging_credentials(tmp_path, sdk, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret")
    args = SimpleNamespace(wandb=True, wandb_mode="online", wandb_project="project", wandb_entity=None)
    cfg = Config()
    first = init_wandb(args, cfg, {"total": 100}, 1, tmp_path)
    second = init_wandb(args, cfg, {"total": 100}, 1, tmp_path, resuming=True)
    assert second.id == first.id
    assert sdk.calls[1]["resume"] == "allow"
    assert sdk.calls[1]["entity"] == "test-team"
    assert "fixture-secret" not in json.dumps(sdk.calls)
    assert "fixture-secret" not in (tmp_path / "wandb_run.json").read_text()
    assert sdk.calls[0]["config"]["model"]["dim"] == cfg.model.dim
    assert sdk.calls[0]["config"]["params"] == {"total": 100}
    args.wandb_project = "different-project"
    with pytest.raises(ValueError, match="differs from the saved run"):
        init_wandb(args, cfg, {}, 1, tmp_path, resuming=True)


@pytest.mark.parametrize("enabled, mode", [(False, "online"), (True, "disabled")])
def test_disabled_tracking_never_initializes_sdk(tmp_path, sdk, enabled, mode):
    args = SimpleNamespace(wandb=enabled, wandb_mode=mode)
    assert init_wandb(args, Config(), {}, 1, tmp_path) is None
    assert not sdk.calls


def test_plain_python_training_logs_train_eval_and_resume(tmp_path, sdk, monkeypatch):
    cfg = Config(
        run_name="tracking-smoke", data_dir=str(tmp_path / "data"),
        model=ModelConfig(vocab_size=32, dim=16, n_layers=2, n_heads=4, seq_len=8,
            attention="differential", ffn="stable_latent_moe", inter_dim=32,
            n_dense_layers=1, latent_dim=8, n_experts=4, top_k=2, expert_inter_dim=8,
            n_shared_experts=2, shared_inter_dim=4, aux_loss_coef=0.0, router_z_coef=0.0),
        train=TrainConfig(batch_size=2, accum_steps=2, max_steps=5, warmup_steps=1,
            amp=False, log_freq=2, eval_freq=3, eval_batches=2, ckpt_freq=3),
    )
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val"):
        np.tile(np.arange(32, dtype=np.uint16), 16).tofile(data / f"{split}.bin")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg)))
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_ENABLED=true\nWANDB_MODE=offline\nWANDB_PROJECT=test-project\n")
    monkeypatch.setattr(train, "bytes_per_token_estimate", lambda *args: 2.0)
    argv = ["src.train", "--config", str(config_path), "--env_file", str(env_file),
            "--device", "cpu", "--out_dir", str(tmp_path / "checkpoints")]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()
    run = sdk.runs[0]
    assert [row["optimizer_step"] for row in run.rows] == [2, 3, 4, 5]
    assert "train/loss" not in run.rows[1]  # evaluation-only step
    last = run.rows[-1]
    for metric in ("train/loss", "train/objective", "train/ppl", "train/lr",
                   "train/grad_norm", "train/tok_per_sec", "train/amp_scale",
                   "train/aux_loss", "train/router_z_loss", "train/skipped_optimizer_steps",
                   "val/nll", "val/ppl", "val/top1", "val/bits_per_token", "val/bits_per_byte",
                   "train/expert_entropy_L1", "val/expert_entropy_L1",
                   "train/expert_imbalance_L1", "val/expert_imbalance_L1",
                   "train/router_bias_L1_E0", "val/router_bias_L1_E0",
                   "train/lambda_mean_L0", "val/lambda_mean_L0"):
        assert math.isfinite(last[metric]), metric
    assert last["train/tokens_trained"] == 160
    assert last["train/tok_per_sec"] > 0
    assert sum(last[f"train/expert_count_L1_E{i}"] for i in range(4)) == 32
    assert sum(last[f"val/expert_count_L1_E{i}"] for i in range(4)) == 64
    hist_counts, hist_edges = last["val/expert_counts_L1"]
    assert hist_counts.sum() == 64
    assert len(hist_edges) == 5
    assert run.finished
    assert run.summary["training/final_loss"] is not None
    assert run.summary["training/tokens"] == 160
    assert (("train/*",), {"step_metric": "optimizer_step"}) in run.definitions

    monkeypatch.setattr(sys, "argv", argv + ["--max_steps", "6"])
    train.main()
    resumed = sdk.runs[1]
    assert resumed.id == run.id
    assert [row["optimizer_step"] for row in resumed.rows] == [6]
    assert "train/loss" in resumed.rows[0] and "val/nll" in resumed.rows[0]
