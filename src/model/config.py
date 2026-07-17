"""Model + training configuration. No import-time side effects."""

from dataclasses import dataclass, field, fields
from typing import Literal, Optional

import yaml


@dataclass
class ModelConfig:
    # embeddings
    vocab_size: int = 4096
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6  # standard-attention head count; differential uses n_heads // 2
    seq_len: int = 512
    tie_embeddings: bool = True

    # attention
    attention: Literal["standard", "differential"] = "standard"
    rope_theta: float = 10000.0

    # ffn
    ffn: Literal["dense", "moe"] = "dense"
    inter_dim: int = 1024  # dense SwiGLU hidden size

    # moe (ignored when ffn == "dense")
    n_dense_layers: int = 2  # first N layers stay dense for stability
    n_experts: int = 8
    top_k: int = 2
    expert_inter_dim: int = 512
    n_shared_experts: int = 0
    shared_inter_dim: int = 512
    aux_loss_coef: float = 0.01
    router_z_coef: float = 0.001

    norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        if self.attention == "differential":
            assert self.n_heads % 2 == 0, "differential attention needs even n_heads"
        head_dim = self.dim // self.n_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"


@dataclass
class TrainConfig:
    batch_size: int = 16
    accum_steps: int = 8
    max_steps: int = 7000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 140
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    amp: bool = True  # fp16 autocast + GradScaler on CUDA; ignored on CPU
    log_freq: int = 20
    eval_freq: int = 500
    eval_batches: int = 50
    ckpt_freq: int = 500
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = "run"
    data_dir: str = "data"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        model_keys = {f.name for f in fields(ModelConfig)}
        train_keys = {f.name for f in fields(TrainConfig)}
        model_raw = {k: v for k, v in (raw.get("model") or {}).items() if k in model_keys}
        train_raw = {k: v for k, v in (raw.get("train") or {}).items() if k in train_keys}
        unknown = (set((raw.get("model") or {})) - model_keys) | (set((raw.get("train") or {})) - train_keys)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(
            model=ModelConfig(**model_raw),
            train=TrainConfig(**train_raw),
            run_name=raw.get("run_name", "run"),
            data_dir=raw.get("data_dir", "data"),
        )
