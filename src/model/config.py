"""Model + training configuration. No import-time side effects."""

import os
from dataclasses import dataclass, field, fields
from typing import Dict, Literal, Optional, Sequence

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
    ffn: Literal["dense", "moe", "stable_latent_moe"] = "dense"
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

    # Stable LatentMoE (ignored by dense and conventional MoE). The router and
    # shared experts stay at full model width; only routed experts use the
    # compact latent width. None selects the Kimi-style 0.5 * dim default.
    latent_dim: Optional[int] = None
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    quantile_balance: bool = True

    norm_eps: float = 1e-6

    # Incremental architecture options. Defaults reproduce the S/A/B tiers exactly;
    # the L tier switches them on.
    norm: Literal["rmsnorm", "dyt", "derf"] = "rmsnorm"  # pre-attn, pre-FFN and final norm
    norm_alpha_init: float = 0.5
    residual: Literal["standard", "attnres", "mhc"] = "standard"
    attnres_block_layers: int = 3  # Block AttnRes: layers per block
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    xsa: bool = False  # Exclusive Self Attention
    doc_mask: bool = False  # block attention across eos_id document boundaries
    flex_attention: bool = True  # False: doc mask via dense SDPA mask instead of FlexAttention
    eos_id: int = -1

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        if self.attention == "differential":
            assert self.n_heads % 2 == 0, "differential attention needs even n_heads"
        head_dim = self.dim // self.n_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert 1 <= self.top_k <= self.n_experts, "top_k must be in [1, n_experts]"
        assert not self.doc_mask or self.eos_id >= 0, "doc_mask needs eos_id"
        assert self.attnres_block_layers >= 1 and self.mhc_streams >= 2
        if self.ffn == "stable_latent_moe":
            if self.latent_dim is None:
                self.latent_dim = self.dim // 2
            assert 0 < self.latent_dim < self.dim, "latent_dim must be between 0 and dim"
            assert self.top_k < self.n_experts, "Quantile Balancing requires top_k < n_experts"
            assert self.n_shared_experts > 0, "Stable LatentMoE requires full-width shared experts"
            assert self.shared_inter_dim > 0, "shared_inter_dim must be positive"
            assert self.situ_beta > 0 and self.situ_linear_beta > 0, "SiTU caps must be positive"
            assert self.aux_loss_coef == 0.0, (
                "Stable LatentMoE uses Quantile Balancing instead of an auxiliary balance loss"
            )
            assert self.router_z_coef == 0.0, (
                "Stable LatentMoE's sigmoid router does not use the softmax router z-loss"
            )


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
    amp: bool = True  # mixed precision on/off
    precision: Literal["fp16", "bf16"] = "fp16"  # fp16 adds a GradScaler; FSDP needs bf16
    log_freq: int = 20
    eval_freq: int = 500
    eval_batches: int = 50
    ckpt_freq: int = 500
    max_best_checkpoints: int = 2
    seed: int = 42
    # 0 keeps accum_steps per rank; >0 derives accum_steps so the optimizer
    # batch stays identical on any GPU count.
    global_batch_tokens: int = 0
    compile: bool = False
    activation_checkpointing: bool = False
    source_weights: Dict[str, float] = field(default_factory=dict)  # train-time sampling
    snapshot_freq: int = 0  # >0 keeps a weights-only checkpoint every N steps
    peak_tflops: float = 0.0  # per-GPU dense peak for MFU logging; 0 disables


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = "run"
    data_dir: str = "data"

    @classmethod
    def from_yaml(cls, path: str, overrides: Sequence[str] = ()) -> "Config":
        raw = apply_overrides(load_yaml(path), overrides)
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


def _merge(base: dict, top: dict) -> dict:
    out = dict(base)
    for k, v in top.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml(path: str) -> dict:
    """YAML with single inheritance: `base: other.yaml` (relative to this file)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    return _merge(load_yaml(os.path.join(os.path.dirname(path), base)), raw) if base else raw


def apply_overrides(raw: dict, overrides: Sequence[str]) -> dict:
    """Apply `section.key=value` strings; values parse as YAML (`model.xsa=false`)."""
    for item in overrides or ():
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like section.key=value: {item!r}")
        *parents, leaf = key.strip().split(".")
        node = raw
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return raw
