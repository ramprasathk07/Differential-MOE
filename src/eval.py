"""Validation metrics: NLL, perplexity, bits/token, bits/byte, top-1 accuracy,
MoE routing entropy/utilization, differential-attention lambda values.

Bits-per-byte is the tokenizer-independent number (docs/plan.md SS3) -- it needs
the raw byte length of the text the evaluated tokens came from, so callers pass
`bytes_per_token` (mean UTF-8 bytes per token, measured once from the val text).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.model import Transformer
from src.model.moe import RoutedMoE


def expert_stats(counts: torch.Tensor) -> tuple:
    """(normalized entropy, imbalance ratio) from a per-expert token-count tensor.
    Normalized entropy: 1.0 = perfectly uniform routing, -> 0 = collapse."""
    counts = counts.float()
    total = counts.sum().clamp_min(1)
    freq = counts / total
    n_experts = freq.numel()
    entropy = -(freq.clamp_min(1e-12) * freq.clamp_min(1e-12).log()).sum().item()
    imbalance = (freq.max() / freq.mean().clamp_min(1e-12)).item()
    return entropy / math.log(n_experts), imbalance


def full_tensor(t: torch.Tensor) -> torch.Tensor:
    """Gather an FSDP2 DTensor (collective: call on every rank); plain tensors pass through."""
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _all_reduce(t: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t)
    return t


@dataclass
class EvalResult:
    nll: float
    perplexity: float
    bits_per_token: float
    bits_per_byte: Optional[float]
    top1_acc: float
    expert_entropy: Dict[int, float] = field(default_factory=dict)
    expert_imbalance: Dict[int, float] = field(default_factory=dict)
    expert_counts: Dict[int, List[int]] = field(default_factory=dict)
    router_bias: Dict[int, List[float]] = field(default_factory=dict)
    lambda_values: Dict[int, List[float]] = field(default_factory=dict)
    per_source: Dict[str, Dict[str, float]] = field(default_factory=dict)
    nll_by_position: List[float] = field(default_factory=list)  # one value per position bucket


@torch.no_grad()
def evaluate(
    model: Transformer,
    batches,
    bytes_per_token: Optional[float] = None,
    *,
    source_names: Sequence[str] = (),
    position_buckets: int = 0,
) -> EvalResult:
    """Batches yield (x, y) or (x, y, source_names). Under torch.distributed every
    rank evaluates its own shard of windows and the sums are all-reduced, so every
    rank returns the same global result (FSDP needs all ranks in the forward)."""
    model.eval()
    dev = next(model.parameters()).device
    names = list(source_names)
    sums = torch.zeros(3, dtype=torch.float64, device=dev)  # nll, correct, tokens
    src_sums = torch.zeros(len(names), 2, dtype=torch.float64, device=dev)
    pos_sums = None
    # accumulated across every eval batch, not just the last one -- last_counts
    # gets overwritten each forward call, so this has to be summed here
    expert_counts_sum: Dict[int, torch.Tensor] = {
        i: torch.zeros(b.ffn.n_experts, dtype=torch.long, device=dev)
        for i, b in enumerate(model.blocks) if b.is_moe and isinstance(b.ffn, RoutedMoE)
    }

    for batch in batches:
        x, y = batch[0], batch[1]
        logits, _, _ = model(x)
        logits = logits.float()
        tok_nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none").view(y.shape)
        sums += torch.stack([tok_nll.sum().double(), (logits.argmax(-1) == y).sum().double(),
                             torch.tensor(float(y.numel()), dtype=torch.float64, device=dev)])
        if pos_sums is None:
            pos_sums = torch.zeros(y.size(1), dtype=torch.float64, device=dev)
            pos_count = torch.zeros((), dtype=torch.float64, device=dev)
        pos_sums += tok_nll.sum(0).double()
        pos_count += y.size(0)
        if names and len(batch) > 2:
            idx = torch.tensor([names.index(n) for n in batch[2]], device=dev)
            src_sums.index_add_(0, idx, torch.stack(
                [tok_nll.sum(1).double(), torch.full((y.size(0),), float(y.size(1)), dtype=torch.float64, device=dev)], 1))
        for i in expert_counts_sum:
            if model.blocks[i].ffn.last_counts is not None:
                expert_counts_sum[i] += model.blocks[i].ffn.last_counts.to(dev)

    _all_reduce(sums)
    _all_reduce(src_sums)
    for c in expert_counts_sum.values():
        _all_reduce(c)
    total_nll, total_correct, total_tokens = sums.tolist()
    nll = total_nll / total_tokens
    ppl = math.exp(min(nll, 50))  # guard exp overflow on early / broken runs
    bits_per_token = nll / math.log(2)
    bits_per_byte = bits_per_token / bytes_per_token if bytes_per_token else None

    result = EvalResult(
        nll=nll,
        perplexity=ppl,
        bits_per_token=bits_per_token,
        bits_per_byte=bits_per_byte,
        top1_acc=total_correct / total_tokens,
    )
    if position_buckets and pos_sums is not None:
        _all_reduce(pos_sums)
        _all_reduce(pos_count)
        per_pos = (pos_sums / pos_count).cpu()
        result.nll_by_position = [float(c.mean()) for c in per_pos.chunk(position_buckets)]
    for j, name in enumerate(names):
        s_nll, s_tok = src_sums[j].tolist()
        if s_tok:
            result.per_source[name] = {"nll": s_nll / s_tok, "tokens": s_tok,
                                       "bits_per_byte": (s_nll / s_tok / math.log(2) / bytes_per_token)
                                       if bytes_per_token else None}

    for i, counts in expert_counts_sum.items():
        result.expert_entropy[i], result.expert_imbalance[i] = expert_stats(counts)
        result.expert_counts[i] = counts.detach().cpu().tolist()
    for i, block in enumerate(model.blocks):
        if block.is_moe and hasattr(block.ffn.gate, "expert_bias"):
            result.router_bias[i] = block.ffn.gate.expert_bias.detach().cpu().tolist()
        if hasattr(block.attn, "current_lambda"):
            result.lambda_values[i] = lambda_of(block.attn)

    return result


def lambda_of(attn) -> List[float]:
    """Per-head lambda, gathering FSDP shards first (collective under FSDP)."""
    q1, k1, q2, k2 = (full_tensor(p.detach()).float()
                      for p in (attn.lambda_q1, attn.lambda_k1, attn.lambda_q2, attn.lambda_k2))
    lam = torch.exp((q1 * k1).sum(-1)) - torch.exp((q2 * k2).sum(-1)) + attn.lambda_init
    return lam.cpu().tolist()


def bytes_per_token_estimate(data_dir: str, bin_path: str, n_sample: int = 2000):
    """Mean UTF-8 bytes per token, measured by decoding a sample of the split.

    Returns None when the data_dir has no resolvable tokenizer, in which case
    bits-per-byte is simply not reported.
    """
    from src.data.dataset import load_tokens
    from src.data.tokenizer import resolve_data_tokenizer

    tok = resolve_data_tokenizer(data_dir)
    if tok is None:
        return None
    data = load_tokens(bin_path)  # reads the dtype prepare.py recorded
    sample = data[:n_sample].astype(int).tolist()
    return len(tok.decode(sample).encode("utf-8")) / len(sample)


@torch.no_grad()
def sample_generations(
    model: Transformer,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 80,
    device: str = "cuda",
) -> List[str]:
    model.eval()
    outputs = []
    for p in prompts:
        ids = tokenizer.encode(p)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        gen = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.9)
        outputs.append(tokenizer.decode(gen[0].tolist()))
    return outputs


def _main():
    """Final, one-time evaluation on the held-out BabyLM test split. Run this
    only after training is finished and a checkpoint is already chosen by
    dev-set performance -- the test split must never influence any decision
    made during training (that's the whole point of keeping it separate from
    val.bin/dev). See docs/plan.md and docs/tokenizer-and-model-training.md.

    Usage:
        python -m src.eval --config configs/a_diffmoe.yaml --run_dir checkpoints/a_diffmoe
    """
    import argparse
    import json as _json
    import os

    from src.data import eval_batches, load_tokens
    from src.model import Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True, help="e.g. checkpoints/a_diffmoe")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="overrides the auto-picked best checkpoint from run_dir/best_index.json")
    ap.add_argument("--data_dir", type=str, default=None, help="overrides config's data_dir")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    data_dir = args.data_dir or cfg.data_dir

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        with open(os.path.join(args.run_dir, "best_index.json")) as f:
            best_index = _json.load(f)
        ckpt_path = best_index[0]["path"]
        print(f"auto-picked best checkpoint (by dev NLL): {ckpt_path}")

    model = Transformer(cfg.model).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])

    from src.data.tokenizer import resolve_data_tokenizer

    test_bin = os.path.join(data_dir, "test.bin")
    test_data = load_tokens(test_bin)
    bytes_per_token = bytes_per_token_estimate(data_dir, test_bin)
    n_batches = (len(test_data) - 1) // (cfg.train.batch_size * cfg.model.seq_len) + 1
    batches = eval_batches(test_data, cfg.train.batch_size, cfg.model.seq_len, n_batches, args.device)
    result = evaluate(model, batches, bytes_per_token)

    generations = {}
    tok = resolve_data_tokenizer(data_dir)
    if tok is not None:
        prompts = [
            "Once upon a time",
            "The little boy said",
            "A:\tHow are you doing",
            "The city is known for",
            "She looked at the",
        ]
        for p, out in zip(prompts, sample_generations(model, tok, prompts, max_new_tokens=60, device=args.device)):
            generations[p] = out

    report = {
        "run_name": cfg.run_name,
        "checkpoint": ckpt_path,
        "checkpoint_step": ckpt.get("step"),
        "test_nll": result.nll,
        "test_ppl": result.perplexity,
        "test_bits_per_byte": result.bits_per_byte,
        "test_top1_acc": result.top1_acc,
        "expert_entropy": result.expert_entropy,
        "expert_imbalance": result.expert_imbalance,
        "router_bias": result.router_bias,
        "lambda_means": {i: sum(v) / len(v) for i, v in result.lambda_values.items()},
        "sample_generations": generations,
    }
    out_path = os.path.join(args.run_dir, "final_test_eval.json")
    with open(out_path, "w") as f:
        _json.dump(report, f, indent=2)
    print(_json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()
