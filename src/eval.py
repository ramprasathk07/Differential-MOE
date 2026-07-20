"""Validation metrics: NLL, perplexity, bits/token, bits/byte, top-1 accuracy,
MoE routing entropy/utilization, differential-attention lambda values.

Bits-per-byte is the tokenizer-independent number (docs/plan.md SS3) -- it needs
the raw byte length of the text the evaluated tokens came from, so callers pass
`bytes_per_token` (mean UTF-8 bytes per token, measured once from the val text).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.model import Transformer
from src.model.moe import MoE


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
    lambda_values: Dict[int, List[float]] = field(default_factory=dict)


@torch.no_grad()
def evaluate(
    model: Transformer,
    batches,
    bytes_per_token: Optional[float] = None,
) -> EvalResult:
    model.eval()
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    # accumulated across every eval batch, not just the last one -- last_counts
    # gets overwritten each forward call, so this has to be summed here
    expert_counts_sum: Dict[int, torch.Tensor] = {}

    for x, y in batches:
        logits, _, _ = model(x)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += loss.item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_tokens += y.numel()
        for i, block in enumerate(model.blocks):
            if block.is_moe and isinstance(block.ffn, MoE) and block.ffn.last_counts is not None:
                if i not in expert_counts_sum:
                    expert_counts_sum[i] = block.ffn.last_counts.clone()
                else:
                    expert_counts_sum[i] += block.ffn.last_counts

    nll = total_nll / total_tokens
    ppl = math.exp(min(nll, 50))  # guard exp overflow on early / broken runs
    bits_per_token = nll / math.log(2)
    bits_per_byte = (nll * total_tokens) / (math.log(2) * bytes_per_token * total_tokens) if bytes_per_token else None
    top1 = total_correct / total_tokens

    result = EvalResult(
        nll=nll,
        perplexity=ppl,
        bits_per_token=bits_per_token,
        bits_per_byte=bits_per_byte,
        top1_acc=top1,
    )

    for i, counts in expert_counts_sum.items():
        result.expert_entropy[i], result.expert_imbalance[i] = expert_stats(counts)
        result.expert_counts[i] = counts.detach().cpu().tolist()
    for i, block in enumerate(model.blocks):
        if hasattr(block.attn, "current_lambda"):
            result.lambda_values[i] = block.attn.current_lambda().detach().cpu().tolist()

    return result


def bytes_per_token_estimate(tokenizer_path: str, val_bin_path: str, n_sample: int = 2000) -> float:
    """Mean UTF-8 bytes per token, measured by decoding a sample of val.bin."""
    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    data = np.memmap(val_bin_path, dtype=np.uint16, mode="r")
    sample = data[:n_sample].astype(int).tolist()
    text = tok.decode(sample)
    return len(text.encode("utf-8")) / len(sample)


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
        ids = tokenizer.encode(p).ids
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

    test_data = load_tokens(os.path.join(data_dir, "test.bin"))
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    bytes_per_token = (
        bytes_per_token_estimate(tokenizer_path, os.path.join(data_dir, "test.bin"))
        if os.path.exists(tokenizer_path)
        else None
    )
    n_batches = (len(test_data) - 1) // (cfg.train.batch_size * cfg.model.seq_len) + 1
    batches = eval_batches(test_data, cfg.train.batch_size, cfg.model.seq_len, n_batches, args.device)
    result = evaluate(model, batches, bytes_per_token)

    generations = {}
    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(tokenizer_path)
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
