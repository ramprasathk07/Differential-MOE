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


@dataclass
class EvalResult:
    nll: float
    perplexity: float
    bits_per_token: float
    bits_per_byte: Optional[float]
    top1_acc: float
    expert_entropy: Dict[int, float] = field(default_factory=dict)
    expert_imbalance: Dict[int, float] = field(default_factory=dict)
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

    for x, y in batches:
        logits, _, _ = model(x)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += loss.item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_tokens += y.numel()

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

    for i, block in enumerate(model.blocks):
        if block.is_moe and isinstance(block.ffn, MoE) and block.ffn.last_counts is not None:
            counts = block.ffn.last_counts.float()
            total = counts.sum().clamp_min(1)
            freq = counts / total
            n_experts = freq.numel()
            entropy = -(freq.clamp_min(1e-12) * freq.clamp_min(1e-12).log()).sum().item()
            result.expert_entropy[i] = entropy / math.log(n_experts)
            result.expert_imbalance[i] = (freq.max() / freq.mean().clamp_min(1e-12)).item()
        if hasattr(block.attn, "lambda_init"):
            lam = (
                torch.exp(torch.sum(block.attn.lambda_q1 * block.attn.lambda_k1, dim=-1).float())
                - torch.exp(torch.sum(block.attn.lambda_q2 * block.attn.lambda_k2, dim=-1).float())
                + block.attn.lambda_init
            )
            result.lambda_values[i] = lam.detach().cpu().tolist()

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
