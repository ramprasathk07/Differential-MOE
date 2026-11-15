"""Memmap token streams + nanoGPT-style batch sampling.

No DataLoader, no workers, no streaming: a seeded torch.Generator draws random
window offsets, its state is checkpointed, so training data order is exactly
reproducible across resume. Validation uses fixed strided windows.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def load_tokens(path: str) -> np.ndarray:
    """Token width is whatever prepare.py wrote -- uint16 for small vocabs,
    uint32 once a frontier tokenizer pushes past 65535. meta.json records it;
    fall back to uint16 for data written before meta.json existed."""
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    dtype = np.uint16
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            dtype = np.dtype(json.load(f).get("dtype", "uint16"))
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: str,
    starts=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform random windows, or the given window `starts` (SourceSampler)."""
    if starts is None:
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack(
        [torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in starts]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64)) for i in starts]
    )
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval_batches(
    data: np.ndarray, batch_size: int, seq_len: int, n_batches: int, device: str
):
    """Deterministic strided windows over the validation stream."""
    n_windows = (len(data) - 1) // seq_len
    n_needed = min(n_batches * batch_size, n_windows)
    offsets = [w * seq_len for w in range(n_needed)]
    for b in range(0, n_needed, batch_size):
        chunk = offsets[b : b + batch_size]
        x = torch.stack(
            [torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in chunk]
        ).to(device)
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64)) for i in chunk]
        ).to(device)
        yield x, y


class SourceSampler:
    """Window starts drawn with probability proportional to source weight per token.

    Reads the document index prepare_hf writes (`train_docs.npz`: doc start
    offsets, source id per doc, source names). Weight 2 for a source makes each
    of its tokens twice as likely to start a window as a weight-1 token.
    """

    def __init__(self, docs_path: str, n_tokens: int, seq_len: int, weights: Dict[str, float]):
        z = np.load(docs_path)
        self.names: List[str] = [str(n) for n in z["sources"]]
        unknown = set(weights) - set(self.names)
        if unknown:
            raise ValueError(f"source_weights names unknown sources {sorted(unknown)}; have {self.names}")
        self.starts = z["starts"].astype(np.int64)
        self.source = z["source"].astype(np.int64)
        per_source = np.array([float(weights.get(n, 1.0)) for n in self.names])
        self.w = per_source[self.source]
        lengths = np.diff(np.append(self.starts, n_tokens))
        self.cum = np.cumsum(lengths * self.w)
        self.limit = n_tokens - seq_len - 1

    def sample(self, batch_size: int, generator: torch.Generator):
        u = torch.rand(batch_size, generator=generator, dtype=torch.float64).numpy() * self.cum[-1]
        doc = np.minimum(np.searchsorted(self.cum, u, side="right"), len(self.cum) - 1)
        before = np.where(doc > 0, self.cum[doc - 1], 0.0)
        pos = self.starts[doc] + ((u - before) / self.w[doc]).astype(np.int64)
        return np.minimum(pos, self.limit).tolist(), self.source[doc]


def eval_windows(n_tokens: int, seq_len: int, n_windows: int,
                 spans: Optional[Dict[str, list]] = None) -> List[Tuple[int, str]]:
    """Deterministic (offset, source) windows. With per-source spans, windows are
    split evenly across sources and stay inside each span."""
    if not spans:  # same windows as eval_batches, so legacy val curves stay comparable
        return [(w * seq_len, "all") for w in range(min(n_windows, (n_tokens - 1) // seq_len))]
    per = max(1, n_windows // len(spans))
    out = []
    for name, (lo, hi) in spans.items():
        total = max(0, (hi - lo - 1) // seq_len)
        n = min(per, total)
        out += [(lo + int(k * total / n) * seq_len, name) for k in range(n)]
    return out


def window_batches(data: np.ndarray, windows, batch_size: int, seq_len: int, device: str):
    """Yields (x, y, source_names) for the given windows."""
    for b in range(0, len(windows), batch_size):
        chunk = windows[b : b + batch_size]
        x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i, _ in chunk])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64)) for i, _ in chunk])
        yield x.to(device), y.to(device), [name for _, name in chunk]
