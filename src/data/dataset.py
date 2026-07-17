"""Memmap token streams + nanoGPT-style batch sampling.

No DataLoader, no workers, no streaming: a seeded torch.Generator draws random
window offsets, its state is checkpointed, so training data order is exactly
reproducible across resume. Validation uses fixed strided windows.
"""

from typing import Tuple

import numpy as np
import torch


def load_tokens(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack(
        [torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix.tolist()]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64)) for i in ix.tolist()]
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
