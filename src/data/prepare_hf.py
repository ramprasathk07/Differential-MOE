"""Stream a Hugging Face text dataset into the trainer's memmap format.

Usage:
    python -m src.data.prepare_hf --config configs/rtx6000ada_stable_latentmoe.yaml

The YAML's `data` section controls preparation. Content hashes assign whole
documents to train/val/test before tokenization. A completed split's documents
are discarded, never redirected to another split. Only the last document in
each split is truncated to its token budget; every document ends with EOS.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from src.model.config import Config
from .tokenizer import load_tokenizer, token_dtype


def document_split(text, seed, val_fraction, test_fraction):
    """Stable across machines and traversal order; equal stripped text stays together."""
    digest = hashlib.sha256(f"{seed}\0{text.strip()}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < val_fraction:
        return "val"
    if value < val_fraction + test_fraction:
        return "test"
    return "train"


def prepare_rows(rows, tok, out_dir, settings, *, model_vocab, seq_len):
    """Consume an iterable once, keeping only one encoded document in memory."""
    budgets = {split: settings[f"{split}_tokens"] for split in ("train", "val", "test")}
    if any(not isinstance(n, int) or isinstance(n, bool) or n < seq_len + 2
           for n in budgets.values()):
        raise ValueError("Every split token budget must be an integer >= seq_len + 2")
    val_fraction, test_fraction = settings["val_fraction"], settings["test_fraction"]
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be positive and sum to < 1")
    if tok.vocab_size > model_vocab:
        raise ValueError(f"Tokenizer vocab {tok.vocab_size} exceeds model vocab {model_vocab}")
    if not 0 <= tok.eot_id < tok.vocab_size:
        raise ValueError("Tokenizer EOS is outside its vocabulary")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} must be empty; use a new data_dir to prepare new data")
    dtype = token_dtype(tok.vocab_size)
    counts = dict.fromkeys(budgets, 0)
    documents = dict.fromkeys(budgets, 0)
    max_id = tok.eot_id
    next_log = 1000000
    # No usable corpus is published until all three budgets have been filled.
    with tempfile.TemporaryDirectory(prefix=".prepare_hf-", dir=out_dir) as staging:
        staging = Path(staging)
        with ExitStack() as stack:
            files = {split: stack.enter_context((staging / f"{split}.bin").open("wb"))
                     for split in budgets}
            for row in rows:
                text = row[settings["text_column"]]
                if not isinstance(text, str) or not text.strip():
                    continue
                text = text.strip()
                split = document_split(text, settings["seed"], val_fraction, test_fraction)
                remaining = budgets[split] - counts[split]
                if remaining == 0:
                    continue
                ids = tok.encode(text)
                if not ids:
                    continue
                # Check BEFORE narrowing to uint16/uint32 to prevent silent overflow.
                if min(ids) < 0 or max(ids) >= tok.vocab_size:
                    raise ValueError("Encoded token ID is outside the tokenizer vocabulary")
                ids = ids[:remaining - 1] + [tok.eot_id]
                arr = np.asarray(ids, dtype=dtype)
                arr.tofile(files[split])
                counts[split] += len(ids)
                documents[split] += 1
                max_id = max(max_id, int(arr.max()))
                total = sum(counts.values())
                if total >= next_log:
                    print(f"tokens: {counts}", flush=True)
                    next_log = total + 1000000
                if all(counts[s] == budgets[s] for s in budgets):
                    break
            else:
                raise ValueError(f"Dataset exhausted before token budgets were met: {counts} / {budgets}")

        meta = {
            "dtype": np.dtype(dtype).name,
            "vocab_size": tok.vocab_size,
            "max_token_id": max_id,
            "tokenizer_spec": tok.spec,
            "track": "huggingface",
            "tokens": counts,
            "documents": documents,
            "source": settings,
            "split_method": "sha256(seed + NUL + stripped_text), first 64 bits",
        }
        domain = settings["dataset"]
        offsets = {split: {domain: [0, count]} for split, count in counts.items()}
        (staging / "domain_offsets.json").write_text(json.dumps(offsets, indent=2), encoding="utf-8")
        (staging / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for name in ["train.bin", "val.bin", "test.bin", "domain_offsets.json", "meta.json"]:
            os.replace(staging / name, out_dir / name)
    print(f"Prepared {counts} as {np.dtype(dtype).name} in {out_dir}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", help="override the YAML data_dir")
    ap.add_argument("--train_tokens", type=int, help="override for a small preparation smoke test")
    ap.add_argument("--val_tokens", type=int)
    ap.add_argument("--test_tokens", type=int)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    with open(args.config, encoding="utf-8") as f:
        settings = yaml.safe_load(f)["data"]
    for split in ("train", "val", "test"):
        value = getattr(args, f"{split}_tokens")
        if value is not None:
            settings[f"{split}_tokens"] = value
    allowed = {"dataset", "subset", "revision", "split", "text_column", "tokenizer",
               "train_tokens", "val_tokens", "test_tokens", "val_fraction", "test_fraction",
               "shuffle_buffer", "seed"}
    if set(settings) != allowed:
        raise ValueError(f"data keys: missing={allowed - set(settings)}, unknown={set(settings) - allowed}")
    if not isinstance(settings["shuffle_buffer"], int) or settings["shuffle_buffer"] < 1:
        raise ValueError("shuffle_buffer must be a positive integer")
    out_dir = args.out_dir or cfg.data_dir
    if Path(out_dir).exists() and any(Path(out_dir).iterdir()):
        raise FileExistsError(f"{out_dir} must be empty; choose a new output directory")

    from datasets import load_dataset

    tok = load_tokenizer(settings["tokenizer"])
    rows = load_dataset(
        settings["dataset"], name=settings["subset"], revision=settings["revision"],
        split=settings["split"], streaming=True,
    ).shuffle(seed=settings["seed"], buffer_size=settings["shuffle_buffer"])
    prepare_rows(rows, tok, out_dir, settings, model_vocab=cfg.model.vocab_size,
                 seq_len=cfg.model.seq_len)


if __name__ == "__main__":
    main()
