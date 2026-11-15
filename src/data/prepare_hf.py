"""Stream a Hugging Face text dataset into the trainer's memmap format.

Usage:
    python -m src.data.prepare_hf --config configs/l_base.yaml

The YAML's `data` section controls preparation. Content hashes assign whole
documents to train/val/test before tokenization. A completed split's documents
are discarded, never redirected to another split. Only the last document in
each budget is truncated; every document ends with EOS.

With `source_column` (e.g. SlimPajama's `meta.redpajama_set_name`) and a
`sources` list, val/test budgets apply per source and each source is written as
one contiguous span, so `domain_offsets.json` supports per-source evaluation.
`train_docs.npz` (document starts + source ids) drives source-weighted sampling.
"""

import argparse
from array import array
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from src.model.config import Config, load_yaml
from .tokenizer import load_tokenizer, token_dtype

ENCODE_BATCH = 256


def document_split(text, seed, val_fraction, test_fraction):
    """Stable across machines and traversal order; equal stripped text stays together."""
    digest = hashlib.sha256(f"{seed}\0{text.strip()}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < val_fraction:
        return "val"
    if value < val_fraction + test_fraction:
        return "test"
    return "train"


def _source(row, column):
    value = row
    for key in column.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def check_sources(unmatched, matched, sources, limit=10_000):
    """A misspelled `sources` list would otherwise skip every row of a 590M-doc stream."""
    if not any(matched.values()) and sum(unmatched.values()) >= limit:
        seen = sorted(unmatched, key=unmatched.get, reverse=True)[:10]
        raise ValueError(f"No row matched data.sources {sources} in {limit} rows; "
                         f"the source column holds e.g. {seen}")


def _batches(rows, size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_rows(rows, tok, out_dir, settings, *, model_vocab, seq_len):
    """Consume an iterable once, keeping only one encode batch (+ val/test) in memory."""
    column = settings.get("source_column")
    sources = list(settings.get("sources") or [settings["dataset"]])
    if column and not settings.get("sources"):
        raise ValueError("source_column needs a `sources` list")
    budgets = {("train", None): settings["train_tokens"]}
    for split in ("val", "test"):
        for src in sources:
            budgets[(split, src)] = settings[f"{split}_tokens"]
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
    held = {key: [] for key in budgets if key[0] != "train"}  # val/test, written per source at the end
    doc_starts, doc_source = array("q"), array("B")
    source_id = {src: i for i, src in enumerate(sources)}
    max_id = tok.eot_id
    unmatched = {}  # source names seen but not configured, for the fail-fast check
    total_budget = sum(budgets.values())
    log_every = max(1_000_000, total_budget // 200)
    next_log = log_every
    # No usable corpus is published until every budget has been filled.
    with tempfile.TemporaryDirectory(prefix=".prepare_hf-", dir=out_dir) as staging:
        staging = Path(staging)
        with ExitStack() as stack:
            train_file = stack.enter_context((staging / "train.bin").open("wb"))
            done = False
            for batch in _batches(rows, ENCODE_BATCH):
                todo = []
                for row in batch:
                    text = row[settings["text_column"]]
                    if not isinstance(text, str) or not text.strip():
                        continue
                    src = _source(row, column) if column else sources[0]
                    if src not in source_id:
                        unmatched[src] = unmatched.get(src, 0) + 1
                        check_sources(unmatched, documents, sources)
                        continue
                    text = text.strip()
                    split = document_split(text, settings["seed"], val_fraction, test_fraction)
                    key = ("train", None) if split == "train" else (split, src)
                    if counts[key] < budgets[key]:
                        todo.append((key, src, text))
                if not todo:
                    continue
                for (key, src, _), ids in zip(todo, tok.encode_batch([t for _, _, t in todo])):
                    remaining = budgets[key] - counts[key]
                    if remaining == 0 or not ids:
                        continue
                    # Check BEFORE narrowing to uint16/uint32 to prevent silent overflow.
                    if min(ids) < 0 or max(ids) >= tok.vocab_size:
                        raise ValueError("Encoded token ID is outside the tokenizer vocabulary")
                    arr = np.asarray(ids[:remaining - 1] + [tok.eot_id], dtype=dtype)
                    if key[0] == "train":
                        doc_starts.append(counts[key])
                        doc_source.append(source_id[src])
                        arr.tofile(train_file)
                    else:
                        held[key].append(arr)
                    counts[key] += len(arr)
                    documents[key] += 1
                    max_id = max(max_id, int(arr.max()))
                    total = sum(counts.values())
                    if total >= next_log:
                        print(f"tokens: {total:,} / {total_budget:,}", flush=True)
                        next_log = total + log_every
                    if all(counts[k] == budgets[k] for k in budgets):
                        done = True
                        break
                if done:
                    break
            if not done:
                raise ValueError(f"Dataset exhausted before token budgets were met: "
                                 f"{ {str(k): v for k, v in counts.items()} } / "
                                 f"{ {str(k): v for k, v in budgets.items()} }")

        n_train = counts[("train", None)]
        offsets = {"train": {settings["dataset"]: [0, n_train]}}
        for split in ("val", "test"):
            offsets[split], pos = {}, 0
            with (staging / f"{split}.bin").open("wb") as f:
                for src in sources:
                    for arr in held[(split, src)]:
                        arr.tofile(f)
                    offsets[split][src] = [pos, pos + counts[(split, src)]]
                    pos += counts[(split, src)]
        np.savez(staging / "train_docs.npz", starts=np.asarray(doc_starts, dtype=np.int64),
                 source=np.asarray(doc_source, dtype=np.uint8), sources=np.asarray(sources))

        split_tokens = {s: sum(v for (sp, _), v in counts.items() if sp == s) for s in ("train", "val", "test")}
        meta = {
            "dtype": np.dtype(dtype).name,
            "vocab_size": tok.vocab_size,
            "max_token_id": max_id,
            "eot_id": tok.eot_id,
            "tokenizer_spec": tok.spec,
            "track": "huggingface",
            "tokens": split_tokens,
            "documents": {s: sum(v for (sp, _), v in documents.items() if sp == s) for s in split_tokens},
            "sources": sources,
            "tokens_by_source": {f"{sp}/{src}": v for (sp, src), v in counts.items() if src},
            "source": settings,
            "split_method": "sha256(seed + NUL + stripped_text), first 64 bits",
        }
        (staging / "domain_offsets.json").write_text(json.dumps(offsets, indent=2), encoding="utf-8")
        (staging / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for name in ["train.bin", "val.bin", "test.bin", "train_docs.npz", "domain_offsets.json", "meta.json"]:
            os.replace(staging / name, out_dir / name)
    print(f"Prepared {split_tokens} as {np.dtype(dtype).name} in {out_dir}", flush=True)
    return meta


def main():
    from src.tracking import load_training_env

    load_training_env()  # HF_TOKEN from .env: authenticated Hub requests get higher rate limits
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", help="override the YAML data_dir")
    ap.add_argument("--train_tokens", type=int, help="override for a small preparation smoke test")
    ap.add_argument("--val_tokens", type=int)
    ap.add_argument("--test_tokens", type=int)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    settings = load_yaml(args.config)["data"]
    for split in ("train", "val", "test"):
        value = getattr(args, f"{split}_tokens")
        if value is not None:
            settings[f"{split}_tokens"] = value
    required = {"dataset", "subset", "revision", "split", "text_column", "tokenizer",
                "train_tokens", "val_tokens", "test_tokens", "val_fraction", "test_fraction",
                "shuffle_buffer", "seed"}
    optional = {"source_column", "sources"}
    if not required <= set(settings) <= required | optional:
        raise ValueError(f"data keys: missing={required - set(settings)}, "
                         f"unknown={set(settings) - required - optional}")
    if not isinstance(settings["shuffle_buffer"], int) or settings["shuffle_buffer"] < 1:
        raise ValueError("shuffle_buffer must be a positive integer")
    out_dir = args.out_dir or cfg.data_dir
    if Path(out_dir).exists() and any(Path(out_dir).iterdir()):
        raise FileExistsError(f"{out_dir} must be empty; choose a new output directory")

    from datasets import load_dataset

    tok = load_tokenizer(settings["tokenizer"])
    if cfg.model.doc_mask and cfg.model.eos_id != tok.eot_id:
        raise ValueError(f"model.eos_id={cfg.model.eos_id} but the tokenizer's EOS is {tok.eot_id}")
    rows = load_dataset(
        settings["dataset"], name=settings["subset"], revision=settings["revision"],
        split=settings["split"], streaming=True,
    ).shuffle(seed=settings["seed"], buffer_size=settings["shuffle_buffer"])
    prepare_rows(rows, tok, out_dir, settings, model_vocab=cfg.model.vocab_size,
                 seq_len=cfg.model.seq_len)


if __name__ == "__main__":
    main()
