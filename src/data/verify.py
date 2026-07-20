"""Sanity-check a prepared data directory before trusting or uploading it.

Tokenized data is opaque -- a wrong dtype or a truncated write does not raise,
it just decodes as garbage and shows up hours into a training run as a loss
that will not come down. This checks the things that fail silently:

  * every split loads at the dtype meta.json claims
  * no token id exceeds the vocabulary (would index past the embedding table)
  * ids decode back to plausible text
  * domain offsets cover the file exactly, with no gaps or overruns
  * a config's vocab_size, if given, can actually cover the data

Usage:
    python -m src.data.verify --data_dir data_s
    python -m src.data.verify --data_dir data_s --config configs/s_dense.yaml
"""

import argparse
import json
import os

import numpy as np

from .dataset import load_tokens
from .tokenizer import resolve_data_tokenizer


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default=None,
                     help="optional: also check this config's vocab_size covers the data")
    args = ap.parse_args()

    problems = []
    meta_path = os.path.join(args.data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"no meta.json in {args.data_dir} -- was it prepared by src.data.prepare?")
    meta = json.load(open(meta_path))
    print(f"tokenizer : {meta['tokenizer_spec']}")
    print(f"vocab     : {meta['vocab_size']} (max id seen {meta['max_token_id']})")
    print(f"dtype     : {meta['dtype']}")
    print(f"track     : {meta['track']}\n")

    tok = resolve_data_tokenizer(args.data_dir)
    if tok is None:
        problems.append("tokenizer could not be resolved from meta.json")

    offsets = {}
    off_path = os.path.join(args.data_dir, "domain_offsets.json")
    if os.path.exists(off_path):
        offsets = json.load(open(off_path))

    total_tokens = 0
    for split, fname in [("train", "train.bin"), ("val", "val.bin"), ("test", "test.bin")]:
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            problems.append(f"{fname} missing")
            continue

        data = load_tokens(path)
        size = os.path.getsize(path)
        total_tokens += len(data)
        mx = int(data.max())

        print(f"{split:>5}: {len(data):>12,} tokens  {human(size):>9}  "
              f"dtype={data.dtype}  max_id={mx}")

        if str(data.dtype) != meta["dtype"]:
            problems.append(f"{split}: dtype {data.dtype} != meta.json {meta['dtype']}")
        if mx >= meta["vocab_size"]:
            problems.append(f"{split}: token id {mx} >= vocab {meta['vocab_size']}")
        if size != len(data) * np.dtype(data.dtype).itemsize:
            problems.append(f"{split}: file size inconsistent with token count (truncated write?)")
        expected = meta.get("tokens", {}).get(split)
        if expected is not None and expected != len(data):
            problems.append(f"{split}: {len(data)} tokens but meta.json says {expected}")

        # domain offsets must tile the file exactly
        if split in offsets:
            spans = sorted(offsets[split].values())
            if spans[0][0] != 0:
                problems.append(f"{split}: domain offsets start at {spans[0][0]}, not 0")
            if spans[-1][1] != len(data):
                problems.append(f"{split}: domain offsets end at {spans[-1][1]}, file has {len(data)}")
            for a, b in zip(spans, spans[1:]):
                if a[1] != b[0]:
                    problems.append(f"{split}: gap/overlap in domain offsets at {a[1]} -> {b[0]}")

        # decoded text should look like language, not noise
        if tok is not None:
            text = tok.decode(data[:60].astype(int).tolist())
            printable = sum(c.isprintable() or c.isspace() for c in text) / max(len(text), 1)
            if printable < 0.9:
                problems.append(f"{split}: decoded sample is mostly unprintable -- wrong dtype?")
            print(f"       sample: {text[:100]!r}")

    print(f"\ntotal: {total_tokens:,} tokens")

    if args.config:
        import yaml

        cfg_vocab = yaml.safe_load(open(args.config))["model"]["vocab_size"]
        if meta["max_token_id"] >= cfg_vocab:
            problems.append(f"{args.config} vocab_size {cfg_vocab} cannot cover max id "
                            f"{meta['max_token_id']} (need >= {meta['max_token_id'] + 1})")
        else:
            print(f"config: {args.config} vocab_size {cfg_vocab} covers max id "
                  f"{meta['max_token_id']}  OK")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nall checks passed -- safe to upload")


if __name__ == "__main__":
    main()
