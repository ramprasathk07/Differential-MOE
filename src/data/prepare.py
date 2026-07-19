"""Tokenize the BabyLM corpus once into uint16 memmap files: train.bin, val.bin
(from the shared dev split -- used for periodic in-training validation and
checkpoint selection), and test.bin (from the shared, held-out test split --
touch this only ONCE, after training is finished and a checkpoint is already
chosen; never let it influence any training-time decision).

Each of the 6 domains is tokenized separately and concatenated with a single
<|endoftext|> at the *domain* boundary (not after every line -- BabyLM's lines
are short conversational/subtitle fragments that are naturally sequential
within a domain). Token-offset boundaries per domain are recorded in
domain_offsets.json -- useful later for checking whether MoE experts
specialize by domain (docs/plan.md).

Usage:
    python -m src.data.prepare --tokenizer data/tokenizer.json --out_dir data --track strict-small
"""

import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer

from .babylm import iter_domains
from .train_tokenizer import EOT


def encode_split(track: str, split: str, tok: Tokenizer, out_path: str, max_lines: int) -> dict:
    eot_id = tok.token_to_id(EOT)
    assert eot_id is not None, "tokenizer missing <|endoftext|>"

    offsets = {}
    pos = 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for domain, lines in iter_domains(track, split, max_lines):
            ids = tok.encode("\n".join(lines)).ids
            ids.append(eot_id)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            offsets[domain] = [pos, pos + len(ids)]
            pos += len(ids)
            print(f"  {split}/{domain}: {len(ids):,} tokens")
    os.replace(tmp_path, out_path)
    print(f"{split}: {pos:,} tokens total -> {out_path}")
    return offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--track", type=str, default="strict-small", choices=["strict-small", "strict"])
    ap.add_argument("--max_lines_per_domain", type=int, default=0, help="0 = full domain files")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size() < 65536, "uint16 memmap requires vocab < 65536"
    os.makedirs(args.out_dir, exist_ok=True)

    train_offsets = encode_split(
        args.track, "train", tok, os.path.join(args.out_dir, "train.bin"), args.max_lines_per_domain
    )
    val_offsets = encode_split(
        args.track, "dev", tok, os.path.join(args.out_dir, "val.bin"), args.max_lines_per_domain
    )
    test_offsets = encode_split(
        args.track, "test", tok, os.path.join(args.out_dir, "test.bin"), args.max_lines_per_domain
    )
    with open(os.path.join(args.out_dir, "domain_offsets.json"), "w") as f:
        json.dump({"train": train_offsets, "val": val_offsets, "test": test_offsets}, f, indent=2)


if __name__ == "__main__":
    main()
