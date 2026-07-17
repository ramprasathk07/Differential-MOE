"""Tokenize TinyStories once into uint16 memmap files (train.bin / val.bin).

Usage:
    python -m src.data.prepare --tokenizer data/tokenizer.json --out_dir data \
        [--max_stories 0]

Output files are flat streams of token ids with <|endoftext|> between stories.
uint16 is safe for vocab sizes < 65536. Roughly ~460M tokens (~0.9 GB) for the
full dataset with a 4k vocab — upload the resulting files as a Kaggle Dataset.
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from .train_tokenizer import EOT


def encode_split(split: str, tok: Tokenizer, out_path: str, max_stories: int) -> int:
    eot_id = tok.token_to_id(EOT)
    assert eot_id is not None, "tokenizer missing <|endoftext|>"
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    buf: list[int] = []
    n_tokens = 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for i, item in enumerate(ds):
            if max_stories and i >= max_stories:
                break
            text = item.get("text", "")
            if not text:
                continue
            buf.extend(tok.encode(text).ids)
            buf.append(eot_id)
            if len(buf) >= 1_000_000:
                arr = np.asarray(buf, dtype=np.uint16)
                arr.tofile(f)
                n_tokens += len(buf)
                buf = []
                if n_tokens % 10_000_000 < 1_000_000:
                    print(f"  {split}: {n_tokens/1e6:.0f}M tokens...")
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            n_tokens += len(buf)
    os.replace(tmp_path, out_path)
    print(f"{split}: {n_tokens:,} tokens -> {out_path}")
    return n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--max_stories", type=int, default=0, help="0 = full dataset")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size() < 65536, "uint16 memmap requires vocab < 65536"
    os.makedirs(args.out_dir, exist_ok=True)
    encode_split("train", tok, os.path.join(args.out_dir, "train.bin"), args.max_stories)
    encode_split("validation", tok, os.path.join(args.out_dir, "val.bin"), args.max_stories)


if __name__ == "__main__":
    main()
