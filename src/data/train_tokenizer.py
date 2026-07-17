"""Train a small byte-level BPE tokenizer on TinyStories.

Usage:
    python -m src.data.train_tokenizer --vocab_size 4096 --out data/tokenizer.json \
        [--max_stories 200000]

A 4k vocab keeps embedding params tiny so almost all parameters live in the
transformer body; TinyStories' vocabulary is small enough that coverage stays good.
"""

import argparse
import os

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

EOT = "<|endoftext|>"


def story_iterator(max_stories: int):
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, item in enumerate(ds):
        if max_stories and i >= max_stories:
            break
        text = item.get("text", "")
        if text:
            yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--out", type=str, default="data/tokenizer.json")
    ap.add_argument("--max_stories", type=int, default=200_000,
                    help="stories used for BPE training; 0 = all (slow)")
    args = ap.parse_args()

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        story_iterator(args.max_stories),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=[EOT],
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"saved tokenizer: vocab={tok.get_vocab_size()} -> {args.out}")


if __name__ == "__main__":
    main()
