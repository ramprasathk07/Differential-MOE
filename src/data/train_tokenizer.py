"""Train a byte-level BPE tokenizer on TinyStories, with an adaptive vocab-size
sweep (docs/plan.md SS2): score candidate vocabs by fertility (tokens/word) and
byte-compression ratio, then keep the requested size or print the sweep table
so you can pick the knee of the curve yourself.

Usage:
    # sweep candidates, print scoring table, exit (no file written)
    python -m src.data.train_tokenizer --sweep --candidates 2048 4096 8192 16384

    # train and save the chosen vocab size
    python -m src.data.train_tokenizer --vocab_size 4096 --out data/tokenizer.json
"""

import argparse
import os

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

EOT = "<|endoftext|>"


def story_iterator(max_stories: int, split: str = "train"):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for i, item in enumerate(ds):
        if max_stories and i >= max_stories:
            break
        text = item.get("text", "")
        if text:
            yield text


def train_bpe(vocab_size: int, max_stories: int) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        story_iterator(max_stories),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT],
    )
    return tok


def score_tokenizer(tok: ByteLevelBPETokenizer, eval_stories: list) -> dict:
    """Fertility (tokens/word, lower=better) and byte-compression (bytes/token,
    higher=more compressed) on a held-out sample."""
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    for text in eval_stories:
        ids = tok.encode(text).ids
        total_tokens += len(ids)
        total_words += len(text.split())
        total_bytes += len(text.encode("utf-8"))
    return {
        "vocab_size": tok.get_vocab_size(),
        "fertility": total_tokens / max(total_words, 1),
        "bytes_per_token": total_bytes / max(total_tokens, 1),
    }


def sweep(candidates: list, train_stories: int, eval_stories: int):
    eval_set = list(story_iterator(eval_stories, split="validation"))
    print(f"{'vocab_size':>12}{'fertility':>12}{'bytes/token':>14}")
    for vs in candidates:
        tok = train_bpe(vs, train_stories)
        s = score_tokenizer(tok, eval_set)
        print(f"{s['vocab_size']:>12}{s['fertility']:>12.3f}{s['bytes_per_token']:>14.3f}")
    print(
        "\nPick the vocab where fertility stops dropping much per doubling "
        "(the knee) -- see docs/plan.md SS2."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="score candidates and exit, no file written")
    ap.add_argument("--candidates", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--out", type=str, default="data/tokenizer.json")
    ap.add_argument("--max_stories", type=int, default=200_000,
                    help="stories used for BPE training; 0 = all (slow)")
    ap.add_argument("--eval_stories", type=int, default=2000)
    args = ap.parse_args()

    if args.sweep:
        sweep(args.candidates, args.max_stories, args.eval_stories)
        return

    tok = train_bpe(args.vocab_size, args.max_stories)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"saved tokenizer: vocab={tok.get_vocab_size()} -> {args.out}")


if __name__ == "__main__":
    main()
