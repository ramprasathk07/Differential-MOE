"""Train a byte-level BPE tokenizer on the BabyLM Challenge corpus, with an
adaptive vocab-size sweep (docs/plan.md SS2): score candidate vocabs by
fertility (tokens/word) and byte-compression ratio, then keep the requested
size or print the sweep table so you can pick the knee of the curve yourself.

BabyLM's six domains (bnc_spoken, childes, gutenberg, open_subtitles,
simple_wiki, switchboard) are more heterogeneous than a single-genre corpus,
so re-run the sweep here rather than assuming a vocab size chosen for a
different dataset still applies.

A pretrained frontier tokenizer can be scored in the same sweep via
`--pretrained`, which is how you justify picking one over a custom BPE: it
shows the fertility gap on *this* corpus rather than assuming a bigger vocab
is better. Pretrained specs need no training step -- pass them straight to
`src.data.prepare --tokenizer hf:<id>`.

Usage:
    # sweep candidates, print scoring table, exit (no file written)
    python -m src.data.train_tokenizer --sweep --candidates 2048 4096 8192 16384

    # compare custom vocabs against frontier tokenizers on the same dev text
    python -m src.data.train_tokenizer --sweep --candidates 4096 16384 \
        --pretrained hf:gpt2 hf:Xenova/gpt-4 hf:Qwen/Qwen2.5-0.5B

    # train and save the chosen vocab size (tier A: strict-small, tier B: strict)
    python -m src.data.train_tokenizer --vocab_size 4096 --out data/tokenizer.json --track strict-small
"""

import argparse
import os

from tokenizers import ByteLevelBPETokenizer

from .babylm import iter_domains
from .tokenizer import EOT, TokenizerAdapter, load_tokenizer, token_dtype


def train_texts(track: str, max_lines: int):
    for _domain, lines in iter_domains(track, "train", max_lines):
        yield "\n".join(lines)


def dev_texts(max_lines: int):
    for _domain, lines in iter_domains("strict-small", "dev", max_lines):
        yield "\n".join(lines)


def train_bpe(vocab_size: int, track: str, max_lines: int) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        train_texts(track, max_lines),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT],
    )
    return tok


def score_tokenizer(tok: TokenizerAdapter, eval_texts: list) -> dict:
    """Fertility (tokens/word, lower=better) and byte-compression (bytes/token,
    higher=more compressed) on held-out (dev) text."""
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    for text in eval_texts:
        ids = tok.encode(text)
        total_tokens += len(ids)
        total_words += len(text.split())
        total_bytes += len(text.encode("utf-8"))
    return {
        "vocab_size": tok.vocab_size,
        "fertility": total_tokens / max(total_words, 1),
        "bytes_per_token": total_bytes / max(total_tokens, 1),
    }


def sweep(candidates: list, pretrained: list, track: str, max_lines: int, eval_max_lines: int):
    eval_set = list(dev_texts(eval_max_lines))
    print(f"{'tokenizer':>26}{'vocab':>9}{'fertility':>12}{'bytes/token':>14}{'storage':>9}")
    for vs in candidates:
        tok = TokenizerAdapter(train_bpe(vs, track, max_lines), "custom", f"custom-{vs}")
        s = score_tokenizer(tok, eval_set)
        dt = token_dtype(s["vocab_size"]).__name__
        print(f"{'custom BPE':>26}{s['vocab_size']:>9}{s['fertility']:>12.3f}"
              f"{s['bytes_per_token']:>14.3f}{dt:>9}")
    for spec in pretrained or []:
        s = score_tokenizer(load_tokenizer(spec), eval_set)
        dt = token_dtype(s["vocab_size"]).__name__
        print(f"{spec:>26}{s['vocab_size']:>9}{s['fertility']:>12.3f}"
              f"{s['bytes_per_token']:>14.3f}{dt:>9}")
    print(
        "\nCustom vocabs: pick where fertility stops dropping much per doubling (the knee).\n"
        "Pretrained: lower fertility is real, but weigh it against embedding cost --\n"
        "vocab x dim params, which dominates a small model (see docs/plan.md SS2).\n"
        "uint32 storage doubles the .bin size versus uint16."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="score candidates and exit, no file written")
    ap.add_argument("--candidates", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--pretrained", type=str, nargs="*", default=[],
                     help="also score these in the sweep, e.g. hf:Xenova/gpt-4 hf:Qwen/Qwen2.5-0.5B")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--track", type=str, default="strict-small", choices=["strict-small", "strict"],
                     help="strict-small (10M words, tier A) or strict (100M words, tier B)")
    ap.add_argument("--out", type=str, default="data/tokenizer.json")
    ap.add_argument("--max_lines_per_domain", type=int, default=0,
                     help="cap lines read per domain for training; 0 = all (full track)")
    ap.add_argument("--eval_max_lines_per_domain", type=int, default=2000,
                     help="cap lines per domain from the dev set used for sweep scoring")
    args = ap.parse_args()

    if args.sweep:
        sweep(args.candidates, args.pretrained, args.track,
              args.max_lines_per_domain, args.eval_max_lines_per_domain)
        return

    tok = train_bpe(args.vocab_size, args.track, args.max_lines_per_domain)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"saved tokenizer: vocab={tok.get_vocab_size()} track={args.track} -> {args.out}")


if __name__ == "__main__":
    main()
