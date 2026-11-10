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

    # L tier: fit to the corpus named in a config's `data` section (SlimPajama),
    # with an equal byte budget per source so web text cannot crowd out code/math
    python -m src.data.train_tokenizer --hf_config configs/l_base.yaml --sweep         --candidates 16384 32768 49152 --pretrained hf:HuggingFaceTB/SmolLM2-135M
    python -m src.data.train_tokenizer --hf_config configs/l_base.yaml --vocab_size 32768         --out tokenizers/slimpajama_bpe32k.json
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


def hf_texts(config_path: str, bytes_per_source: int, eval_bytes_per_source: int):
    """Equal-size per-source samples of the config's corpus: train texts from the
    train hash split, held-out texts from the val split (never trained on)."""
    from datasets import load_dataset

    from src.model.config import load_yaml
    from .prepare_hf import _source, document_split

    d = load_yaml(config_path)["data"]
    sources = list(d.get("sources") or [d["dataset"]])
    want = {"train": bytes_per_source, "val": eval_bytes_per_source}
    texts = {split: {s: [] for s in sources} for split in want}
    used = {split: dict.fromkeys(sources, 0) for split in want}
    rows = load_dataset(d["dataset"], name=d["subset"], revision=d["revision"],
                        split=d["split"], streaming=True).shuffle(seed=d["seed"], buffer_size=d["shuffle_buffer"])
    for row in rows:
        text = row[d["text_column"]]
        src = _source(row, d["source_column"]) if d.get("source_column") else sources[0]
        if src not in used["train"] or not isinstance(text, str) or not text.strip():
            continue
        split = document_split(text, d["seed"], d["val_fraction"], d["test_fraction"])
        if split == "test" or used[split][src] >= want[split]:
            continue
        texts[split][src].append(text)
        used[split][src] += len(text.encode("utf-8"))
        if all(used[s][x] >= want[s] for s in want for x in sources):
            break
    print("sampled bytes:", used)
    return texts["train"], texts["val"]


def train_bpe(vocab_size: int, track: str, max_lines: int, texts=None) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        texts if texts is not None else train_texts(track, max_lines),
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


def sweep(candidates: list, pretrained: list, track: str, max_lines: int, eval_max_lines: int,
          train_set=None, eval_sets=None):
    """eval_sets: {name: [texts]} scored separately (per source); None = BabyLM dev."""
    eval_sets = eval_sets or {"babylm-dev": list(dev_texts(eval_max_lines))}
    tokenizers = [(f"custom BPE {vs}", TokenizerAdapter(
        train_bpe(vs, track, max_lines, None if train_set is None else iter(train_set)), "custom", f"custom-{vs}"))
        for vs in candidates]
    tokenizers += [(spec, load_tokenizer(spec)) for spec in pretrained or []]
    print(f"{'tokenizer':>30}{'eval set':>24}{'vocab':>9}{'fertility':>12}{'bytes/token':>14}{'storage':>9}")
    for name, tok in tokenizers:
        for set_name, texts in eval_sets.items():
            s = score_tokenizer(tok, texts)
            dt = token_dtype(s["vocab_size"]).__name__
            print(f"{name:>30}{set_name:>24}{s['vocab_size']:>9}{s['fertility']:>12.3f}"
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
    ap.add_argument("--hf_config", default=None,
                     help="train on the corpus in this YAML's `data` section instead of BabyLM")
    ap.add_argument("--mb_per_source", type=int, default=200, help="--hf_config training sample per source")
    ap.add_argument("--eval_mb_per_source", type=int, default=4, help="--hf_config held-out sample per source")
    args = ap.parse_args()

    train_set = eval_sets = None
    if args.hf_config:
        by_source, held_out = hf_texts(args.hf_config, args.mb_per_source * 2**20, args.eval_mb_per_source * 2**20)
        train_set = [t for texts in by_source.values() for t in texts]
        eval_sets = held_out

    if args.sweep:
        sweep(args.candidates, args.pretrained, args.track,
              args.max_lines_per_domain, args.eval_max_lines_per_domain, train_set, eval_sets)
        return

    tok = train_bpe(args.vocab_size, args.track, args.max_lines_per_domain,
                    None if train_set is None else iter(train_set))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    source = args.hf_config or f"babylm {args.track}"
    print(f"saved tokenizer: vocab={tok.get_vocab_size()} source={source} -> {args.out}")


if __name__ == "__main__":
    main()
