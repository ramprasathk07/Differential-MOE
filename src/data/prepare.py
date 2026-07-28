"""Tokenize the BabyLM corpus once into memmap files: train.bin, val.bin
(from the shared dev split -- used for periodic in-training validation and
checkpoint selection), and test.bin (from the shared, held-out test split --
touch this only ONCE, after training is finished and a checkpoint is already
chosen; never let it influence any training-time decision).

Token width follows the vocab: uint16 below 65536, uint32 above (a frontier
tokenizer like cl100k or Qwen forces uint32, doubling the .bin size). The
chosen dtype is recorded in meta.json so `load_tokens` reads it back correctly
instead of guessing.

Each of the 6 domains is tokenized separately and concatenated with a single
<|endoftext|> at the *domain* boundary (not after every line -- BabyLM's lines
are short conversational/subtitle fragments that are naturally sequential
within a domain). Token-offset boundaries per domain are recorded in
domain_offsets.json -- useful later for checking whether MoE experts
specialize by domain (docs/plan.md).

Usage:
    python -m src.data.prepare --tokenizer data/tokenizer.json --out_dir data --track strict-small
    python -m src.data.prepare --tokenizer hf:Xenova/gpt-4 --out_dir data_s --track strict
"""

import argparse
import json
import os

import numpy as np

from .babylm import iter_domains
from .tokenizer import TokenizerAdapter, load_tokenizer, token_dtype


CHUNK_LINES = 50_000


def encode_split(track: str, split: str, tok: TokenizerAdapter, out_path: str,
                 max_lines: int, dtype) -> tuple:
    """Encode and write one chunk of lines at a time.

    Encoding a whole domain in a single call would build a >100MB string and a
    tens-of-millions-long Python int list before anything reaches disk, which
    spikes memory into the gigabytes on the larger domains and gives no sign of
    progress. Chunking bounds both.

    Chunks split on line boundaries and carry the "\\n" that would have joined
    them, so the text encoded is identical to the single-call version; only BPE
    merges that would have spanned a chunk edge differ, and every run reads the
    same file, so the ablation is unaffected.
    """
    eot_id = tok.eot_id
    offsets = {}
    pos = 0
    max_id = 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for domain, lines in iter_domains(track, split, max_lines):
            start = pos
            for i in range(0, len(lines), CHUNK_LINES):
                text = "\n".join(lines[i:i + CHUNK_LINES])
                if i + CHUNK_LINES < len(lines):
                    text += "\n"          # the separator joining this chunk to the next
                arr = np.asarray(tok.encode(text), dtype=dtype)
                arr.tofile(f)
                if arr.size:
                    max_id = max(max_id, int(arr.max()))
                pos += arr.size
                print(f"    {split}/{domain}: {min(i + CHUNK_LINES, len(lines)):,}/{len(lines):,} lines, "
                      f"{pos - start:,} tokens", flush=True)
            np.asarray([eot_id], dtype=dtype).tofile(f)   # domain boundary
            pos += 1
            max_id = max(max_id, eot_id)
            offsets[domain] = [start, pos]
            print(f"  {split}/{domain}: {pos - start:,} tokens", flush=True)
    os.replace(tmp_path, out_path)
    print(f"{split}: {pos:,} tokens total -> {out_path}", flush=True)
    return offsets, pos, max_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                     help="path to a custom tokenizer.json, or hf:<id> / tiktoken:<encoding>")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--track", type=str, default="strict-small", choices=["strict-small", "strict"])
    ap.add_argument("--max_lines_per_domain", type=int, default=0, help="0 = full domain files")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    dtype = token_dtype(tok.vocab_size)
    print(f"tokenizer: {args.tokenizer} | vocab={tok.vocab_size} | storing as {np.dtype(dtype).name}")
    os.makedirs(args.out_dir, exist_ok=True)

    offsets, counts, max_ids = {}, {}, []
    for split, fname in [("train", "train.bin"), ("dev", "val.bin"), ("test", "test.bin")]:
        key = "val" if split == "dev" else split
        offsets[key], counts[key], mx = encode_split(
            args.track, split, tok, os.path.join(args.out_dir, fname),
            args.max_lines_per_domain, dtype,
        )
        max_ids.append(mx)

    # An id at or above vocab_size would index past the embedding table at train
    # time and crash mid-run; catch it here instead.
    max_id = max(max_ids)
    assert max_id < tok.vocab_size, (
        f"token id {max_id} >= declared vocab_size {tok.vocab_size} -- "
        f"the model's embedding table would be too small"
    )

    with open(os.path.join(args.out_dir, "domain_offsets.json"), "w") as f:
        json.dump(offsets, f, indent=2)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({
            "dtype": np.dtype(dtype).name,
            "vocab_size": tok.vocab_size,
            "max_token_id": max_id,
            "tokenizer_spec": args.tokenizer,
            "track": args.track,
            "tokens": counts,
        }, f, indent=2)
    print(f"\nset vocab_size: {tok.vocab_size} in your config (max id seen: {max_id})")


if __name__ == "__main__":
    main()
