"""BLiMP: does the perplexity win translate into grammatical competence?

This project borrowed BabyLM's corpus but not its evaluation. BabyLM scores models
on BLiMP -- 67 paradigms of minimal pairs, each a grammatical sentence next to a
minimally different ungrammatical one. A model "gets it right" when it assigns the
grammatical sentence higher total log-probability.

That measures something perplexity does not. Perplexity rewards predicting frequent
continuations; BLiMP asks whether the model has internalised a syntactic constraint.
A mechanism can buy a lot of the first while buying none of the second, and the
question of whether MoE's large NLL win transfers is genuinely open.

Protocol: standard full-sentence log-likelihood ("simple LM method"). Tokens are
scored from the second onward, since the first has no context to be predicted from.
Sums, not per-token averages -- BLiMP pairs are near length-matched by construction
and the accepted protocol compares totals.

    python scripts/eval_blimp.py
"""

import gc
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import Transformer                       # noqa: E402
from scripts.eval_all import (                          # noqa: E402
    infer_config, best_checkpoint, DATA_DIR, CKPT_ROOT, SEQ_LEN, DEVICE, RUNS,
)

OUT_JSON = os.path.join(ROOT, "docs/blog/runs_export/eval_blimp.json")
BATCH = 64
MAX_LEN = 128          # BLiMP sentences are short; this never truncates in practice


def load_blimp():
    from datasets import load_dataset, get_dataset_config_names
    names = get_dataset_config_names("nyu-mll/blimp")
    print(f"BLiMP: {len(names)} paradigms")
    pairs = []
    for n in names:
        ds = load_dataset("nyu-mll/blimp", n, split="train")
        for r in ds:
            pairs.append((r["sentence_good"], r["sentence_bad"],
                          n, r["linguistics_term"], r["field"]))
    print(f"  {len(pairs)} minimal pairs\n")
    return pairs


@torch.no_grad()
def sentence_logprobs(model, tok, sentences, pad_id=0):
    """Total log P(sentence) for each input, scoring from the 2nd token on."""
    out = np.empty(len(sentences), dtype=np.float64)
    encoded = [tok.encode(s)[:MAX_LEN] for s in sentences]
    order = np.argsort([len(e) for e in encoded])          # group similar lengths
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        chunk = [encoded[i] for i in idx]
        L = max(len(c) for c in chunk)
        x = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.bool)
        for j, c in enumerate(chunk):
            x[j, :len(c)] = torch.tensor(c)
            mask[j, :len(c)] = True
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        logits, _, _ = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        tgt = x[:, 1:]
        got = logp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, L-1)
        valid = mask[:, 1:]
        totals = (got * valid).sum(-1)
        for j, i in enumerate(idx):
            out[i] = totals[j].item()
        del logits, logp
    return out


def main():
    pairs = load_blimp()
    good = [p[0] for p in pairs]
    bad = [p[1] for p in pairs]
    uids = np.array([p[2] for p in pairs])
    terms = np.array([p[3] for p in pairs])
    fields = np.array([p[4] for p in pairs])

    from src.data.tokenizer import resolve_data_tokenizer
    tok = resolve_data_tokenizer(DATA_DIR)
    if tok is None:
        print("no tokenizer resolved; aborting")
        return

    results = {}
    for run in RUNS:
        ckpt_path = best_checkpoint(os.path.join(CKPT_ROOT, run))
        if ckpt_path is None:
            print(f"skip {run}: no checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        sd = ckpt["model"]
        cfg = infer_config(sd)
        model = Transformer(cfg).to(DEVICE)
        model.load_state_dict(sd, strict=False)
        del ckpt, sd
        gc.collect()
        model.eval()

        lg = sentence_logprobs(model, tok, good)
        lb = sentence_logprobs(model, tok, bad)
        correct = (lg > lb).astype(np.float64)

        by_term = {t: float(correct[terms == t].mean()) for t in np.unique(terms)}
        by_field = {f: float(correct[fields == f].mean()) for f in np.unique(fields)}
        by_uid = {u: float(correct[uids == u].mean()) for u in np.unique(uids)}
        overall = float(correct.mean())
        # paradigm-level macro average -- BLiMP's headline number
        macro = float(np.mean(list(by_uid.values())))

        results[run] = {"overall_micro": overall, "overall_macro": macro,
                        "by_field": by_field, "by_term": by_term, "by_paradigm": by_uid,
                        "n_pairs": int(len(correct))}
        print(f"{run:10s} BLiMP macro {macro*100:5.2f}%  micro {overall*100:5.2f}%")

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
