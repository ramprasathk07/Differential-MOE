"""High-intensity evaluation: metrics that test *mechanism*, not just quality.

NLL tells you a model is better. It does not tell you why, or whether the thing
you built does what it claims. Every metric here comes from the same forward pass
over the same byte-identical windows already used by eval_all.py, so the marginal
cost is one extra pass per model.

  1. NLL vs position in the context window
     Differential attention's whole claim is that softmax leaks attention onto
     irrelevant positions and that the leak *accumulates with context*. If that is
     true, its advantage over standard attention must GROW with position index.
     This is the closest thing to a direct test of the mechanism available without
     retraining at longer seq_len -- and it is a falsifiable prediction either way.

  2. Effective context use
     Same idea from the other side: re-score late positions with the context
     truncated to k tokens. A model that genuinely exploits distant context loses
     more when you take it away.

  3. Predictive entropy and calibration (ECE)
     Two models with equal NLL can differ sharply in how confident they are and
     whether that confidence is earned. Orthogonal to NLL, free to compute.

  4. NLL stratified by token frequency
     Rare tokens are where capacity shows up; frequent tokens are where fluency
     does. A mixture of experts and a sharper attention should move different
     deciles, which is the token-level analogue of the per-domain result.

    python scripts/eval_deep.py
"""

import gc
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import Transformer                       # noqa: E402
from src.data import load_tokens                        # noqa: E402
from scripts.eval_all import (                          # noqa: E402
    infer_config, best_checkpoint, window_offsets,
    DATA_DIR, CKPT_ROOT, SEQ_LEN, DEVICE, BATCH, RUNS,
)

OUT_JSON = os.path.join(ROOT, "docs/blog/runs_export/eval_deep.json")
N_WINDOWS = 1600           # half of eval_all's 3200 -- this pass is heavier per window
POS_BUCKETS = 32           # 512 positions / 32 = 16 positions per bucket
N_CAL_BINS = 15            # calibration reliability bins
TRUNC_K = [16, 32, 64, 128, 256, 511]   # context lengths for the truncation probe
FREQ_DECILES = 10


def token_frequency_table(train_path, n_sample=20_000_000):
    """Unigram counts from a prefix of the training stream, used only to bucket
    test tokens into frequency deciles."""
    data = load_tokens(train_path)
    sample = np.asarray(data[:min(n_sample, len(data))])
    counts = Counter(sample.tolist())
    return counts


@torch.no_grad()
def deep_pass(model, data, offsets):
    """One pass collecting position-resolved loss, entropy, calibration, and the
    per-token loss/id pairs needed for frequency stratification."""
    pos_loss = np.zeros(SEQ_LEN, dtype=np.float64)
    pos_ent = np.zeros(SEQ_LEN, dtype=np.float64)
    pos_n = 0
    conf_sum = np.zeros(N_CAL_BINS)
    corr_sum = np.zeros(N_CAL_BINS)
    bin_n = np.zeros(N_CAL_BINS)
    tok_ids, tok_loss = [], []

    for b in range(0, len(offsets), BATCH):
        chunk = offsets[b:b + BATCH]
        x = torch.stack([torch.from_numpy(data[i:i + SEQ_LEN].astype("int64")) for i in chunk]).to(DEVICE)
        y = torch.stack([torch.from_numpy(data[i + 1:i + SEQ_LEN + 1].astype("int64")) for i in chunk]).to(DEVICE)
        logits, _, _ = model(x)
        logits = logits.float()

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none").view(y.shape)          # (B, T)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)                                        # (B, T)
        conf, pred = p.max(-1)
        correct = (pred == y).float()

        pos_loss += loss.sum(0).double().cpu().numpy()
        pos_ent += ent.sum(0).double().cpu().numpy()
        pos_n += x.shape[0]

        # calibration: bucket every prediction by its confidence
        cb = torch.clamp((conf * N_CAL_BINS).long(), max=N_CAL_BINS - 1).view(-1)
        conf_f, corr_f = conf.view(-1), correct.view(-1)
        conf_sum += torch.zeros(N_CAL_BINS, device=DEVICE).index_add_(0, cb, conf_f).cpu().numpy()
        corr_sum += torch.zeros(N_CAL_BINS, device=DEVICE).index_add_(0, cb, corr_f).cpu().numpy()
        bin_n += torch.zeros(N_CAL_BINS, device=DEVICE).index_add_(
            0, cb, torch.ones_like(conf_f)).cpu().numpy()

        # keep a subsample of (token id, loss) for frequency deciles
        tok_ids.append(y.view(-1)[::7].cpu().numpy())
        tok_loss.append(loss.view(-1)[::7].float().cpu().numpy())

        del logits, logp, p, ent, loss
    return dict(pos_loss=pos_loss / pos_n, pos_ent=pos_ent / pos_n,
                conf_sum=conf_sum, corr_sum=corr_sum, bin_n=bin_n,
                tok_ids=np.concatenate(tok_ids), tok_loss=np.concatenate(tok_loss))


@torch.no_grad()
def truncation_probe(model, data, offsets, ks, n_windows=320):
    """NLL of the FINAL predicted token when the model may only see the last k
    tokens of context. A model that uses distant context degrades more as k shrinks."""
    out = {}
    offs = offsets[:n_windows]
    for k in ks:
        total, n = 0.0, 0
        for b in range(0, len(offs), BATCH):
            chunk = offs[b:b + BATCH]
            # take the k tokens ending just before the final target
            x = torch.stack([torch.from_numpy(data[i + SEQ_LEN - k:i + SEQ_LEN].astype("int64"))
                             for i in chunk]).to(DEVICE)
            y = torch.stack([torch.from_numpy(data[i + SEQ_LEN - k + 1:i + SEQ_LEN + 1].astype("int64"))
                             for i in chunk]).to(DEVICE)
            logits, _, _ = model(x)
            last = F.cross_entropy(logits.float()[:, -1, :], y[:, -1], reduction="sum")
            total += last.item()
            n += x.shape[0]
            del logits
        out[k] = total / n
    return out


def main():
    test = load_tokens(os.path.join(DATA_DIR, "test.bin"))
    offsets = window_offsets(0, len(test), N_WINDOWS)
    print(f"{len(offsets)} windows x {SEQ_LEN} tokens on {DEVICE}")
    print("building unigram frequency table from train.bin ...")
    counts = token_frequency_table(os.path.join(DATA_DIR, "train.bin"))
    print(f"  {len(counts)} distinct token ids seen\n")

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

        d = deep_pass(model, test, offsets)
        trunc = truncation_probe(model, test, offsets, TRUNC_K)

        # position buckets
        per = SEQ_LEN // POS_BUCKETS
        pos_bucket = d["pos_loss"].reshape(POS_BUCKETS, per).mean(1)
        ent_bucket = d["pos_ent"].reshape(POS_BUCKETS, per).mean(1)

        # calibration -> ECE
        nz = d["bin_n"] > 0
        acc = np.zeros(N_CAL_BINS); cnf = np.zeros(N_CAL_BINS)
        acc[nz] = d["corr_sum"][nz] / d["bin_n"][nz]
        cnf[nz] = d["conf_sum"][nz] / d["bin_n"][nz]
        w = d["bin_n"] / d["bin_n"].sum()
        ece = float((w * np.abs(acc - cnf)).sum())

        # frequency deciles
        freqs = np.array([counts.get(int(t), 0) for t in d["tok_ids"]], dtype=np.float64)
        order = np.argsort(freqs)
        dec_edges = np.array_split(order, FREQ_DECILES)
        dec_loss = [float(d["tok_loss"][idx].mean()) for idx in dec_edges]
        dec_medfreq = [float(np.median(freqs[idx])) for idx in dec_edges]

        results[run] = dict(
            overall_nll=float(d["pos_loss"].mean()),
            pos_bucket_nll=pos_bucket.tolist(),
            pos_bucket_entropy=ent_bucket.tolist(),
            nll_first_16=float(d["pos_loss"][:16].mean()),
            nll_last_16=float(d["pos_loss"][-16:].mean()),
            mean_pred_entropy=float(d["pos_ent"].mean()),
            ece=ece,
            calibration_acc=acc.tolist(),
            calibration_conf=cnf.tolist(),
            calibration_n=d["bin_n"].tolist(),
            truncation_nll={str(k): v for k, v in trunc.items()},
            freq_decile_nll=dec_loss,
            freq_decile_median_count=dec_medfreq,
        )
        print(f"{run:10s} NLL {results[run]['overall_nll']:.4f} | "
              f"pos 0-15 {results[run]['nll_first_16']:.4f} -> 496-511 {results[run]['nll_last_16']:.4f} "
              f"(drop {results[run]['nll_first_16']-results[run]['nll_last_16']:+.4f}) | "
              f"ECE {ece:.4f} | H {results[run]['mean_pred_entropy']:.3f}")

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
