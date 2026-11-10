"""Evaluate L-tier (or S-tier) checkpoints on one consumer GPU -- sized for an
RTX 3060 (12 GB) -- or on CPU.

Memory plan: the model is built on the meta device and the (mmap'd) checkpoint
tensors are assigned directly, so host RAM never holds an extra fp32 copy; all
parameters run in bf16 (a 2B-param MoE is ~4 GB), buffers (RoPE, router bias)
stay fp32, batch 1 x 4096 tokens. Document masking falls back to a dense SDPA
mask when Triton is missing (Windows), so no compiler is needed.

Suites (all run on byte-identical windows for every checkpoint):
  core         test NLL / ppl / bits-per-byte / top-1, global and per source;
               NLL by position; calibration (ECE) and entropy; NLL by token-frequency
               decile; routing entropy + expert-by-source specialisation; lambda,
               AttnRes depth weights, Derf/DyT scalars; per-window NLL; samples
  position     long-context: NLL by position 0..seq_len on windows cut from single
               long documents (no boundary inside), plus a context-truncation probe
  attention    effective attention sharpness (entropy, support, top-k mass,
               distance, negative mass), computed in query chunks to fit 12 GB
  blimp        BLiMP minimal pairs (67 paradigms)
  uncertainty  paired bootstrap of every run pair on the core windows: iid,
               source-stratified and source-clustered 95% CIs (needs `core`)

Usage:
    python scripts/eval_l.py --runs checkpoints/l_dense checkpoints/l_diff \
        checkpoints/l_moe checkpoints/l_diffmoe --data_dir data_l
    python scripts/eval_l.py --runs checkpoints/l_diffmoe --kind snapshot --suite core position
    python scripts/eval_l.py --from_mlflow <run_id> --data_dir data_l   # pull best weights logged to MLflow

Results: <out_dir>/<run>/<kind>_step<N>.json, <out_dir>/comparison.json and
<out_dir>/per_window_nll.npz. --mlflow logs test metrics back into each
training run (via <run_dir>/mlflow_run.json); --wandb logs one eval run.
"""

import argparse
import gc
import glob
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import load_tokens                                  # noqa: E402
from src.data.tokenizer import resolve_data_tokenizer             # noqa: E402
from src.eval import bytes_per_token_estimate, expert_stats       # noqa: E402
from src.model import ModelConfig, Transformer, count_params      # noqa: E402
from src.model.attention import DifferentialAttention, StandardAttention, apply_rope, build_rope_cache  # noqa: E402
from src.train import train_time_diagnostics                      # noqa: E402

PROMPTS = ["The history of the city", "def fibonacci(n):", "In this paper we prove that",
           "Once upon a time", "The results show"]
N_CAL_BINS = 15
FREQ_DECILES = 10


# ---------------------------------------------------------------- loading ---
def find_checkpoints(target, kind):
    """A .pt path, or a run dir -> its best (by val NLL), every snapshot, or last.pt."""
    if target.endswith(".pt"):
        return [target]
    if kind == "best":
        index = os.path.join(target, "best_index.json")
        if not os.path.isfile(index):
            return []
        with open(index) as f:
            entries = json.load(f)
        # recorded paths are from the training box; only the file name survives the copy
        local = [os.path.join(target, "best", os.path.basename(e["path"])) for e in entries[:1]]
        return [p for p in local if os.path.isfile(p)]
    if kind == "snapshot":
        snaps = glob.glob(os.path.join(target, "snapshots", "step*.pt"))
        return sorted(snaps, key=lambda p: int(os.path.basename(p)[4:-3]))
    last = os.path.join(target, "last.pt")
    return [last] if os.path.isfile(last) else []


def load_model(path, device, dtype):
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = ckpt["model"]
    if ckpt.get("model_config"):
        cfg = ModelConfig(**ckpt["model_config"])
    else:  # S-tier Kaggle checkpoints carry no config
        from scripts.eval_all import infer_config

        cfg = infer_config(sd)
    with torch.device("meta"):
        model = Transformer(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if m != "rope"]
    if missing or unexpected:
        raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
    model.rope = build_rope_cache(cfg.seq_len, cfg.dim // cfg.n_heads, cfg.rope_theta)
    for p in model.parameters():  # one dtype for every parameter; buffers stay fp32
        p.data = p.data.to(dtype)
    model.to(device).eval()
    info = {k: ckpt.get(k) for k in ("step", "tokens", "epoch", "best_val", "val_nll", "format")}
    del ckpt, sd
    return model, cfg, info


# ---------------------------------------------------------------- windows ---
def source_spans(data_dir, split="test"):
    path = os.path.join(data_dir, "domain_offsets.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get(split)


def core_windows(n_tokens, spans, seq_len, per_source):
    spans = spans or {"all": [0, n_tokens]}
    out = []
    for name, (lo, hi) in spans.items():
        total = max(0, (hi - lo - 1) // seq_len)
        n = min(per_source, total)
        out += [(lo + int(k * total / n) * seq_len, name) for k in range(n)]
    return out


def long_doc_windows(data, spans, seq_len, eos_id, per_source, sources=None):
    """Windows of seq_len+1 tokens lying inside one document (no EOS before the last target)."""
    spans = spans or {"all": [0, len(data)]}
    out = []
    for name, (lo, hi) in spans.items():
        if sources and name not in sources:
            continue
        seg = np.asarray(data[lo:hi])
        ends = np.flatnonzero(seg == eos_id) if eos_id >= 0 else np.array([len(seg) - 1])
        starts = np.concatenate([[0], ends[:-1] + 1])
        picked = [lo + s for s, e in zip(starts, ends) if e - s >= seq_len + 1]
        out += [(o, name) for o in picked[:per_source]]
    return out


def batches(data, windows, batch, seq_len, device):
    for b in range(0, len(windows), batch):
        chunk = windows[b:b + batch]
        x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i, _ in chunk])
        y = torch.stack([torch.from_numpy(data[i + 1:i + seq_len + 1].astype(np.int64)) for i, _ in chunk])
        yield x.to(device), y.to(device), [n for _, n in chunk]


# ------------------------------------------------------------------ suites ---
@torch.no_grad()
def suite_core(model, cfg, data, windows, args, bpt, freq, tok):
    names = sorted({n for _, n in windows})
    S = cfg.seq_len
    win_nll = np.empty(len(windows))
    pos_sum = np.zeros(S)
    src = {n: {"nll": 0.0, "tokens": 0, "correct": 0} for n in names}
    counts = {}  # (source, layer) -> expert counts
    conf_sum, corr_sum, bin_n = np.zeros(N_CAL_BINS), np.zeros(N_CAL_BINS), np.zeros(N_CAL_BINS)
    ent_sum = 0.0
    tok_ids, tok_loss = [], []
    i = 0
    for x, y, srcs in batches(data, windows, args.batch, S, args.device):
        logits = model(x)[0].float()
        logp = F.log_softmax(logits, -1)
        loss = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)                    # (B, S)
        p = logp.exp()
        ent_sum += float(-(p * logp).sum(-1).sum())
        conf, pred = p.max(-1)
        correct = (pred == y)
        cb = torch.clamp((conf * N_CAL_BINS).long(), max=N_CAL_BINS - 1).view(-1)
        conf_sum += torch.zeros(N_CAL_BINS, device=cb.device).index_add_(0, cb, conf.view(-1)).cpu().numpy()
        corr_sum += torch.zeros(N_CAL_BINS, device=cb.device).index_add_(0, cb, correct.view(-1).float()).cpu().numpy()
        bin_n += torch.bincount(cb, minlength=N_CAL_BINS).cpu().numpy()
        pos_sum += loss.sum(0).double().cpu().numpy()
        tok_ids.append(y.view(-1)[::7].cpu().numpy())
        tok_loss.append(loss.view(-1)[::7].cpu().numpy())
        for j, name in enumerate(srcs):
            win_nll[i + j] = float(loss[j].mean())
            src[name]["nll"] += float(loss[j].sum())
            src[name]["tokens"] += S
            src[name]["correct"] += int(correct[j].sum())
        for L, block in enumerate(model.blocks):
            if block.is_moe and block.ffn.last_counts is not None:
                # routing of a multi-window batch is pooled; batch 1 keeps it per source
                for name in set(srcs):
                    key = (name, L)
                    counts[key] = counts.get(key, 0) + block.ffn.last_counts.cpu().numpy()
        i += len(srcs)
        del logits, logp, p

    total_tok = sum(s["tokens"] for s in src.values())
    nll = sum(s["nll"] for s in src.values()) / total_tok
    nz = bin_n > 0
    acc = np.where(nz, corr_sum / np.maximum(bin_n, 1), 0)
    cnf = np.where(nz, conf_sum / np.maximum(bin_n, 1), 0)
    ece = float(((bin_n / bin_n.sum()) * np.abs(acc - cnf)).sum())
    out = {
        "nll": nll, "ppl": math.exp(min(nll, 50)), "bits_per_token": nll / math.log(2),
        "bits_per_byte": nll / math.log(2) / bpt if bpt else None,
        "top1": sum(s["correct"] for s in src.values()) / total_tok,
        "mean_pred_entropy": ent_sum / total_tok, "ece": ece,
        "n_windows": len(windows), "tokens": total_tok,
        "per_source": {n: {"nll": s["nll"] / s["tokens"], "top1": s["correct"] / s["tokens"],
                           "bits_per_byte": s["nll"] / s["tokens"] / math.log(2) / bpt if bpt else None}
                       for n, s in src.items() if s["tokens"]},
        "nll_by_position_64": [float(c.mean()) for c in np.array_split(pos_sum / len(windows), max(1, S // 64))],
    }
    if freq is not None:
        ids, losses = np.concatenate(tok_ids), np.concatenate(tok_loss)
        f = freq[ids]
        deciles = np.array_split(np.argsort(f, kind="stable"), FREQ_DECILES)
        out["freq_decile_nll"] = [float(losses[d].mean()) for d in deciles]
        out["freq_decile_median_count"] = [float(np.median(f[d])) for d in deciles]
    layers = sorted({L for _, L in counts})
    routing = {}
    for L in layers:
        pooled = sum(counts[(n, L)] for n in names if (n, L) in counts)
        ent, imb = expert_stats(torch.as_tensor(pooled))
        rows = {n: (counts[(n, L)] / max(1, counts[(n, L)].sum())).tolist() for n in names if (n, L) in counts}
        tv = float(np.mean([0.5 * np.abs(np.asarray(r) - 1 / len(r)).sum() for r in rows.values()]))
        routing[f"L{L}"] = {"entropy": ent, "imbalance": imb, "source_x_expert": rows, "specialization_tv": tv}
    out["routing"] = routing
    diag = train_time_diagnostics(model)
    out["lambda_per_head"] = diag["lambda_raw"]
    out["attnres_weights"] = diag["attnres_weights"]
    out["norm_params"] = diag["norm_params"]
    out["mhc_res_diag"] = diag["mhc_res_diag"]
    if tok is not None:
        torch.manual_seed(0)  # same seed for every run -> comparable samples
        gens = {}
        for prompt in PROMPTS:
            ids = torch.tensor([tok.encode(prompt)], device=args.device)
            gens[prompt] = tok.decode(model.generate(ids, max_new_tokens=args.gen_tokens)[0].tolist())
        out["samples"] = gens
    return out, win_nll


@torch.no_grad()
def suite_position(model, cfg, data, windows, args):
    S = cfg.seq_len
    by_src, n_src = {}, {}
    for x, y, srcs in batches(data, windows, args.batch, S, args.device):
        loss = F.cross_entropy(model(x)[0].float().transpose(1, 2), y, reduction="none").cpu().numpy()
        for j, name in enumerate(srcs):
            by_src[name] = by_src.get(name, 0) + loss[j]
            n_src[name] = n_src.get(name, 0) + 1
    if not by_src:
        return {"note": f"no test document is longer than {S} tokens"}
    overall = sum(by_src.values()) / sum(n_src.values())
    buckets = max(1, S // 64)
    out = {"n_windows": sum(n_src.values()), "bucket_tokens": S // buckets,
           "nll_by_position": [float(c.mean()) for c in np.array_split(overall, buckets)],
           "per_source": {n: {"n_windows": n_src[n],
                              "nll_by_position": [float(c.mean()) for c in np.array_split(v / n_src[n], buckets)]}
                          for n, v in by_src.items()}}
    # truncation probe: NLL of the last target when only the last k tokens are visible
    trunc = {}
    probe = windows[: args.trunc_windows]
    for k in [k for k in (64, 256, 1024, 2048, S - 1) if k < S] + [S]:
        total = 0.0
        for i, _ in probe:
            x = torch.from_numpy(data[i + S - k:i + S].astype(np.int64))[None].to(args.device)
            total += float(F.cross_entropy(model(x)[0][0, -1].float()[None], torch.tensor([int(data[i + S])], device=args.device)))
        trunc[str(k)] = total / max(1, len(probe))
    out["truncation_last_token_nll"] = trunc
    return out


ATTN_STATS = {}
CHUNK = 256
MIN_POS = 64
TOPK = (1, 8, 32)


def _row_stats(a, rows, differential):
    """Sums over rows of a (B, H, C, S) effective-attention chunk."""
    absa = a.abs()
    p = absa / absa.sum(-1, keepdim=True).clamp_min(1e-9)
    ent = -(p * (p + 1e-12).log()).sum(-1)
    k_idx = torch.arange(a.shape[-1], device=a.device).view(1, 1, 1, -1)
    dist = (rows.view(1, 1, -1, 1) - k_idx).clamp_min(0).float()
    out = {"entropy": ent.sum().item(), "attn_distance": (p * dist).sum(-1).sum().item(),
           "row_sum": a.sum(-1).sum().item(), "rows": float(ent.numel())}
    for k in TOPK:
        out[f"top{k}_mass"] = p.topk(min(k, p.shape[-1]), -1).values.sum(-1).sum().item()
    neg = a.clamp_max(0).abs().sum(-1) / absa.sum(-1).clamp_min(1e-9) if differential else torch.zeros_like(ent)
    out["negative_mass_frac"] = neg.sum().item()
    return out


def _attention_forward(self, x, rope, mask=None):
    """Same arithmetic as the model's forward, materialising attention per query chunk."""
    assert mask is None, "attention suite uses single-document windows (plain causal)"
    B, S, D = x.shape
    differential = isinstance(self, DifferentialAttention)
    H, hd = self.n_heads, self.head_dim
    width = 2 * hd if differential else hd
    q = self.wq(x).view(B, S, H, width)
    k = self.wk(x).view(B, S, H, width)
    v = self.wv(x).view(B, S, H, width).transpose(1, 2)
    if differential:
        (q1, q2), (k1, k2) = q.chunk(2, -1), k.chunk(2, -1)
        qs = [apply_rope(t, rope).transpose(1, 2).float() for t in (q1, q2)]
        ks = [apply_rope(t, rope).transpose(1, 2).float() for t in (k1, k2)]
        lam = self.current_lambda().view(1, H, 1, 1)
    else:
        qs = [apply_rope(q, rope).transpose(1, 2).float()]
        ks = [apply_rope(k, rope).transpose(1, 2).float()]
    stats = ATTN_STATS.setdefault(id(self), {})
    outs = []
    for c0 in range(0, S, CHUNK):
        rows = torch.arange(c0, min(S, c0 + CHUNK), device=x.device)
        causal = rows.view(-1, 1) >= torch.arange(S, device=x.device).view(1, -1)
        maps = [(qq[:, :, rows] @ kk.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)
                for qq, kk in zip(qs, ks)]
        a = maps[0] - lam * maps[1] if differential else maps[0]
        keep = rows >= min(MIN_POS, S // 4)  # early rows have too little context to be "spread out"
        if keep.any():
            for key, val in _row_stats(a[:, :, keep], rows[keep], differential).items():
                stats[key] = stats.get(key, 0.0) + val
        outs.append(a.to(v.dtype) @ v)
    o = torch.cat(outs, 2)
    if self.xsa:
        from src.model.attention import exclude_self

        o = exclude_self(o, v)
    if differential:
        o = self.subln(o) * (1.0 - self.lambda_init)
    return self.wo(o.transpose(1, 2).reshape(B, S, D))


@torch.no_grad()
def suite_attention(model, cfg, data, windows, args):
    original = StandardAttention.forward, DifferentialAttention.forward
    StandardAttention.forward = DifferentialAttention.forward = _attention_forward
    ATTN_STATS.clear()
    try:
        for x, _, _ in batches(data, windows[: args.attention_windows], 1, cfg.seq_len, args.device):
            model(x)
    finally:
        StandardAttention.forward, DifferentialAttention.forward = original
    layers = []
    for block in model.blocks:
        s = ATTN_STATS.get(id(block.attn), {})
        rows = s.pop("rows", 0.0) or 1.0
        layer = {k: v / rows for k, v in s.items()}
        layer["effective_support"] = math.exp(layer.get("entropy", 0.0))
        layers.append(layer)
    overall = {k: float(np.mean([l[k] for l in layers])) for k in layers[0]} if layers else {}
    return {"n_windows": min(len(windows), args.attention_windows), "min_pos": min(MIN_POS, cfg.seq_len // 4),
            "per_layer": layers, "overall": overall}


def suite_blimp(model, tok, pairs):
    from scripts import eval_blimp

    eval_blimp.DEVICE = next(model.parameters()).device
    good = eval_blimp.sentence_logprobs(model, tok, [p[0] for p in pairs])
    bad = eval_blimp.sentence_logprobs(model, tok, [p[1] for p in pairs])
    correct = (good > bad).astype(np.float64)
    uids, fields = np.array([p[2] for p in pairs]), np.array([p[4] for p in pairs])
    by_uid = {u: float(correct[uids == u].mean()) for u in np.unique(uids)}
    return {"overall_macro": float(np.mean(list(by_uid.values()))), "overall_micro": float(correct.mean()),
            "by_field": {f: float(correct[fields == f].mean()) for f in np.unique(fields)},
            "by_paradigm": by_uid, "n_pairs": int(len(correct))}


def suite_uncertainty(win_nll, groups):
    from scripts.eval_bootstrap_clustered import boot_clustered, boot_iid, boot_stratified, ci

    out = {}
    runs = list(win_nll)
    for ai, a in enumerate(runs):
        for b in runs[ai + 1:]:
            d = win_nll[b] - win_nll[a]
            entry = {"delta": float(d.mean()), "window_win_rate": float((d < 0).mean()),
                     "iid": ci(boot_iid(d)), "stratified": ci(boot_stratified(d, groups))}
            if len(np.unique(groups)) > 1:
                entry["clustered"] = ci(boot_clustered(d, groups))
            out[f"{b}_minus_{a}"] = entry
    return out


# ---------------------------------------------------------------- logging ---
def flat_metrics(result, prefix="test"):
    m = {}
    core = result.get("core", {})
    for k in ("nll", "ppl", "bits_per_token", "bits_per_byte", "top1", "ece", "mean_pred_entropy"):
        if core.get(k) is not None:
            m[f"{prefix}/{k}"] = core[k]
    for name, r in core.get("per_source", {}).items():
        m[f"{prefix}/nll_src/{name}"] = r["nll"]
    if "nll_by_position" in result.get("position", {}):
        pos = result["position"]["nll_by_position"]
        m[f"{prefix}/longctx_nll_first_bucket"] = pos[0]
        m[f"{prefix}/longctx_nll_last_bucket"] = pos[-1]
    if "overall_macro" in result.get("blimp", {}):
        m[f"{prefix}/blimp_macro"] = result["blimp"]["overall_macro"]
    for k, v in result.get("attention", {}).get("overall", {}).items():
        m[f"{prefix}/attn_{k}"] = v
    return m


def log_mlflow(run_dir, metrics, step, json_path):
    state = os.path.join(run_dir, "mlflow_run.json")
    if not os.path.exists(state):
        print(f"  mlflow: no {state}; skipped")
        return
    import mlflow

    with open(state) as f:
        run_id = json.load(f)["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics, step=step or 0)
        mlflow.log_artifact(json_path, artifact_path="eval")


def pull_from_mlflow(run_id, cache):
    """Download the best/snapshot weights a training run logged (--mlflow_log_checkpoints)."""
    import mlflow

    run = mlflow.get_run(run_id)
    name = run.data.tags.get("run_name", run_id)
    dest = os.path.join(cache, name)
    os.makedirs(dest, exist_ok=True)
    for sub in ("best", "snapshot"):
        try:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"checkpoints/{sub}", dst_path=dest)
        except Exception as e:  # nothing logged under this kind
            print(f"  mlflow: no checkpoints/{sub} ({type(e).__name__})")
    return sorted(glob.glob(os.path.join(dest, "checkpoints", "*", "*.pt")))


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run dirs (checkpoints/<run>) or .pt files")
    ap.add_argument("--from_mlflow", nargs="*", default=[], help="MLflow training run ids to pull weights from")
    ap.add_argument("--kind", choices=["best", "snapshot", "last"], default="best")
    ap.add_argument("--data_dir", default="data_l")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--suite", nargs="+", default=["core", "position", "attention", "uncertainty"],
                    choices=["core", "position", "attention", "blimp", "uncertainty"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                    help="parameter dtype; default bf16 on GPU, fp32 on CPU")
    ap.add_argument("--batch", type=int, default=1, help="windows per forward; 1 x 4096 fits 12 GB")
    ap.add_argument("--windows_per_source", type=int, default=32)
    ap.add_argument("--long_windows_per_source", type=int, default=16)
    ap.add_argument("--long_sources", nargs="*", default=["RedPajamaBook", "RedPajamaArXiv", "RedPajamaGithub"],
                    help="sources for the long-context suite; empty = all")
    ap.add_argument("--attention_windows", type=int, default=4)
    ap.add_argument("--trunc_windows", type=int, default=32)
    ap.add_argument("--gen_tokens", type=int, default=48)
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--mlflow", action="store_true", help="log metrics into each training run's MLflow run")
    ap.add_argument("--wandb", action="store_true", help="log one W&B eval run (job_type=eval)")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT") or "diff-moe")
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype or ("bf16" if args.device.startswith("cuda") else "fp32")]

    targets = [(p, os.path.dirname(os.path.dirname(p)) if p.endswith(".pt") else p)
               for r in args.runs for p in find_checkpoints(r, args.kind)]
    for run_id in args.from_mlflow:
        targets += [(p, None) for p in pull_from_mlflow(run_id, os.path.join(args.out_dir, "_mlflow_cache"))]
    if not targets:
        raise SystemExit("no checkpoints found; pass --runs checkpoints/<run> (with best/ or snapshots/) or .pt files")

    data = load_tokens(os.path.join(args.data_dir, f"{args.split}.bin"))
    spans = source_spans(args.data_dir, args.split)
    tok = resolve_data_tokenizer(args.data_dir)
    bpt = bytes_per_token_estimate(args.data_dir, os.path.join(args.data_dir, f"{args.split}.bin"))
    freq = None
    train_bin = os.path.join(args.data_dir, "train.bin")
    if "core" in args.suite and os.path.exists(train_bin):
        sample = np.asarray(load_tokens(train_bin)[:20_000_000])
        freq = np.bincount(sample, minlength=max(int(sample.max()), int(np.asarray(data).max())) + 1)
    blimp_pairs = None
    if "blimp" in args.suite:
        from scripts.eval_blimp import load_blimp

        blimp_pairs = load_blimp()

    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(project=args.wandb_project, job_type="eval",
                        config={"data_dir": args.data_dir, "split": args.split, "suite": args.suite,
                                "kind": args.kind, "device": args.device, "dtype": str(dtype)})

    os.makedirs(args.out_dir, exist_ok=True)
    comparison, win_nll, groups = {}, {}, None
    for path, run_dir in targets:
        t0 = time.time()
        model, cfg, info = load_model(path, args.device, dtype)
        name = os.path.basename(run_dir) if run_dir else os.path.basename(os.path.dirname(os.path.dirname(path)))
        label = f"{name}@{info.get('step')}"
        print("=" * 72)
        print(f"{label}  <- {path}\n  {cfg.attention}/{cfg.ffn} norm={cfg.norm} residual={cfg.residual} "
              f"xsa={cfg.xsa} d{cfg.dim} L{cfg.n_layers} ctx{cfg.seq_len} | {dtype} on {args.device}")
        result = {"run": name, "checkpoint": path, "checkpoint_info": info, "model": asdict(cfg),
                  "params": count_params(model), "eval": {"split": args.split, "dtype": str(dtype)}}

        windows = core_windows(len(data), spans, cfg.seq_len, args.windows_per_source)
        if "core" in args.suite:
            result["core"], win_nll[label] = suite_core(model, cfg, data, windows, args, bpt, freq, tok)
            groups = np.array([n for _, n in windows])
            c = result["core"]
            print(f"  [core] NLL {c['nll']:.4f} ppl {c['ppl']:.2f} bpb {c['bits_per_byte']} top1 {c['top1']:.4f} "
                  f"ECE {c['ece']:.4f}")
            for src_name, r in c["per_source"].items():
                print(f"         {src_name:26s} NLL {r['nll']:.4f}")
        long_windows = None
        if "position" in args.suite or "attention" in args.suite:
            long_windows = long_doc_windows(data, spans, cfg.seq_len, cfg.eos_id if cfg.doc_mask else -1,
                                            args.long_windows_per_source, args.long_sources)
        if "position" in args.suite:
            result["position"] = suite_position(model, cfg, data, long_windows, args)
            pos = result["position"].get("nll_by_position")
            if pos:
                print(f"  [position] {result['position']['n_windows']} single-doc windows: "
                      f"NLL {pos[0]:.3f} (first {result['position']['bucket_tokens']} tok) -> {pos[-1]:.3f} (last)")
        if "attention" in args.suite and long_windows:
            result["attention"] = suite_attention(model, cfg, data, long_windows, args)
            o = result["attention"]["overall"]
            print(f"  [attention] entropy {o['entropy']:.3f} support {math.exp(o['entropy']):.1f} "
                  f"top8 {o['top8_mass']:.3f} dist {o['attn_distance']:.1f} neg {o['negative_mass_frac']:.4f}")
        if blimp_pairs is not None and tok is not None:
            result["blimp"] = suite_blimp(model, tok, blimp_pairs)
            print(f"  [blimp] macro {result['blimp']['overall_macro'] * 100:.2f}%")

        out_path = os.path.join(args.out_dir, name, f"{args.kind}_step{info.get('step')}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        metrics = flat_metrics(result)
        if args.mlflow and run_dir:
            log_mlflow(run_dir, metrics, info.get("step"), out_path)
        if wb:
            wb.log({f"{name}/{k}": v for k, v in metrics.items()} | {f"{name}/step": info.get("step") or 0})
        comparison[label] = {"checkpoint": path, **info, **metrics}
        print(f"  wrote {out_path} ({time.time() - t0:.0f}s)")

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "uncertainty" in args.suite and len(win_nll) > 1:
        comparison["_paired"] = suite_uncertainty(win_nll, groups)
        print("\npaired deltas (negative = second better), 95% CI iid / clustered by source:")
        for pair, r in comparison["_paired"].items():
            print(f"  {pair:44s} {r['delta']:+.4f}  {r['iid']}  {r.get('clustered')}")
    if win_nll:
        np.savez_compressed(os.path.join(args.out_dir, "per_window_nll.npz"),
                            groups=groups, **{k.replace("@", "_step"): v for k, v in win_nll.items()})
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    if wb:
        wb.summary.update({"comparison": comparison})
        wb.finish()
    print("\nwrote", os.path.join(args.out_dir, "comparison.json"))


if __name__ == "__main__":
    main()
