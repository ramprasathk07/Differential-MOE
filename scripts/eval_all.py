"""Evaluate every finished tier-S checkpoint on the held-out BabyLM test split
under one identical protocol, including the Stable LatentMoE attention pair.

The .pt files carry no config and came from a runtime-shrunk Kaggle run, so the
repo YAMLs do not match their geometry -- the ModelConfig is reconstructed from
the weight shapes instead (same approach as eval_local.py, which this replaces
for multi-run use).

Two passes per model, both on exactly the same token windows for every run:

  A. global   -- windows spread evenly across the whole test split; headline
                 NLL / ppl / bits-per-byte / top-1, plus routing and lambda stats.
  B. per-domain -- windows spread evenly *within* each of BabyLM's six domain
                 spans (from domain_offsets.json). This is the number that says
                 whether sparsity is paying off where the corpus is heterogeneous.

Writes <run_dir>/final_test_eval.json per run and a combined
docs/blog/runs_export/eval_comparison.json.

    python scripts/eval_all.py
"""

import gc
import json
import os
import sys
import time
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import Transformer, count_params          # noqa: E402
from src.model.config import ModelConfig                 # noqa: E402
from src.data import load_tokens                         # noqa: E402
from src.eval import evaluate, bytes_per_token_estimate  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data_s")
CKPT_ROOT = os.path.join(ROOT, "model_checkpoints/checkpoints")
OUT_DIR = os.path.join(ROOT, "docs/blog/runs_export")
SEQ_LEN = 512          # fixed throughout the project; not stored in the ckpt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16
N_GLOBAL_WINDOWS = 3200    # ~1.6M tokens, spread across the full 16.1M-token split
N_DOMAIN_WINDOWS = 384     # per domain, ~0.2M tokens each
GEN_MAX_NEW = 48
PROMPTS = ["Once upon a time", "The little boy said",
           "The city is known for", "She looked at the"]

# The 2x2. Any run whose directory is absent is skipped, so this file needs no
# edit as the remaining cells finish.
RUNS = [
    "s_dense",
    "s_diff",
    "s_moe",
    "s_diffmoe",
    "s_stable_latentmoe",
    "s_diff_stable_latentmoe",
]


def best_checkpoint(run_dir):
    """The checkpoint training itself picked as best by dev NLL. Reading
    best_index.json rather than hardcoding filenames means a rerun that lands on
    a different step is picked up automatically."""
    index = os.path.join(run_dir, "best_index.json")
    if not os.path.isfile(index):
        return None
    with open(index) as f:
        entries = json.load(f)
    if not entries:
        return None
    # the path recorded is the Kaggle-side absolute path; only the name survives
    local = os.path.join(run_dir, "best", os.path.basename(entries[0]["path"]))
    return local if os.path.isfile(local) else None


def infer_config(sd, stored_config=None) -> ModelConfig:
    """Rebuild the ModelConfig from state-dict tensor shapes alone."""
    if stored_config:
        return ModelConfig(**stored_config)

    vocab, dim = sd["embed.weight"].shape
    layer_ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    n_layers = max(layer_ids) + 1

    differential = any(k.endswith("attn.lambda_q1") for k in sd)
    if differential:
        lq = sd["blocks.0.attn.lambda_q1"]        # (n_heads//2, head_dim)
        head_dim = lq.shape[1]
        n_heads = dim // head_dim
        assert n_heads == 2 * lq.shape[0], "head inference mismatch"
    else:
        n_heads = dim // 64                        # standard: head_dim absent from shapes

    moe_blocks = [i for i in layer_ids if f"blocks.{i}.ffn.gate.weight" in sd]
    dense_blocks = [i for i in layer_ids if f"blocks.{i}.ffn.gate.weight" not in sd]
    stable_latent = any(
        k.endswith("ffn.routed_expert_down_proj.weight") for k in sd
    )
    ffn = "stable_latent_moe" if stable_latent else "moe" if moe_blocks else "dense"
    n_dense_layers = min(moe_blocks) if moe_blocks else n_layers

    if moe_blocks:
        b = moe_blocks[0]
        n_experts = len({int(k.split(".")[4]) for k in sd
                         if k.startswith(f"blocks.{b}.ffn.experts.")})
        expert_inter = sd[f"blocks.{b}.ffn.experts.0.w1.weight"].shape[0]
    else:
        n_experts, expert_inter = 8, 512
    inter_dim = (sd[f"blocks.{dense_blocks[0]}.ffn.w1.weight"].shape[0]
                 if dense_blocks else 4 * dim)

    latent_dim = (
        sd[f"blocks.{moe_blocks[0]}.ffn.routed_expert_down_proj.weight"].shape[0]
        if stable_latent
        else None
    )
    return ModelConfig(
        vocab_size=vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
        seq_len=SEQ_LEN, tie_embeddings=True,
        attention="differential" if differential else "standard",
        ffn=ffn, inter_dim=inter_dim,
        n_dense_layers=n_dense_layers, n_experts=n_experts,
        top_k=4 if stable_latent else 2,
        expert_inter_dim=expert_inter,
        n_shared_experts=2 if stable_latent else 0,
        shared_inter_dim=latent_dim if stable_latent else 512,
        latent_dim=latent_dim,
        aux_loss_coef=0.0 if stable_latent else 0.01,
        router_z_coef=0.0 if stable_latent else 0.001,
    )


def window_offsets(lo, hi, n_windows, seq=SEQ_LEN):
    """Start indices spread evenly over [lo, hi), each leaving room for seq+1
    tokens. Deterministic, so every model sees byte-identical windows."""
    total = max(0, (hi - lo - 1) // seq)
    n = min(n_windows, total)
    return [lo + int(k * total / n) * seq for k in range(n)] if n else []


def batches_from(data, offsets, batch=BATCH, seq=SEQ_LEN, device=DEVICE):
    for b in range(0, len(offsets), batch):
        chunk = offsets[b:b + batch]
        x = torch.stack([torch.from_numpy(data[i:i + seq].astype("int64")) for i in chunk]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + seq + 1].astype("int64")) for i in chunk]).to(device)
        yield x, y


def main():
    test = load_tokens(os.path.join(DATA_DIR, "test.bin"))
    with open(os.path.join(DATA_DIR, "domain_offsets.json")) as f:
        domains = json.load(f)["test"]
    try:
        bpt = bytes_per_token_estimate(DATA_DIR, os.path.join(DATA_DIR, "test.bin"))
    except Exception as e:
        print("bytes_per_token_estimate failed (bpb will be null):", e)
        bpt = None

    global_offsets = window_offsets(0, len(test), N_GLOBAL_WINDOWS)
    domain_offsets = {d: window_offsets(lo, hi, N_DOMAIN_WINDOWS)
                      for d, (lo, hi) in domains.items()}
    print(f"test split {len(test)/1e6:.1f}M tokens | global {len(global_offsets)} windows "
          f"({len(global_offsets)*SEQ_LEN/1e6:.2f}M tok) | domains: "
          + ", ".join(f"{d}={len(o)}" for d, o in domain_offsets.items()))
    print(f"device {DEVICE} | bytes/token {bpt}\n")

    try:
        from src.data.tokenizer import resolve_data_tokenizer
        tok = resolve_data_tokenizer(DATA_DIR)
    except Exception as e:
        print("tokenizer unavailable, skipping generations:", e)
        tok = None

    comparison = {}
    for run_name in RUNS:
        run_dir = os.path.join(CKPT_ROOT, run_name)
        ckpt_path = best_checkpoint(run_dir)
        if ckpt_path is None:
            print(f"skip {run_name}: no best checkpoint under {run_dir}")
            continue
        print("=" * 70)
        print(f"{run_name}  <-  {os.path.relpath(ckpt_path, run_dir)}")
        t0 = time.time()

        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        sd = ckpt["model"]
        step, best_val = ckpt.get("step"), ckpt.get("best_val")
        cfg = infer_config(sd, ckpt.get("model_config"))
        print(f"  step {step} | best_val {best_val} | {cfg.attention}/{cfg.ffn} "
              f"dim{cfg.dim} L{cfg.n_layers} H{cfg.n_heads} "
              f"experts={cfg.n_experts if cfg.ffn != 'dense' else '-'}")

        model = Transformer(cfg).to(DEVICE)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith("rope")]  # non-persistent buffer
        if missing or unexpected:
            print("  WARN load_state_dict -- missing:", missing, "unexpected:", unexpected)
        del ckpt, sd
        gc.collect()
        model.eval()
        counts = count_params(model)
        print("  params:", {k: f"{v/1e6:.1f}M" for k, v in counts.items()})

        # ---- pass A: global -------------------------------------------------
        res = evaluate(model, batches_from(test, global_offsets), bpt)
        print(f"  [global]  NLL {res.nll:.4f}  ppl {res.perplexity:.2f}  "
              f"bpb {res.bits_per_byte:.4f}  top1 {res.top1_acc:.4f}  ({time.time()-t0:.0f}s)")

        # ---- pass B: per domain ---------------------------------------------
        per_domain = {}
        for d, offs in domain_offsets.items():
            if not offs:
                continue
            dr = evaluate(model, batches_from(test, offs), bpt)
            per_domain[d] = {"nll": dr.nll, "ppl": dr.perplexity,
                             "top1": dr.top1_acc, "n_windows": len(offs),
                             "expert_entropy": dr.expert_entropy,
                             "expert_imbalance": dr.expert_imbalance,
                             "expert_counts": dr.expert_counts}
            print(f"    {d:16s} NLL {dr.nll:.4f}  ppl {dr.perplexity:8.2f}  top1 {dr.top1_acc:.4f}")

        # ---- generations -----------------------------------------------------
        generations = {}
        if tok is not None:
            torch.manual_seed(0)   # same seed for every run -> comparable samples
            for p in PROMPTS:
                ids = torch.tensor([tok.encode(p)], dtype=torch.long).to(DEVICE)
                out = model.generate(ids, max_new_tokens=GEN_MAX_NEW, temperature=0.8, top_p=0.9)
                generations[p] = tok.decode(out[0].tolist())

        report = {
            "run_name": run_name,
            "checkpoint": os.path.relpath(ckpt_path, ROOT),
            "checkpoint_step": step,
            "best_val_at_train": best_val,
            "eval_note": f"identical protocol across runs: {len(global_offsets)} windows "
                         f"({len(global_offsets)*SEQ_LEN/1e6:.2f}M tokens) spread evenly over the "
                         f"{len(test)}-token test split, batch {BATCH}x{SEQ_LEN}, fp32 on {DEVICE}",
            "model": asdict(cfg),
            "params": counts,
            "test_nll": res.nll,
            "test_ppl": res.perplexity,
            "test_bits_per_token": res.bits_per_token,
            "test_bits_per_byte": res.bits_per_byte,
            "test_top1_acc": res.top1_acc,
            "per_domain": per_domain,
            "expert_entropy": res.expert_entropy,
            "expert_imbalance": res.expert_imbalance,
            "expert_counts": res.expert_counts,
            "router_bias": res.router_bias,
            "lambda_means": {i: sum(v) / len(v) for i, v in res.lambda_values.items()},
            "lambda_per_head": res.lambda_values,
            "sample_generations": generations,
        }
        with open(os.path.join(run_dir, "final_test_eval.json"), "w") as f:
            json.dump(report, f, indent=2)
        comparison[run_name] = report
        print(f"  wrote {run_dir}/final_test_eval.json  ({time.time()-t0:.0f}s total)\n")

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eval_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print("wrote", os.path.join(OUT_DIR, "eval_comparison.json"))

    print("\n" + "=" * 70)
    print(f"{'run':12s} {'step':>5s} {'test NLL':>9s} {'ppl':>8s} {'bpb':>7s} {'top1':>7s}")
    for name, r in comparison.items():
        print(f"{name:12s} {r['checkpoint_step']:>5d} {r['test_nll']:>9.4f} "
              f"{r['test_ppl']:>8.2f} {r['test_bits_per_byte']:>7.4f} {r['test_top1_acc']:>7.4f}")


if __name__ == "__main__":
    main()
