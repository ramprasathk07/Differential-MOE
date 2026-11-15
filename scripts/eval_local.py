"""Evaluate a local checkpoint on CPU, without needing its training config.

The .pt stores no config, and this checkpoint came from a runtime-shrunk Kaggle
run, so the repo YAMLs won't match its geometry. We instead reconstruct the exact
ModelConfig from the weight *shapes*, rebuild the model, load the weights, and run
a capped test-set pass (full test set is too slow on CPU) reusing src.eval.evaluate
-- which already reports NLL / ppl / bits-per-byte / top-1 plus per-layer lambda
and expert-routing entropy. Writes final_test_eval.json + a minimal report.json
into the run dir so docs/blog/make_figures_part2.py can render the figures.

    python scripts/eval_local.py
"""

import json
import os
import sys
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import Transformer, count_params          # noqa: E402
from src.model.config import ModelConfig                 # noqa: E402
from src.data import eval_batches, load_tokens           # noqa: E402
from src.eval import evaluate, bytes_per_token_estimate  # noqa: E402

CKPT = os.path.join(ROOT, "model_checkpoints/checkpoints/s_diffmoe/best/step2250_nll3.6133.pt")
RUN_DIR = os.path.join(ROOT, "model_checkpoints/checkpoints/s_diffmoe")
DATA_DIR = os.path.join(ROOT, "data_s")
SEQ_LEN = 512        # kept fixed throughout the project; not stored in the ckpt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16             # fp32 logits ~3.3GB at 16x512x100k -- fine on a 12GB card
N_EVAL_WINDOWS = 3200  # windows sampled EVENLY across the full test split -- representative
#                        of all 6 domains yet small (~1.6M tokens) so it runs in ~1-2 min
GEN_MAX_NEW = 48


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
        n_heads = dim // 64                        # standard: head_dim not in shapes; 64 is the project default

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
    cfg = ModelConfig(
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
    return cfg


def strided_batches(data, batch, seq, n_windows, device):
    """Windows sampled EVENLY across the whole stream -- unlike src.data.eval_batches
    (contiguous from the start), so a small sample still covers every domain in the
    test split. Representative and fast."""
    total = (len(data) - 1) // seq
    n = min(n_windows, total)
    offsets = [int(k * total / n) * seq for k in range(n)]
    for b in range(0, n, batch):
        chunk = offsets[b:b + batch]
        x = torch.stack([torch.from_numpy(data[i:i + seq].astype("int64")) for i in chunk]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + seq + 1].astype("int64")) for i in chunk]).to(device)
        yield x, y


def main():
    torch.manual_seed(0)
    print(f"loading {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", mmap=True)
    sd = ckpt["model"]
    step = ckpt.get("step")
    best_val = ckpt.get("best_val")
    print("checkpoint step:", step)
    print("checkpoint best_val:", best_val)

    cfg = infer_config(sd, ckpt.get("model_config"))
    print("inferred config:", cfg)

    model = Transformer(cfg).to(DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("rope")]  # rope buffer is non-persistent
    if missing or unexpected:
        print("WARN load_state_dict -- missing:", missing, "unexpected:", unexpected)
    del ckpt, sd
    model.eval()

    counts = count_params(model)
    print("params:", {k: f"{v/1e6:.1f}M" for k, v in counts.items()})

    test = load_tokens(os.path.join(DATA_DIR, "test.bin"))
    try:
        bpt = bytes_per_token_estimate(DATA_DIR, os.path.join(DATA_DIR, "test.bin"))
    except Exception as e:
        print("bytes_per_token_estimate failed (bpb will be null):", e)
        bpt = None

    n_windows = min(N_EVAL_WINDOWS, (len(test) - 1) // SEQ_LEN)
    n_batches = (n_windows + BATCH - 1) // BATCH
    batches = strided_batches(test, BATCH, SEQ_LEN, n_windows, DEVICE)
    print(f"evaluating {n_windows} windows spread across the whole {len(test)/1e6:.1f}M-token "
          f"test split ({n_windows*SEQ_LEN/1e6:.2f}M tokens, {n_batches} batches) on {DEVICE} ...")
    result = evaluate(model, batches, bpt)
    print(f"test NLL {result.nll:.4f}  ppl {result.perplexity:.2f}  "
          f"bpb {result.bits_per_byte}  top1 {result.top1_acc:.4f}")

    # sample generations -- optional, only if a tokenizer resolves offline
    generations = {}
    try:
        from src.data.tokenizer import resolve_data_tokenizer
        tok = resolve_data_tokenizer(DATA_DIR)
        if tok is not None:
            prompts = ["Once upon a time", "The little boy said",
                       "The city is known for", "She looked at the"]
            for p in prompts:
                ids = torch.tensor([tok.encode(p)], dtype=torch.long).to(DEVICE)
                out = model.generate(ids, max_new_tokens=GEN_MAX_NEW, temperature=0.8, top_p=0.9)
                generations[p] = tok.decode(out[0].tolist())
                print(f"  [gen] {p!r} -> {generations[p]!r}")
    except Exception as e:
        print("generations skipped:", e)

    n_batches_tokens = n_windows * SEQ_LEN
    eval_report = {
        "run_name": "s_diffmoe",
        "checkpoint": os.path.relpath(CKPT, ROOT),
        "checkpoint_step": step,
        "eval_note": f"eval over {n_batches_tokens/1e6:.2f}M test tokens on {DEVICE} "
                     f"({n_batches} batches of {BATCH}x{SEQ_LEN}); test split is {len(test)} tokens",
        "best_val_at_train": best_val,
        "test_nll": result.nll,
        "test_ppl": result.perplexity,
        "test_bits_per_token": result.bits_per_token,
        "test_bits_per_byte": result.bits_per_byte,
        "test_top1_acc": result.top1_acc,
        "expert_entropy": result.expert_entropy,
        "expert_imbalance": result.expert_imbalance,
        "router_bias": result.router_bias,
        "lambda_means": {i: sum(v) / len(v) for i, v in result.lambda_values.items()},
        "lambda_per_head": result.lambda_values,
        "sample_generations": generations,
    }
    with open(os.path.join(RUN_DIR, "final_test_eval.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    # minimal report.json so make_figures_part2.py has model geometry + best_val
    report = {
        "run_name": "s_diffmoe",
        "model": asdict(cfg),
        "params": counts,
        "best_val": best_val,
        "training": {"checkpoint_step": step},
    }
    with open(os.path.join(RUN_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nwrote", os.path.join(RUN_DIR, "final_test_eval.json"))
    print("wrote", os.path.join(RUN_DIR, "report.json"))


if __name__ == "__main__":
    main()
