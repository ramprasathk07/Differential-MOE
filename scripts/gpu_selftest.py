"""GPU self-test: one forward + backward per architecture feature at L-tier shapes.

Each case runs in its own process (a CUDA fault poisons the whole context) with
CUDA_LAUNCH_BLOCKING=1, so a failure names the op that actually faulted instead of
whichever later call noticed it. Run it on the training box after any torch, CUDA,
Triton or driver change, and before a multi-day run:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python scripts/gpu_selftest.py
    python scripts/gpu_selftest.py --cases moe moe+ckpt          # a subset
"""

import argparse
import dataclasses
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Everything off, then one feature at a time, then the real L cell.
PLAIN = dict(attention="standard", ffn="dense", norm="rmsnorm", residual="standard",
             xsa=False, doc_mask=False, flex_attention=False)
CASES = {
    "plain": dict(PLAIN),
    "diff_attention": dict(PLAIN, attention="differential"),
    "moe": dict(PLAIN, ffn="moe"),
    "moe+ckpt": dict(PLAIN, ffn="moe"),
    "derf": dict(PLAIN, norm="derf"),
    "attnres": dict(PLAIN, residual="attnres"),
    "xsa": dict(PLAIN, xsa=True),
    "doc_mask_sdpa": dict(PLAIN, doc_mask=True),
    "diff+doc_mask_sdpa": dict(PLAIN, attention="differential", doc_mask=True),
    "doc_mask_flex": dict(PLAIN, doc_mask=True, flex_attention=True),
    "diff+doc_mask_flex": dict(PLAIN, attention="differential", doc_mask=True, flex_attention=True),
    "l_diffmoe": {},           # configs/l_diffmoe.yaml as is
    "l_diffmoe+ckpt": {},
    "l_diffmoe+ckpt+fsdp": {},
}


def run_case(name, device, seq_len):
    import torch
    import torch.nn.functional as F

    from src.model import Config, Transformer

    cfg = Config.from_yaml(os.path.join(ROOT, "configs", "l_diffmoe.yaml")).model
    cfg = dataclasses.replace(cfg, n_layers=2, n_dense_layers=1, seq_len=seq_len or cfg.seq_len, **CASES[name])
    fsdp = "fsdp" in name
    if fsdp:
        import torch.distributed as dist
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        dist.init_process_group("nccl" if device == "cuda" else "gloo", rank=0, world_size=1)
    torch.manual_seed(0)
    with torch.device(device):
        model = Transformer(cfg)
    model.gradient_checkpointing = "ckpt" in name
    if fsdp:
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in model.blocks:
            fully_shard(block, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
    x = torch.randint(1, cfg.vocab_size, (2, cfg.seq_len), device=device)
    x[:, ::997] = max(cfg.eos_id, 0)  # a few document boundaries
    with torch.autocast(device, dtype=torch.bfloat16, enabled=not fsdp):
        logits, aux, z = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), x.view(-1)) + 0.01 * aux + 0.001 * z
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"peak_mem_gb={torch.cuda.max_memory_allocated() / 2**30:.1f}")
    assert torch.isfinite(loss), f"loss {loss.item()}"


def summarize(stderr):
    """The error line plus the deepest frame inside this repo."""
    lines = [l.strip() for l in stderr.splitlines()]
    err = next((l for l in reversed(lines) if "Error" in l and not l.startswith("File")), "no error line")
    ours = [l for l in lines if l.startswith("File") and os.sep + "src" + os.sep in l.replace("/", os.sep)]
    return err[:140], (ours[-1].split(", in ")[0].split(os.sep + "src" + os.sep)[-1] if ours else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    ap.add_argument("--case", help=argparse.SUPPRESS)  # internal: run one case in this process
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_len", type=int, default=0, help="0 = the config's 4096")
    args = ap.parse_args()
    if args.case:
        run_case(args.case, args.device, args.seq_len)
        return

    import torch

    info = [torch.__version__, f"cuda {torch.version.cuda}"]
    if args.device == "cuda" and torch.cuda.is_available():
        info.append(torch.cuda.get_device_name(0))
    print("selftest:", " | ".join(info))
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", PYTHONPATH=ROOT)
    failed = 0
    for name in args.cases:
        t0 = time.time()
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", name,
                               "--device", args.device, "--seq_len", str(args.seq_len)],
                              env=env, capture_output=True, text=True)
        triton = "triton" if "TRITON_DEFAULT_FP_FUSION" in proc.stderr else ""
        mem = next((l.split("=")[1] + " GB" for l in proc.stdout.splitlines() if l.startswith("peak_mem_gb=")), "")
        if proc.returncode == 0:
            print(f"  OK    {name:22s} {time.time() - t0:5.0f}s {mem:>8s} {triton}")
        else:
            failed += 1
            err, where = summarize(proc.stderr)
            print(f"  FAIL  {name:22s} {time.time() - t0:5.0f}s {triton}\n        {err}\n        at src/{where}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
