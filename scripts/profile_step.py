"""Profile one training micro-batch (forward + backward with activation checkpointing)
on one GPU: tokens/s, how busy the GPU is, and the ops that cost the most time.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python scripts/profile_step.py \
        --config configs/l_diffmoe.yaml --batch 1 [--fsdp] [--override model.residual=standard]

GPU busy % far below 100 means the GPU waits on Python / host syncs (launch-bound):
fusing (compile) and removing syncs help. Near 100% means the kernels themselves are slow.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402

from src.model import Config, Transformer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/l_diffmoe.yaml")
    ap.add_argument("--override", action="append", default=[], help="e.g. model.n_layers=4")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--fsdp", action="store_true", help="wrap like the trainer (FSDP2, world size 1)")
    ap.add_argument("--compile", action="store_true", help="model.compile_blocks(), like train.compile=true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = Config.from_yaml(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config,
                           args.override).model
    dev = args.device
    with torch.device(dev):
        model = Transformer(cfg)
    model.gradient_checkpointing = True
    if args.compile:
        model.compile_blocks()  # same order as src/train.py: compile, then shard
    if args.fsdp:
        import torch.distributed as dist
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29581")
        dist.init_process_group("nccl" if dev == "cuda" else "gloo", rank=0, world_size=1)
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for block in model.blocks:
            fully_shard(block, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
    x = torch.randint(1, cfg.vocab_size, (args.batch, cfg.seq_len), device=dev)
    x[:, ::997] = max(cfg.eos_id, 0)  # document boundaries, like real packed data

    def micro():
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=not args.fsdp):
            logits, aux, z = model(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), x.view(-1)) + 0.01 * aux + 0.001 * z
        loss.backward()

    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    for _ in range(4 if args.compile else 2):  # warm-up: kernel compilation, allocator growth
        micro()
    sync()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        micro()
    sync()
    dt = (time.perf_counter() - t0) / args.iters
    tok_s = args.batch * cfg.seq_len / dt
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0

    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev == "cuda" else [])
    with torch.profiler.profile(activities=acts) as prof:
        t0 = time.perf_counter()
        micro()
        sync()
        wall = time.perf_counter() - t0
    events = prof.key_averages()
    sort = "self_device_time_total" if dev == "cuda" else "self_cpu_time_total"
    print(events.table(sort_by=sort, row_limit=25, max_name_column_width=60))
    # Summary last, so `| tail -3` keeps it.
    print(f"\n{cfg.attention}/{cfg.ffn} norm={cfg.norm} residual={cfg.residual} xsa={cfg.xsa} "
          f"flex={cfg.flex_attention} fsdp={args.fsdp} compile={args.compile} batch={args.batch}x{cfg.seq_len}")
    print(f"{tok_s:,.0f} tok/s  ({dt:.2f} s per micro-batch)  peak {peak:.1f} GB  "
          f"-> 10B tokens in {10e9 / tok_s / 86400:.1f} days on this GPU (optimizer step not included)")
    if dev == "cuda":
        # Kernel rows only: op rows repeat their kernels' time and would double-count.
        busy = sum(e.self_device_time_total for e in events if e.device_type == DeviceType.CUDA) / 1e6
        print(f"GPU busy {100 * busy / wall:.0f}% of the micro-batch wall time ({busy:.2f} of {wall:.2f} s)")


if __name__ == "__main__":
    main()
