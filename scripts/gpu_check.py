"""Quick hardware check of one GPU, pure PyTorch (no model code):
  1. matmul fp32/bf16 vs a CPU reference
  2. FlashAttention forward+backward vs the math backend, 10 trials
  3. memory pattern test over (almost) all free memory

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python scripts/gpu_check.py

Healthy A6000: matmul OK, flash 0/10 bad, memtest 0 bad words. Exit code 1 on any failure.
A crash (illegal memory access) poisons the CUDA context, so the memory test runs last.
"""

import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def matmul_check():
    torch.manual_seed(0)
    a, b = torch.randn(4096, 4096), torch.randn(4096, 4096)
    ref = a @ b
    ok = True
    for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 2e-2)):
        out = (a.cuda().to(dtype) @ b.cuda().to(dtype)).float().cpu()
        err = ((out - ref).norm() / ref.norm()).item()
        ok &= err < tol
        print(f"  matmul {str(dtype):15s} rel err {err:.1e}  {'OK' if err < tol else 'FAIL'}")
    return ok


def attention(backend, q, k, v, do):
    q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
    with sdpa_kernel(backend):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o.backward(do)
    return o.float(), q.grad.float()


def flash_check(trials=10, tol=5e-2):
    bad, worst = 0, 0.0
    for t in range(trials):
        torch.manual_seed(t)
        q, k, v, do = (torch.randn(2, 16, 4096, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        o_ref, g_ref = attention(SDPBackend.MATH, q, k, v, do)
        o, g = attention(SDPBackend.FLASH_ATTENTION, q, k, v, do)
        err = max(((o - o_ref).norm() / o_ref.norm()).item(), ((g - g_ref).norm() / g_ref.norm()).item())
        bad += not err < tol  # NaN counts as bad
        worst = max(worst, err) if err == err else float("nan")
    print(f"  flash attention fwd+bwd: {bad}/{trials} bad, worst rel err {worst:.1e}  "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


def memtest():
    n = 2**28  # 1 GiB of int32 per chunk
    base = torch.arange(n, dtype=torch.int32, device="cuda")
    chunks = []
    while torch.cuda.mem_get_info()[0] > 3 * 2**30:  # leave room for the check temporaries
        chunks.append(torch.empty(n, dtype=torch.int32, device="cuda"))
    bad = 0
    for flip in (False, True):  # every bit is tested as both 0 and 1
        for i, c in enumerate(chunks):
            torch.add(base, i, out=c)
            if flip:
                c.bitwise_not_()
        torch.cuda.synchronize()
        for i, c in enumerate(chunks):
            back = c.bitwise_not() if flip else c.clone()
            bad += int(back.sub_(i).ne(base).sum())
    print(f"  memtest {len(chunks)} GiB: {bad} bad words  {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  {p.total_memory / 2**30:.1f} GiB  pci_bus_id={getattr(p, 'pci_bus_id', '?')}  "
          f"torch {torch.__version__}")
    ok = True
    for name, check in (("matmul", matmul_check), ("flash", flash_check), ("memtest", memtest)):
        try:
            ok &= check()
        except Exception as e:  # illegal address etc.: the context is dead, stop here
            print(f"  {name}: CRASH {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            ok = False
            break
    print("RESULT:", "PASS" if ok else "FAIL  (admin: sudo dmesg | grep -i xid; nvidia-smi -r -i <id>)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
