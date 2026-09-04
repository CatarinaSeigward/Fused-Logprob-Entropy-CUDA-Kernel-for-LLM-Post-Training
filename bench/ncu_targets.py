"""Nsight Compute profiling targets — one backend per process invocation.

Why one backend per process: ncu's kernel-name filter keeps reports clean, and
mixing backends in one process makes --launch-skip arithmetic fragile.

The profiled region is bracketed by torch.cuda.profiler.start()/stop(), so the
recommended invocation is:

    ncu --profile-from-start off --set full -o <out> \
        python bench/ncu_targets.py --backend ours_fwd --shape 1024x128k

With --profile-from-start off, ncu captures ONLY the launches between
start() and stop() — i.e. exactly one iteration, after warmup. No
--launch-skip arithmetic needed.

Backends
--------
ours_fwd  : K1 v1 forward (the production kernel)
ours_bwd  : K1 v1 forward + backward (times the backward kernel)
naive     : K1 naive one-thread-per-row reference (shows what block-per-row buys)
trl       : TRL's selective_log_softmax + entropy_from_logits eager path

Run without ncu and it still works — profiler.start/stop are no-ops when no
profiler is attached, so you can smoke-test the script standalone.
"""
from __future__ import annotations

import argparse
import sys

import torch


# (B*S, V) shapes spanning the realistic GRPO regime.
SHAPES = {
    "256x32k":   (256, 32000),      # small batch, Llama-2-ish vocab
    "1024x32k":  (1024, 32000),
    "4096x32k":  (4096, 32000),     # large batch — where K1 fwd peaked at 82.6%
    "256x128k":  (256, 128256),     # Llama-3 vocab
    "1024x128k": (1024, 128256),    # the canonical mid-size shape
    "1024x152k": (1024, 152064),    # Qwen2.5 vocab
}

# Kernel-name regexes to pass to `ncu -k regex:...` for each backend. TRL's
# eager path launches many aten kernels, so we deliberately do NOT filter it
# (capturing all of them is the point — it shows the launch-count blowup).
KERNEL_FILTERS = {
    "ours_fwd": "fused_logprob_entropy_v1_kernel",
    "ours_bwd": "fused_logprob_entropy_v1_backward_kernel",
    "naive":    "fused_logprob_entropy_naive_kernel",
    "trl":      None,
}


def build_inputs(bs: int, vocab: int, dtype: torch.dtype, requires_grad: bool):
    torch.manual_seed(1234)
    logits = torch.randn(1, bs, vocab, device="cuda", dtype=dtype)
    if requires_grad:
        logits = logits.detach().requires_grad_(True)
    targets = torch.randint(0, vocab, (1, bs), device="cuda", dtype=torch.int64)
    return logits, targets


def run_ours_fwd(logits, targets, profile: bool):
    from kernel_opt import fused_logprob_entropy_forward
    if profile:
        torch.cuda.profiler.start()
    out = fused_logprob_entropy_forward(logits, targets)
    torch.cuda.synchronize()
    if profile:
        torch.cuda.profiler.stop()
    return out


def run_naive(logits, targets, profile: bool):
    from kernel_opt import fused_logprob_entropy_naive
    if profile:
        torch.cuda.profiler.start()
    out = fused_logprob_entropy_naive(logits, targets)
    torch.cuda.synchronize()
    if profile:
        torch.cuda.profiler.stop()
    return out


def run_ours_bwd(logits, targets, grads, profile: bool):
    """Forward is outside the profiled region; only backward is captured."""
    from kernel_opt import fused_logprob_entropy
    g_logp, g_ent, g_lse = grads
    logp, ent, lse = fused_logprob_entropy(logits, targets)
    loss = (logp * g_logp + ent * g_ent + lse * g_lse).sum()
    torch.cuda.synchronize()

    if profile:
        torch.cuda.profiler.start()
    loss.backward()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.profiler.stop()
    return logits.grad


def run_trl(logits, targets, profile: bool):
    """TRL's production eager path: two separate passes over the logits."""
    from trl.trainer.utils import selective_log_softmax, entropy_from_logits
    if profile:
        torch.cuda.profiler.start()
    logps = selective_log_softmax(logits, targets)
    entropies = entropy_from_logits(logits)
    torch.cuda.synchronize()
    if profile:
        torch.cuda.profiler.stop()
    return logps, entropies


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True,
                    choices=["ours_fwd", "ours_bwd", "naive", "trl"])
    ap.add_argument("--shape", default="1024x128k", choices=sorted(SHAPES))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--warmup", type=int, default=3,
                    help="iterations before the profiled one (default 3)")
    ap.add_argument("--no-profile", action="store_true",
                    help="skip profiler.start/stop; use for standalone smoke-testing")
    ap.add_argument("--print-cmd", action="store_true",
                    help="print the recommended ncu command for this config and exit")
    args = ap.parse_args()

    bs, vocab = SHAPES[args.shape]
    dtype = getattr(torch, args.dtype)

    if args.print_cmd:
        kfilter = KERNEL_FILTERS[args.backend]
        k_arg = f' -k regex:"{kfilter}"' if kfilter else ""
        print(
            f'ncu --profile-from-start off --set full{k_arg} '
            f'-o bench/ncu/{args.backend}_{args.shape} --force-overwrite '
            f'python bench/ncu_targets.py --backend {args.backend} --shape {args.shape}'
        )
        return

    profile = not args.no_profile
    needs_grad = args.backend == "ours_bwd"
    logits, targets = build_inputs(bs, vocab, dtype, requires_grad=needs_grad)

    grads = None
    if needs_grad:
        grads = (
            torch.randn(1, bs, device="cuda") * 0.5,
            torch.randn(1, bs, device="cuda") * 0.5,
            torch.randn(1, bs, device="cuda") * 0.5,
        )

    bytes_read = logits.numel() * logits.element_size()
    print(f"[ncu_targets] backend={args.backend} shape={args.shape} "
          f"(B*S={bs}, V={vocab}) dtype={args.dtype}")
    print(f"[ncu_targets] logits tensor: {bytes_read / 1e6:.1f} MB")
    print(f"[ncu_targets] warmup={args.warmup} profile={'on' if profile else 'off'}")

    # ---- warmup (NOT profiled: profiler.start() comes after) ----
    for _ in range(args.warmup):
        if args.backend == "ours_fwd":
            run_ours_fwd(logits, targets, profile=False)
        elif args.backend == "naive":
            run_naive(logits, targets, profile=False)
        elif args.backend == "trl":
            run_trl(logits, targets, profile=False)
        elif args.backend == "ours_bwd":
            if logits.grad is not None:
                logits.grad = None
            run_ours_bwd(logits, targets, grads, profile=False)
    torch.cuda.synchronize()

    # ---- the one profiled iteration ----
    if args.backend == "ours_fwd":
        out = run_ours_fwd(logits, targets, profile)
        shape_str = tuple(out[0].shape)
    elif args.backend == "naive":
        out = run_naive(logits, targets, profile)
        shape_str = tuple(out[0].shape)
    elif args.backend == "trl":
        out = run_trl(logits, targets, profile)
        shape_str = tuple(out[0].shape)
    else:  # ours_bwd
        if logits.grad is not None:
            logits.grad = None
        out = run_ours_bwd(logits, targets, grads, profile)
        shape_str = tuple(out.shape)

    print(f"[ncu_targets] done. output shape {shape_str}")


if __name__ == "__main__":
    sys.exit(main())
