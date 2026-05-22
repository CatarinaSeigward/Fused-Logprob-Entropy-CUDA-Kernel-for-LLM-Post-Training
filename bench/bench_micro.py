"""K1 microbenchmark — fused logprob+entropy vs eager baselines.

Every K1 perf change in Stage 2/3 is measured by THIS harness, against the
SAME shape grid, so apples-to-apples comparisons are forced from day 1.

Baselines:
  - `ours`             : kernel_opt.fused_logprob_entropy_naive (now);
                         later replaced by the optimized variant.
  - `trl_eager`        : trl.trainer.utils.selective_log_softmax(logits, ids)
                         + entropy_from_logits(logits). The actual GRPO path.
  - `pytorch_separate` : F.log_softmax + gather, then softmax * log_softmax
                         sum reduction. The "obvious" eager implementation.

Metrics:
  - latency (ms, median of N runs after warmup)
  - effective DRAM bandwidth (GB/s) — bytes_read / latency, where bytes_read =
    B*S*V * dtype_size  (one streaming pass is the lower bound; eager paths do
    multiple passes so their effective bandwidth will exceed peak)
  - peak alloc (MB) during the call

Output:
  CSV in bench/results/bench_micro_<timestamp>.csv (auditable).
  Pretty table to stdout for quick scanning.
"""
import argparse
import csv
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kernel_opt import fused_logprob_entropy, fused_logprob_entropy_naive
from trl.trainer.utils import selective_log_softmax, entropy_from_logits


# RTX 4060 Laptop peak DRAM bandwidth = 256 GB/s (8 GB GDDR6 @ 16 Gbps × 128-bit)
PEAK_DRAM_GBPS = 256.0


@dataclass
class BenchResult:
    name: str
    shape: tuple
    dtype: str
    latency_ms: float
    bandwidth_gbps: float
    bandwidth_pct: float
    peak_alloc_mb: float


def _bench_one(name: str, fn, logits: torch.Tensor, targets: torch.Tensor,
               n_warmup: int = 5, n_iter: int = 30) -> BenchResult:
    """Time `fn(logits, targets)`. Returns latency stats + bandwidth + alloc."""
    # Warmup
    for _ in range(n_warmup):
        out = fn(logits, targets)
        del out
    torch.cuda.synchronize()

    # Reset alloc tracking
    torch.cuda.reset_peak_memory_stats()
    pre_alloc = torch.cuda.memory_allocated()

    # Time with CUDA events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        out = fn(logits, targets)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]
    del out

    peak_alloc = (torch.cuda.max_memory_allocated() - pre_alloc) / (1024**2)

    # Bandwidth: single streaming pass over the [B, S, V] logits.
    bytes_read = logits.numel() * logits.element_size()
    bandwidth_gbps = (bytes_read / 1e9) / (median_ms / 1e3)

    return BenchResult(
        name=name,
        shape=tuple(logits.shape),
        dtype=str(logits.dtype).replace("torch.", ""),
        latency_ms=median_ms,
        bandwidth_gbps=bandwidth_gbps,
        bandwidth_pct=100.0 * bandwidth_gbps / PEAK_DRAM_GBPS,
        peak_alloc_mb=peak_alloc,
    )


def _ours_naive(logits, targets):
    return fused_logprob_entropy_naive(logits, targets)


def _ours_v1(logits, targets):
    return fused_logprob_entropy(logits, targets)


def _trl_eager(logits, targets):
    logp = selective_log_softmax(logits, targets)
    ent = entropy_from_logits(logits)
    return logp, ent


def _pytorch_separate(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p = log_probs.exp()
    ent = -(p * log_probs).sum(-1)
    return logp, ent


BASELINES = {
    "ours_v1": _ours_v1,
    "ours_naive": _ours_naive,
    "trl_eager": _trl_eager,
    "pytorch_separate": _pytorch_separate,
}


def memory_for_shape(B, S, V, dtype: torch.dtype) -> int:
    """Conservative VRAM footprint estimate (bytes) for the comparison."""
    elem = torch.tensor([], dtype=dtype).element_size()
    logits_bytes = B * S * V * elem
    # Eager paths can allocate up to ~4× logits in fp32 intermediate
    return logits_bytes * 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, nargs="+", default=[32000, 128256, 152064])
    ap.add_argument("--bs", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES.keys()),
                    choices=list(BASELINES.keys()))
    ap.add_argument("--vram-budget-gb", type=float, default=6.0,
                    help="skip shapes whose conservative estimate exceeds this")
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "results",
        f"bench_micro_{int(time.time())}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    print(f"{'shape':>22}  {'dtype':>5}  {'baseline':>18}  "
          f"{'lat_ms':>8}  {'bw_GBps':>8}  {'bw_%peak':>8}  {'alloc_MB':>9}")
    print("-" * 96)

    for V in args.vocab:
        for BS in args.bs:
            B, S = 1, BS
            est = memory_for_shape(B, S, V, dtype)
            if est > args.vram_budget_gb * 1024**3:
                print(f"  skip (B,S,V)=({B},{S},{V}) {args.dtype}: estimate "
                      f"{est/1024**3:.1f} GB > budget {args.vram_budget_gb:.1f} GB")
                continue

            torch.manual_seed(0)
            logits = torch.randn(B, S, V, device="cuda", dtype=dtype)
            targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

            for name in args.baselines:
                try:
                    r = _bench_one(name, BASELINES[name], logits, targets,
                                   n_iter=args.n_iter)
                    results.append(r)
                    print(f"{str((B,S,V)):>22}  {r.dtype:>5}  {name:>18}  "
                          f"{r.latency_ms:>8.3f}  {r.bandwidth_gbps:>8.1f}  "
                          f"{r.bandwidth_pct:>7.1f}%  {r.peak_alloc_mb:>9.1f}")
                except Exception as e:
                    print(f"  FAIL {name} on {(B,S,V)} {dtype}: {e}")

            del logits, targets
            torch.cuda.empty_cache()

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "B", "S", "V", "dtype",
                    "latency_ms", "bandwidth_gbps", "bandwidth_pct_peak", "peak_alloc_mb"])
        for r in results:
            B, S, V = r.shape
            w.writerow([r.name, B, S, V, r.dtype,
                        f"{r.latency_ms:.4f}", f"{r.bandwidth_gbps:.2f}",
                        f"{r.bandwidth_pct:.2f}", f"{r.peak_alloc_mb:.2f}"])
    print(f"\nwrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
