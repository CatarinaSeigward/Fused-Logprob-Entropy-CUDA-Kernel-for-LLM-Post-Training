"""K1 backward microbenchmark — vs PyTorch autograd through eager reference.

Forward perf already shown by bench_micro.py. This file only times backward,
to confirm:
  (1) Our streaming backward kernel is competitive with eager autograd.
  (2) Memory profile stays flat (no [B,S,V] intermediate alloc).

Sample shapes pulled from the realistic GRPO regime.
"""
import time
import torch
import torch.nn.functional as F

from kernel_opt import fused_logprob_entropy

PEAK_DRAM_GBPS = 256.0


def _eager_reference(logits, targets):
    """Pure-PyTorch forward; differentiable."""
    log_probs = F.log_softmax(logits, dim=-1)
    logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p = log_probs.exp()
    ent = -(p * log_probs).sum(-1)
    lse = torch.logsumexp(logits, dim=-1)
    return logp, ent, lse


def _bench_backward(name, fwd_fn, base, targets, g_logp, g_ent, g_lse,
                    n_warmup=5, n_iter=20):
    # Warmup
    for _ in range(n_warmup):
        x = base.clone().requires_grad_(True)
        logp, ent, lse = fwd_fn(x, targets)
        loss = (logp * g_logp + ent * g_ent + lse * g_lse).sum()
        loss.backward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        x = base.clone().requires_grad_(True)
        logp, ent, lse = fwd_fn(x, targets)
        loss = (logp * g_logp + ent * g_ent + lse * g_lse).sum()
        starts[i].record()
        loss.backward()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median = times[len(times) // 2]

    peak_mb = (torch.cuda.max_memory_allocated() - pre) / (1024**2)

    bytes_traffic = base.numel() * base.element_size() * 2  # read + write d_logits
    bw = (bytes_traffic / 1e9) / (median / 1e3)

    print(f"{name:>20}  bwd_ms={median:>7.3f}  bw_GBps={bw:>7.1f}  "
          f"bw_%peak={100*bw/PEAK_DRAM_GBPS:>5.1f}%  bwd_alloc_MB={peak_mb:>6.1f}")
    return median


def main():
    shapes = [(1, 256, 32000), (1, 1024, 128256), (1, 1024, 152064), (1, 4096, 32000)]
    for B, S, V in shapes:
        print(f"\n=== shape (B,S,V)=({B},{S},{V}) bf16 ===")
        torch.manual_seed(0)
        base = torch.randn(B, S, V, device="cuda", dtype=torch.bfloat16)
        targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)
        g_logp = torch.randn(B, S, device="cuda")
        g_ent = torch.randn(B, S, device="cuda")
        g_lse = torch.randn(B, S, device="cuda")

        t_ours = _bench_backward("ours_v1", fused_logprob_entropy, base, targets,
                                 g_logp, g_ent, g_lse)
        t_eager = _bench_backward("pytorch_autograd", _eager_reference, base, targets,
                                  g_logp, g_ent, g_lse)
        print(f"  -> backward speedup: {t_eager/t_ours:.2f}x")


if __name__ == "__main__":
    main()
