"""Python wrappers around the CUDA kernels.

Public entry points:
  - `fused_logprob_entropy(logits, targets)` — autograd-aware K1.
    Use this from training code.
  - `fused_logprob_entropy_forward(logits, targets)` — pure forward, no
    autograd graph. Use this in benchmarks / inference-only paths.
  - `fused_logprob_entropy_naive(logits, targets)` — reference, never fast.
    Used by tests as a correctness anchor.
"""
import torch

from kernel_opt import _C


def _check_inputs(logits: torch.Tensor, targets: torch.Tensor):
    assert logits.is_cuda and targets.is_cuda
    assert logits.is_contiguous() and targets.is_contiguous()
    assert targets.dtype == torch.int64
    assert logits.shape[:-1] == targets.shape, \
        f"logits[...,V]={logits.shape} vs targets[...]={targets.shape}"


def _alloc_outputs(targets: torch.Tensor):
    out_shape = targets.shape
    return (
        torch.empty(out_shape, device=targets.device, dtype=torch.float32),
        torch.empty(out_shape, device=targets.device, dtype=torch.float32),
        torch.empty(out_shape, device=targets.device, dtype=torch.float32),
    )


def fused_logprob_entropy_naive(logits: torch.Tensor, targets: torch.Tensor):
    """Naive reference K1 forward (one-thread-per-row, scalar V loop)."""
    _check_inputs(logits, targets)
    logprob, entropy, lse = _alloc_outputs(targets)
    _C.fused_logprob_entropy_naive(logits, targets, logprob, entropy, lse)
    return logprob, entropy, lse


def fused_logprob_entropy_forward(logits: torch.Tensor, targets: torch.Tensor):
    """K1 forward only (no autograd). Use in benches / no-grad paths."""
    _check_inputs(logits, targets)
    logprob, entropy, lse = _alloc_outputs(targets)
    _C.fused_logprob_entropy_v1(logits, targets, logprob, entropy, lse)
    return logprob, entropy, lse


class _FusedLogprobEntropy(torch.autograd.Function):
    """K1 forward + backward as a torch autograd op.

    Forward saves (logits, targets, lse, entropy) for backward. Backward
    recomputes p = exp(x - lse) on the fly in a single streaming pass — no
    [B,S,V] intermediate allocation.
    """

    @staticmethod
    def forward(ctx, logits, targets):
        _check_inputs(logits, targets)
        logprob, entropy, lse = _alloc_outputs(targets)
        _C.fused_logprob_entropy_v1(logits, targets, logprob, entropy, lse)
        ctx.save_for_backward(logits, targets, lse, entropy)
        return logprob, entropy, lse

    @staticmethod
    def backward(ctx, g_logp, g_ent, g_lse):
        logits, targets, lse, entropy = ctx.saved_tensors

        # Outputs that the user didn't connect downstream get None grads;
        # substitute zeros so the backward kernel can run unconditionally.
        if g_logp is None:
            g_logp = torch.zeros_like(lse)
        if g_ent is None:
            g_ent = torch.zeros_like(lse)
        if g_lse is None:
            g_lse = torch.zeros_like(lse)

        # The kernel demands fp32 contiguous upstream grads.
        g_logp = g_logp.to(torch.float32, copy=False).contiguous()
        g_ent = g_ent.to(torch.float32, copy=False).contiguous()
        g_lse = g_lse.to(torch.float32, copy=False).contiguous()

        d_logits = torch.empty_like(logits)
        _C.fused_logprob_entropy_v1_backward(
            logits, targets, lse, entropy,
            g_logp, g_ent, g_lse, d_logits)

        # targets has no grad
        return d_logits, None


def fused_logprob_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """K1 (production), autograd-aware.

    Replaces TRL's `selective_log_softmax(logits, targets)` +
    `entropy_from_logits(logits)` with a single streaming pass per row,
    plus a streaming backward that recomputes softmax from the saved lse.

    Args:
        logits:  [..., V] bf16 / fp16 / fp32, contiguous on CUDA.
        targets: [...]    int64, contiguous on CUDA.

    Returns:
        (logprob, entropy, lse), each [...] fp32 on CUDA.
            logprob[i] = logits[i, targets[i]] - logsumexp(logits[i])
            entropy[i] = -sum_v softmax(logits[i])[v] * log_softmax(logits[i])[v]
            lse[i]     = logsumexp(logits[i])

    `targets` has no gradient.
    """
    return _FusedLogprobEntropy.apply(logits, targets)
