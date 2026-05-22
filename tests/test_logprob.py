"""Stage 2 correctness: K1 naive forward must match the eager reference.

Two reference paths:
  - PyTorch fp32: `F.log_softmax + gather`, `softmax * log_softmax sum`,
    `logsumexp`. Used as ground truth in fp32, tight tolerance.
  - TRL eager:    `selective_log_softmax(logits, ids)`, `entropy_from_logits(logits)`
    in bf16. Looser tolerance because TRL's bf16 path materializes log_softmax
    per row (its own precision profile).
"""
import pytest
import torch
import torch.nn.functional as F

from kernel_opt import fused_logprob_entropy, fused_logprob_entropy_naive


def _ref_fp32(logits: torch.Tensor, targets: torch.Tensor):
    """Ground-truth reference in fp32. Returns (logprob, entropy, lse)."""
    logits_f = logits.float()
    log_probs = F.log_softmax(logits_f, dim=-1)
    logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p = log_probs.exp()
    ent = -(p * log_probs).sum(-1)
    lse = torch.logsumexp(logits_f, dim=-1)
    return logp, ent, lse


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2, 16, 4096), (1, 1, 32000), (3, 8, 8192)])
def test_naive_matches_fp32_reference(dtype, shape):
    """Across dtypes and shapes, the kernel output matches the fp32 reference."""
    torch.manual_seed(0)
    B, S, V = shape
    logits = torch.randn(B, S, V, device="cuda", dtype=dtype) * 2.0  # nontrivial spread
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    logp, ent, lse = fused_logprob_entropy_naive(logits, targets)

    ref_logp, ref_ent, ref_lse = _ref_fp32(logits, targets)

    # Tolerance scales with dtype. fp32 -> tight; bf16/fp16 -> loose because
    # the kernel reads bf16 inputs and accumulates in fp32, so error is
    # bounded by the input quantization.
    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-4
    else:
        atol, rtol = 5e-3, 5e-3

    assert torch.allclose(logp, ref_logp, atol=atol, rtol=rtol), \
        f"logprob mismatch (dtype={dtype}): max abs err {(logp - ref_logp).abs().max()}"
    assert torch.allclose(ent, ref_ent, atol=atol, rtol=rtol), \
        f"entropy mismatch (dtype={dtype}): max abs err {(ent - ref_ent).abs().max()}"
    assert torch.allclose(lse, ref_lse, atol=atol, rtol=rtol), \
        f"lse mismatch (dtype={dtype}): max abs err {(lse - ref_lse).abs().max()}"


def test_at_least_as_accurate_as_trl_bf16():
    """In bf16, our fp32-accumulated kernel must be at least as close to the
    fp32 ground truth as TRL's eager path.

    Notable finding: TRL's `entropy_from_logits` runs the full
    `F.log_softmax + exp + mul + sum` chain in bf16 (output dtype follows
    input). The result is bf16-quantized to values like 8.5000 / 8.5625 / ...
    Our kernel reads bf16 inputs but accumulates the running max / sumexp /
    sum-x-exp(x) entirely in fp32 — measurably more accurate. Captured here
    for REPORT as a real side-benefit of fusing.
    """
    from trl.trainer.utils import selective_log_softmax, entropy_from_logits

    torch.manual_seed(1)
    B, S, V = 2, 32, 8192
    logits = torch.randn(B, S, V, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    # Ours
    logp, ent, _lse = fused_logprob_entropy_naive(logits, targets)
    # TRL's eager path
    trl_logp = selective_log_softmax(logits, targets).float()
    trl_ent = entropy_from_logits(logits).float()
    # fp32 ground truth
    gt_logp, gt_ent, _ = _ref_fp32(logits, targets)

    ours_logp_err = (logp - gt_logp).abs().max().item()
    trl_logp_err = (trl_logp - gt_logp).abs().max().item()
    ours_ent_err = (ent - gt_ent).abs().max().item()
    trl_ent_err = (trl_ent - gt_ent).abs().max().item()

    # Allow a small slack so this isn't flaky on a different RNG seed.
    assert ours_logp_err <= trl_logp_err * 1.2 + 1e-4, \
        f"logprob: ours {ours_logp_err:.4g} > TRL {trl_logp_err:.4g}"
    assert ours_ent_err <= trl_ent_err * 1.2 + 1e-4, \
        f"entropy: ours {ours_ent_err:.4g} > TRL {trl_ent_err:.4g}"

    # Also assert we're closer for entropy specifically (this is the strong
    # claim that ends up in REPORT).
    assert ours_ent_err < trl_ent_err, \
        f"entropy: expected ours strictly more accurate than TRL bf16, " \
        f"got ours={ours_ent_err:.4g}, trl={trl_ent_err:.4g}"


def test_extreme_values_no_overflow():
    """Numerical stability: large positive logits must not produce inf/nan."""
    B, S, V = 1, 4, 1024
    logits = torch.randn(B, S, V, device="cuda", dtype=torch.float32) + 60.0
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    logp, ent, lse = fused_logprob_entropy_naive(logits, targets)

    assert torch.isfinite(logp).all(), "logprob has inf/nan with large logits"
    assert torch.isfinite(ent).all(), "entropy has inf/nan with large logits"
    assert torch.isfinite(lse).all(), "lse has inf/nan with large logits"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2, 16, 4096), (1, 1, 32000), (3, 8, 8192), (4, 64, 128256)])
def test_v1_matches_fp32_reference(dtype, shape):
    """The optimized v1 kernel must match the fp32 reference under the same
    tolerances as naive."""
    torch.manual_seed(2)
    B, S, V = shape
    logits = torch.randn(B, S, V, device="cuda", dtype=dtype) * 2.0
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    logp, ent, lse = fused_logprob_entropy(logits, targets)
    ref_logp, ref_ent, ref_lse = _ref_fp32(logits, targets)

    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-4
    else:
        atol, rtol = 5e-3, 5e-3

    assert torch.allclose(logp, ref_logp, atol=atol, rtol=rtol), \
        f"v1 logprob mismatch: max abs err {(logp - ref_logp).abs().max()}"
    assert torch.allclose(ent, ref_ent, atol=atol, rtol=rtol), \
        f"v1 entropy mismatch: max abs err {(ent - ref_ent).abs().max()}"
    assert torch.allclose(lse, ref_lse, atol=atol, rtol=rtol), \
        f"v1 lse mismatch: max abs err {(lse - ref_lse).abs().max()}"


def test_uniform_logits_entropy_equals_logV():
    """Sanity: entropy of uniform distribution over V classes is log(V)."""
    B, S, V = 1, 1, 4096
    logits = torch.zeros(B, S, V, device="cuda", dtype=torch.float32)
    targets = torch.zeros(B, S, device="cuda", dtype=torch.int64)

    _logp, ent, lse = fused_logprob_entropy_naive(logits, targets)

    expected_ent = torch.tensor([float(torch.log(torch.tensor(V, dtype=torch.float64)))],
                                device="cuda")
    expected_lse = expected_ent.clone()  # logsumexp(0,...,0) = log(V)
    assert torch.allclose(ent.flatten(), expected_ent, atol=1e-4)
    assert torch.allclose(lse.flatten(), expected_lse, atol=1e-4)


# ============================================================================
# Backward correctness — vs pure-PyTorch autograd through the eager reference.
# ============================================================================


def _ref_forward_pt(logits: torch.Tensor, targets: torch.Tensor):
    """Pure-PyTorch forward. Differentiable. Returns (logp, ent, lse)."""
    log_probs = F.log_softmax(logits, dim=-1)
    logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p = log_probs.exp()
    ent = -(p * log_probs).sum(-1)
    lse = torch.logsumexp(logits, dim=-1)
    return logp, ent, lse


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 4, 256), (1, 8, 4096), (3, 16, 8192)])
def test_backward_matches_pytorch_autograd(dtype, shape):
    """Our backward must match what torch.autograd would produce on the
    eager reference computation, for arbitrary upstream gradients."""
    torch.manual_seed(7)
    B, S, V = shape
    base = torch.randn(B, S, V, device="cuda", dtype=dtype)
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    g_logp = torch.randn(B, S, device="cuda", dtype=torch.float32) * 0.5
    g_ent = torch.randn(B, S, device="cuda", dtype=torch.float32) * 0.5
    g_lse = torch.randn(B, S, device="cuda", dtype=torch.float32) * 0.5

    # Reference: pure PyTorch autograd
    x_ref = base.clone().requires_grad_(True)
    logp_r, ent_r, lse_r = _ref_forward_pt(x_ref, targets)
    loss_r = (logp_r * g_logp + ent_r * g_ent + lse_r * g_lse).sum()
    loss_r.backward()
    d_ref = x_ref.grad

    # Ours
    x_ours = base.clone().requires_grad_(True)
    logp, ent, lse = fused_logprob_entropy(x_ours, targets)
    loss = (logp * g_logp + ent * g_ent + lse * g_lse).sum()
    loss.backward()
    d_ours = x_ours.grad

    assert d_ours.dtype == dtype, f"d_logits dtype {d_ours.dtype} != input {dtype}"
    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-4
    else:
        # bf16 input → bf16 d_logits; same precision profile as TRL's path.
        atol, rtol = 1e-2, 1e-2

    assert torch.allclose(d_ours.float(), d_ref.float(), atol=atol, rtol=rtol), \
        f"max abs err {(d_ours.float() - d_ref.float()).abs().max()}"


def test_backward_only_logprob_grad():
    """Realistic case: only logprob grad flows back (entropy and lse outputs
    are unused). Must not crash and must match reference."""
    torch.manual_seed(8)
    B, S, V = 2, 8, 4096
    base = torch.randn(B, S, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)
    g_logp = torch.randn(B, S, device="cuda")

    # Reference
    x_ref = base.clone().requires_grad_(True)
    logp_r, _ent_r, _lse_r = _ref_forward_pt(x_ref, targets)
    (logp_r * g_logp).sum().backward()
    d_ref = x_ref.grad

    # Ours: do not use ent, lse downstream → autograd passes None for those
    x_ours = base.clone().requires_grad_(True)
    logp, _ent, _lse = fused_logprob_entropy(x_ours, targets)
    (logp * g_logp).sum().backward()
    d_ours = x_ours.grad

    assert torch.allclose(d_ours, d_ref, atol=1e-4, rtol=1e-4), \
        f"max abs err {(d_ours - d_ref).abs().max()}"


def test_gradcheck_small():
    """torch.autograd.gradcheck via finite differences on a tiny shape.
    Uses fp32 input + double-precision wrapper so gradcheck's eps math works.
    """
    torch.manual_seed(9)
    B, S, V = 1, 2, 64
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    def fn(x_double):
        # Cast double -> fp32 for the kernel, results back to double
        x = x_double.float()
        logp, ent, lse = fused_logprob_entropy(x, targets)
        return logp.double(), ent.double(), lse.double()

    x = torch.randn(B, S, V, device="cuda", dtype=torch.float64,
                    requires_grad=True)
    # Loose tolerances: eps -> fp32 quantization noise floor
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-3, atol=1e-2, rtol=1e-2,
                                     nondet_tol=1e-3)
