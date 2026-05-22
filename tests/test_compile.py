"""Stage 3 compile-compat: K1 must round-trip through `torch.compile`.

We don't insist on a graph-fused result here — Dynamo will graph-break at the
custom autograd.Function boundary (it's a pybind C extension, not a registered
torch.library op; that's a noted Stage 3 follow-up). The contract is:
  - `torch.compile(fn)` doesn't crash when fn calls our op
  - the compiled result matches eager (forward + backward)

We use `backend="aot_eager"` to avoid the Triton/Inductor dependency on
Windows — Inductor would only compile the surrounding ops anyway (our op
is opaque to it via graph-break), so we don't lose any signal.
"""
import pytest
import torch

from kernel_opt import fused_logprob_entropy


# `aot_eager`: AOTAutograd traces forward+backward, executes via eager. No
# Triton needed. Catches autograd-integration bugs (None grads, dtype, shape).
COMPILE_BACKEND = "aot_eager"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compile_forward_matches_eager(dtype):
    torch.manual_seed(0)
    B, S, V = 2, 8, 4096
    logits = torch.randn(B, S, V, device="cuda", dtype=dtype)
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    def fn(x, t):
        return fused_logprob_entropy(x, t)

    eager_out = fn(logits, targets)
    compiled = torch.compile(fn, backend=COMPILE_BACKEND, dynamic=False)
    comp_out = compiled(logits, targets)

    for e, c in zip(eager_out, comp_out):
        atol = 1e-5 if dtype == torch.float32 else 5e-3
        assert torch.allclose(e, c, atol=atol, rtol=atol), \
            f"compiled diverged from eager: max err {(e - c).abs().max()}"


def test_compile_backward_matches_eager():
    """Backward through a compiled function must match eager backward."""
    torch.manual_seed(1)
    B, S, V = 2, 4, 2048
    base = torch.randn(B, S, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

    def loss_fn(x, t):
        logp, ent, _lse = fused_logprob_entropy(x, t)
        return (logp + 0.1 * ent).sum()

    x_eager = base.clone().requires_grad_(True)
    loss_fn(x_eager, targets).backward()
    g_eager = x_eager.grad

    x_comp = base.clone().requires_grad_(True)
    compiled = torch.compile(loss_fn, backend=COMPILE_BACKEND, dynamic=False)
    compiled(x_comp, targets).backward()
    g_comp = x_comp.grad

    assert torch.allclose(g_comp, g_eager, atol=1e-4, rtol=1e-4), \
        f"compiled backward differs: max err {(g_comp - g_eager).abs().max()}"
