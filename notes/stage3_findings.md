# Stage 3 findings — K1 backward + autograd + compile

## Status: complete

PLAN exit criterion (`pytest tests/test_logprob.py tests/test_compile.py passes`) — met.

- All 35 tests pass (32 in `test_logprob.py` + 3 in `test_compile.py`).
- Backward kernel matches PyTorch autograd through the eager reference for fp32 / bf16 across multiple shapes.
- `torch.autograd.gradcheck` (finite-difference) passes on a small fp32 shape.
- Forward + backward both round-trip through `torch.compile(backend="aot_eager")`.

## Backward derivation (the math)

K1 forward outputs three quantities per row, all functions of `logits[v]`:

```
logprob = x_t - lse                       (t = target index)
entropy = lse - sum_v p_v * x_v
lse     = log sum_v exp(x_v)
```

Upstream gradients arrive as `(g_logp, g_ent, g_lse)`, each `[B, S]`.
Per-element gradient w.r.t. `x_v` (using `p_v = exp(x_v - lse)`, `log_p_v = x_v - lse`):

| from            | term                                                 |
|-----------------|------------------------------------------------------|
| logprob output  | `g_logp * (δ(v==t) - p_v)`                          |
| lse output      | `g_lse * p_v`                                       |
| entropy output  | `g_ent * (-p_v) * (log_p_v + entropy)`              |

Combined and factored:

```
d_x[v] = p_v * (g_lse - g_logp - g_ent * (log_p_v + entropy))
       + (v == t ? g_logp : 0)
```

The kernel hoists the v-independent part out of the V loop (`const_part = g_lse - g_logp - g_ent * entropy`), so each iteration does:
- one global read (`x_v`)
- one `expf`, one mul, one fma to compute `dval`
- one global write (`d_x[v]`)

That's **2 global accesses per V step**, the theoretical minimum for a pointwise op that produces `d_x` from `x`.

## Backward perf (bf16, RTX 4060)

| (B, S, V)         | ours ms | pytorch autograd ms | **speedup** | ours bw % peak | bwd alloc reduction |
|-------------------|--------:|--------------------:|------------:|---------------:|--------------------:|
| (1, 256, 32000)   |   0.190 |               1.577 |    **8.3×** |          67.2% |            5×       |
| (1, 1024, 128256) |   2.446 |              34.761 |   **14.2×** |        **83.9%** |            5×       |
| (1, 1024, 152064) |   3.135 |              41.103 |   **13.1×** |          77.6% |            5×       |
| (1, 4096, 32000)  |   2.333 |              33.428 |   **14.3×** |        **87.8%** |            5×       |

**Backward is faster than forward at peak bandwidth** (87.8% vs 82.6%) — likely because backward is purely pointwise (no warp/block reduction needed). It just streams x → d_x with a couple of `expf` calls per element.

The `bwd_alloc_MB` 250 MB number for ours is **just the d_logits output tensor itself** (1024 × 128256 × 2 = 263 MB) — required output, not overhead. PyTorch's 1250 MB includes the materialized softmax + log_softmax intermediates kept around for autograd.

## Why backward is straightforward but easy to get wrong

The math is one streaming pass per row, and the values needed are:
- `lse[i]` (saved from forward, scalar per row)
- `entropy[i]` (saved from forward, scalar per row)
- `logits[i, v]` (re-read in backward, `[B,S,V]` total)
- `targets[i]` (saved, scalar per row)
- `g_logp[i]`, `g_ent[i]`, `g_lse[i]` (upstream, scalars per row)

**No softmax materialization.** `p_v` is computed on the fly inside the V loop register. Same memory footprint as forward.

The trap I almost fell into: the `entropy` saved value vs recomputing entropy in backward. Saving lets us do one pass; recomputing would require two passes (one to get entropy, then one for d_logits using it). Adding `entropy` to `save_for_backward` is "free" — it's just `[B,S]` floats, tiny next to logits.

## torch.compile compatibility

Forward and backward both round-trip through `torch.compile(backend="aot_eager", dynamic=False)`. AOTAutograd traces forward + backward, executes via eager — catches autograd-integration bugs without needing Triton.

**Known limitation (deferred to Stage 6 stretch):** Dynamo graph-breaks at the call to our pybind C extension because it's not a registered `torch.library` op. PyTorch warns:

> Graph break due to unsupported builtin kernel_opt._C.PyCapsule.fused_logprob_entropy_v1.

The fix is to register K1 as a `torch.library.custom_op` with a `meta` kernel (returns dummy tensors with correct shape/dtype for shape inference). Then Dynamo sees the op as a known node and can keep compiling around it. Mechanically:

```python
@torch.library.custom_op("kernel_opt::fused_logprob_entropy", mutates_args=())
def _op(logits, targets):
    return fused_logprob_entropy_forward(logits, targets)

@_op.register_fake
def _meta(logits, targets):
    out_shape = targets.shape
    return (torch.empty(out_shape, dtype=torch.float32, device=logits.device),
            torch.empty(out_shape, dtype=torch.float32, device=logits.device),
            torch.empty(out_shape, dtype=torch.float32, device=logits.device))
```

Plus a separate registration for the backward gradient (using `torch.library.register_autograd`). Adds maybe 30 lines but requires care with the `mutates_args` semantics. Worth doing in Stage 6 if there's time, since it'd make the project look more "production".

For MVP: graph-breaking around K1 is **not a perf problem** — the surrounding ops are tiny compared to K1's own time, and the actual K1 call is just as fast either way.

## Triton on Windows: a brief detour

Tried `pip install triton-windows` to make `torch.compile(backend="inductor")` work; the wheel installs but its API doesn't match what PyTorch 2.6 expects (`AttrsDescriptor` import error). Uninstalled and switched all compile tests to `aot_eager` backend instead. Inductor on Windows is known-flaky as of mid-2025; sticking with `aot_eager` for `torch.compile` tests is the right MVP choice.

If a real perf comparison vs Inductor is needed for Stage 5/6, run from WSL2 (Inductor + Triton work cleanly there). Document either way.

## Public API surface (post-Stage-3)

Three public names from `kernel_opt`:

| function | use |
|---|---|
| `fused_logprob_entropy(logits, targets)` | autograd-aware. Use in training. |
| `fused_logprob_entropy_forward(logits, targets)` | pure forward (no autograd graph). Use in benches / inference. |
| `fused_logprob_entropy_naive(logits, targets)` | reference. Used by tests. |

The autograd-aware one wraps a `torch.autograd.Function`. Targets has no gradient (returned as `None` from backward).

## Test inventory after Stage 3

- `tests/test_logprob.py` — 32 tests
  - 9 forward fp32-reference tests for naive (3 shapes × 3 dtypes)
  - 1 vs-TRL accuracy test (we win)
  - 1 extreme-values stability test
  - 12 forward fp32-reference tests for v1 (4 shapes × 3 dtypes)
  - 1 uniform-entropy sanity test
  - 6 backward-vs-pytorch-autograd tests (3 shapes × 2 dtypes)
  - 1 backward-only-logprob-grad test (realistic GRPO case)
  - 1 finite-difference gradcheck test
- `tests/test_compile.py` — 3 tests
  - 2 forward compile-matches-eager tests (fp32, bf16)
  - 1 backward compile-matches-eager test

## Next

Stage 4: subclass `trl.GRPOTrainer` → `KernelOptGRPOTrainer`. Override
`_get_per_token_logps_and_entropies` (located in Stage 1, lines 1046-1125 of
trl/trainer/grpo_trainer.py) to call our op. Verify 1 GRPO step matches stock
TRL within tolerance. That's the integration story for the resume bullet.
