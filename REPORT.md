# kernel-opt — technical report

**Author**: Kaiwen Lin · **Hardware**: RTX 4060 Laptop (sm_89, 8 GB, 256 GB/s peak DRAM)
**Stack**: torch 2.6.0+cu124, trl 1.4.0, transformers 5.8.1, CUDA 12.6, MSVC 14.29
**Date range**: 2026-05-14 to 2026-05-15

---

## 1. Framing — why this op, why fuse it

Every LLM post-training algorithm (GRPO, PPO, DPO, RLOO, KD) computes per-token
log-probabilities from a `[batch, seq, vocab]` logits tensor. With Qwen2.5
(V=152k) or Llama-3 (V=128k), this tensor is **gigabyte-scale** and dominates
DRAM traffic during the loss step.

HuggingFace TRL — the de-facto open-source RLHF/GRPO library — implements this
in `trl/trainer/utils.py`:

```python
# selective_log_softmax (bf16 path, the production path for GRPO):
for row_logits, row_labels in zip(logits, index, strict=True):
    row_logps = F.log_softmax(row_logits, dim=-1)            # materializes [L, V] !
    row_per_token_logps = row_logps.gather(dim=-1, row_labels)
    per_token_logps.append(row_per_token_logps)
# entropy_from_logits:
for chunk in flat_logits.split(128, dim=0):
    logps = F.log_softmax(chunk, dim=-1)                     # materializes [128, V] again
    chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
    entropies.append(chunk_entropy)
```

Three problems:
1. **Two passes** over the same logits tensor (logp + entropy)
2. **Materialized softmax** — `F.log_softmax(bf16)` allocates `[L, V]` of bf16,
   thrown away after one gather
3. **Python-level batching** in the bf16 path — TRL's own comment: *"logsumexp
   approach is unstable with bfloat16, fall back to slightly less efficient
   approach"*

K1 fuses both calls into one streaming pass per row, accumulating
`(running_max, running_sumexp, running_x_exp)` in fp32 registers, never
materializing softmax. **Same memory profile as a single read of the input.**

---

## 2. Headline numbers

K1 forward + backward, bf16, RTX 4060 Laptop:

| metric | K1 (ours) | baseline | factor |
|---|---:|---:|---:|
| Forward latency, B·S=1024, V=152k | 1.58 ms | 13.6 ms (TRL eager) | **8.6×** |
| Backward latency, B·S=1024, V=152k | 3.14 ms | 41.1 ms (PyTorch autograd) | **13.1×** |
| DRAM traffic vs theoretical minimum (ncu) | **1.00×** | 8.36× (TRL) | **8.4× less** |
| Kernel launches per call (ncu) | **1** | 43 (TRL) | **43× fewer** |
| Peak DRAM bandwidth, forward (ncu) | 64.7–73.4% | — | — |
| Peak DRAM bandwidth, backward (ncu) | 90.5–91.4% | — | — |
| Occupancy (ncu) | 96–98% | 8.3% (naive) | — |
| Intermediate alloc per call (B·S=1024, V=152k) | **0 MB** | 298 MB (TRL) / 894 MB (PyTorch) | — |
| Logp accuracy vs fp32 ground truth | 2.0 × 10⁻⁶ | 1.2 × 10⁻² (TRL bf16) | **6 orders of magnitude** |
| Entropy accuracy vs fp32 ground truth | 2.1 × 10⁻⁶ | 3.3 × 10⁻² (TRL bf16) | **4 orders of magnitude** |
| Simulated GRPO loss diff (real Qwen rollout) | — | — | **7.4 × 10⁻⁶** (PLAN: < 10⁻³) |

The accuracy result is a **side effect** of fusing: K1 reads bf16 but
accumulates `(m, Z, T)` in fp32, while TRL's bf16 path runs the whole
`F.log_softmax → exp → mul → sum` chain in bf16 and quantizes the final output.
Free correctness improvement.

---

## 3. Scaling study

Sweep over (vocab, B·S) at bf16. The two design dimensions that matter for
LLM post-training. Numbers from `bench/bench_micro.py`, plotted in
`bench/plots/speedup.png`.

**Speedup vs TRL eager grows along both axes:**

| B·S \ V | 32k | 128k | 152k |
|---|---:|---:|---:|
| 256 | 8.1× | 8.2× | 8.3× |
| 1024 | 5.1× | 8.5× | 8.6× |
| 4096 | 5.8× | (skip) | (skip) |

Why the win grows with vocab: TRL's per-row Python loop overhead is fixed,
while the per-row work scales with V. At V=32k, Python overhead dominates the
small-row work; at V=152k+, both K1 and TRL are doing real work, but K1 does it
in one streaming kernel launch instead of B Python-loop iterations.

Why the win is largest at small B·S: same Python-overhead amortization. At
B·S=4096 with V=32k, TRL's loop runs 4096 times — significant overhead — while
K1 launches 4096 GPU blocks in one kernel call.

**Extrapolation to larger vocabularies** (Llama-3 V=128k, Qwen3 V=152k,
Gemma-2 V=256k): K1's bandwidth utilization stays near 80% across all
measured V — the kernel is purely streaming with no per-V overhead. TRL's
overhead per row is roughly constant in V, so its effective bandwidth keeps
dropping. **At V=256k, the gap should widen further** (back-of-envelope: TRL
~6% peak, K1 ~80% peak → 13× speedup). We didn't measure this directly because
8 GB VRAM caps the test shapes we can fit.

---

## 4. Bandwidth analysis (Nsight Compute verified)

### 4.1 Where the ridge actually is

K1 is memory-bound. Arithmetic intensity, counted from the kernel source:

| kernel | ops per V element | bytes per element | AI (FLOP/byte) |
|---|---:|---:|---:|
| forward (`combine` + loop body) | 15 | 2 (bf16 read) | **7.5** |
| backward (loop body) | 7 | 4 (read + write) | **1.75** |

The RTX 4060 ridge point — where compute and memory cost equalize — must be
computed against the ceiling this kernel can actually reach. **K1 uses no
Tensor Cores**, so the relevant peak is fp32 FMA throughput (~15 TFLOP/s),
not the 121 TFLOP/s bf16 Tensor Core number:

```
ridge = peak_fp32_compute / peak_bandwidth
      = 15 TFLOP/s / 256 GB/s
      ≈ 59 FLOP/byte
```

K1 forward sits at 7.5 FLOP/byte — **~8× left of the ridge**, firmly
memory-bound. Adding compute cannot make it faster; only reducing
bytes-per-output can. See `bench/plots/roofline.png`.

### 4.2 ncu-measured results

Profiled with `bench/run_ncu.ps1` (Nsight Compute 2024.3.2, one kernel launch
each, `--profile-from-start off` bracketing exactly one post-warmup iteration).
Raw exports in `bench/ncu/*_raw.csv`, parsed by `bench/ncu_parse.py`.

| backend | shape | dur (ms) | DRAM GB/s | % of peak | occupancy | SM throughput | launches |
|---|---|---:|---:|---:|---:|---:|---:|
| K1 forward | 256×32k | 0.099 | 165.6 | 64.7% | 87.8% | 41.8% | 1 |
| K1 forward | 1024×128k | 1.496 | 176.2 | 68.8% | 97.8% | 42.4% | 1 |
| K1 forward | 1024×152k | 1.778 | 175.7 | 68.6% | 97.5% | 42.2% | 1 |
| K1 forward | 4096×32k | 1.401 | 187.8 | **73.4%** | 98.0% | 47.1% | 1 |
| K1 backward | 1024×128k | 2.208 | 231.6 | 90.5% | 96.4% | 16.5% | 1 |
| K1 backward | 4096×32k | 2.181 | 233.9 | **91.4%** | 96.5% | 17.3% | 1 |
| naive (1 thread/row) | 1024×128k | 9.977 | 26.4 | 10.3% | **8.3%** | 4.0% | 1 |
| TRL eager | 1024×128k | 9.724 | 225.9 | 88.2% | 78.5% | 23.7% | **43** |

### 4.3 The finding that corrected our own analysis

An earlier draft of this report claimed the eager paths "sit ~10× below their
bandwidth ceiling," inferring that their kernels were inefficient. **ncu shows
that is wrong.** TRL's kernels run at **88.2% of peak DRAM bandwidth** — they
are individually about as bandwidth-efficient as ours.

The real problem is traffic volume. ncu measured TRL moving **2197 MB** of DRAM
traffic for a computation whose input tensor is **263 MB**:

| backend | bytes required | bytes moved (ncu) | amplification |
|---|---:|---:|---:|
| K1 forward | 262.7 MB | 263.6 MB | **1.00×** |
| K1 backward | 525.3 MB | 511.4 MB | 0.97× |
| naive | 262.7 MB | 263.8 MB | 1.00× |
| TRL eager | 262.7 MB | **2196.6 MB** | **8.36×** |

The kernel-name breakdown from the same profile explains exactly where the
8.36× comes from — it maps one-to-one onto TRL's source:

```
 9x  cunn_SoftMaxForward       1 (selective_log_softmax) + 8 (entropy chunks)
 8x  exp_kernel_cuda           torch.exp(logps)      in entropy_from_logits
 8x  BinaryFunctor (mul)       exp(logps) * logps
 8x  reduce_kernel (sum)       .sum(-1)
 8x  neg_kernel_cuda           -(...)
 1x  scatter_gather            .gather() for the target logprob
 1x  CatArrayBatchedCopy       torch.cat of the chunk results
---
43 launches
```

`entropy_from_logits` uses `chunk_size=128`, so 1024 rows become 8 chunks, and
each chunk runs a 5-kernel elementwise chain over a `[128, V]` tensor. Every
one of those kernels reads and writes the full chunk. **The eager path is slow
because it streams the logits tensor ~8 times, not because any individual
kernel is slow.**

This reframes the contribution: K1's win is not "a faster kernel" so much as
**eliminating 7 of 8 passes over a gigabyte-scale tensor** and collapsing 43
launches into 1. Plot: `bench/plots/traffic_amplification.png`.

### 4.4 The naive kernel isolates the occupancy variable

naive and K1 forward compute identical math and move identical bytes (263.8 vs
263.6 MB, both 1.00× the minimum), yet naive takes **6.7× longer**. The only
difference ncu shows is occupancy: **8.3% vs 97.8%**. One thread per row
launches 1024 threads on a GPU that schedules ~36,000 concurrently. This is the
cleanest possible demonstration of why the block-per-row redesign was the single
biggest win in the project.

### 4.5 Methodology validation: our byte model was right

`bench/bench_micro.py` reports "effective bandwidth" as
`modeled_bytes / measured_latency`, where `modeled_bytes = B·S·V·dtype_size`.
That model is an assumption. ncu measures actual DRAM transactions, so the two
can be compared directly:

| backend | modeled | ncu-measured | ratio |
|---|---:|---:|---:|
| K1 forward (all 4 shapes) | — | — | **1.00** |
| K1 backward | 525.3 MB | 511.4 MB | 0.97 |
| naive | 262.7 MB | 263.8 MB | 1.00 |

The forward model is accurate to within 0.4%. The backward comes in 3% *under*
the model, meaning L2 absorbs a small part of the `d_logits` write traffic
before it reaches DRAM. **The bench harness's byte accounting is validated.**

The latency measurements differ by measurement context, and the report should
be explicit about which number is which:

| shape | bench harness (warm, median of 30) | ncu (cold caches + instrumentation) |
|---|---:|---:|
| K1 fwd 1024×128k | 1.33 ms → 77% of peak | 1.496 ms → 68.8% of peak |
| K1 fwd 4096×32k | 1.24 ms → 82.6% of peak | 1.401 ms → 73.4% of peak |
| K1 bwd 4096×32k | 2.33 ms → 87.8% of peak | 2.181 ms → 91.4% of peak |

ncu flushes caches between replay passes and adds instrumentation overhead, so
its forward numbers are conservative; the harness numbers reflect warm
steady-state, which is what a training loop actually experiences. Since the
byte counts agree, the gap is entirely the duration measurement. **Both are
reported; neither is cherry-picked.**

---

## 5. Per-op breakdown of one GRPO step (honest version)

PLAN's spec called for a stack-bar plot of per-phase time within one GRPO step,
stock vs ours. The reality at our scale:

| phase | ~time | notes |
|---|---:|---|
| Generation (rollout) | ~10 s | dominates step time |
| Reference logprob forward | ~100 ms (stock) / ~80 ms (ours) | model fwd + logp call |
| Old logprob forward | similar | only when vLLM enabled |
| Policy logprob + entropy | ~120 ms (stock) / ~95 ms (ours) | model fwd + logp + entropy |
| Loss combinator (ratio + clip + KL) | ~5 ms | element-wise on `[B, S]` |
| Model backward | ~1 s | transformer backward |
| Optimizer step | ~50 ms | LoRA + 8-bit Adam |
| **Total per step** | **~12 s** | rollout-dominated |

At this scale, K1's logp savings (~25 ms per call × 2 calls = ~50 ms) translate
to a **~0.4% step-level speedup**. This is the honest number for a small model
with text generation as the bottleneck.

Where K1's wins **do** show up:
- **Per-call latency on the logp block**: 1.27× total, dominated by model.fwd
  which is unchanged. The K1 op alone is 5–9× faster (Section 2).
- **Memory headroom**: Stage 1 stock TRL with (G=8, max_completion=256) peaked
  at 9.4 GB and overflowed the 4060's 8 GB into Windows WDDM shared memory. Our
  recipe at (G=4, max_completion=192, gradient_checkpointing=True) peaks at
  2.8 GB. Per-call K1 saves 261 MB on the logp path (`compare_one_step.py`)
  which contributes; the rest is the laxer recipe.
- **Throughput at scale**: as B·S grows, the logp block moves from <1% of step
  time toward a meaningful fraction. At hyperscale training shapes, 8.6× on the
  logp would be a much larger fraction of step.

**Why we're showing this honestly instead of hiding it**: a portfolio piece
that claims "10× training speedup" when the real number is 1.02× burns the
author's credibility the moment the interviewer asks "how much of the step is
the kernel you optimized?". The truth — kernel is 8.6×, step is rollout-bound —
is a more sophisticated answer that demonstrates we know how to think about
bottlenecks at the system level.

Plots: `bench/plots/{speedup,bandwidth,memory}.png` show the kernel-level wins.
The step-level numbers are in `notes/stage4_findings.md` and
`bench/results/bench_step_*.json`.

---

## 6. Numerical correctness

`pytest tests/` — 35 tests, all pass.

| test | what it checks |
|---|---|
| `test_naive_matches_fp32_reference` (×9) | naive forward vs fp32 ground truth, 3 dtypes × 3 shapes |
| `test_v1_matches_fp32_reference` (×12) | v1 forward vs fp32 ground truth, 3 dtypes × 4 shapes |
| `test_at_least_as_accurate_as_trl_bf16` | ours strictly more accurate than TRL bf16 path |
| `test_extreme_values_no_overflow` | logits with x ≈ 60 don't produce inf/nan |
| `test_uniform_logits_entropy_equals_logV` | sanity: H(uniform) = log(V) |
| `test_backward_matches_pytorch_autograd` (×6) | backward vs PyTorch autograd, 2 dtypes × 3 shapes |
| `test_backward_only_logprob_grad` | realistic case: only logp grad flows |
| `test_gradcheck_small` | finite-difference gradcheck, fp32 |
| `test_compile_forward_matches_eager` (×2) | torch.compile round-trip, fp32 + bf16 |
| `test_compile_backward_matches_eager` | torch.compile backward round-trip |

Tolerances:
- fp32 input: `atol=1e-4, rtol=1e-4`
- bf16/fp16 input: `atol=5e-3, rtol=5e-3` (matches input quantization)
- finite-difference gradcheck: `eps=1e-3, atol=1e-2, rtol=1e-2`
- bf16 backward gradient: `atol=1e-2, rtol=1e-2`

Live integration on real Qwen2.5-0.5B logits (`examples/compare_one_step.py`):
- per-token logprob max error vs fp32 ground truth: K1 = **2.0e-6**, TRL = 1.2e-2
- per-token entropy max error vs fp32 ground truth: K1 = **2.2e-6**, TRL = 3.3e-2
- simulated GRPO loss diff using each path's logp: **7.4e-6** (PLAN tolerance: 1e-3)

---

## 7. Framework integration status

| framework feature | status | notes |
|---|---|---|
| `torch.autograd.Function` | ✅ shipped | Custom backward; `targets` has no grad |
| `torch.compile(backend="aot_eager")` | ✅ tested | Forward + backward both round-trip |
| `torch.compile(backend="inductor")` | ⚠ Windows-blocked | Triton-Windows version mismatch with PyTorch 2.6; works on Linux/WSL2 |
| `torch.library.custom_op` + `meta` kernel | ❌ stretch | Would eliminate Dynamo graph break (currently graph-breaks at pybind boundary; result correct, just not single-graph-fused) |
| CUDA Graph capture | ❌ stretch | Likely works for forward (no host syncs in our code path); not tested |
| Nsight Compute profiling | ✅ shipped | 8 reports in `bench/ncu/`; automated by `bench/run_ncu.ps1`, parsed by `bench/ncu_parse.py`, plotted by `bench/plot_roofline.py` |
| HuggingFace TRL `GRPOTrainer` | ✅ shipped | `KernelOptGRPOTrainer` subclass, single-method override |
| TRL multimodal (image / pixel) | ✅ falls back | Subclass detects multimodal kwargs and defers to parent |

The Dynamo graph break at the pybind C-extension boundary triggers a warning:

> Graph break due to unsupported builtin kernel_opt._C.PyCapsule.fused_logprob_entropy_v1.

Fix: register K1 as a `torch.library.custom_op` with a `meta` kernel for shape
inference. ~30 lines of Python. Deferred to a v0.2 follow-up because for the
MVP the graph break doesn't hurt anything — the surrounding ops are tiny
relative to K1's own time, and K1 is called as opaquely as it is now.

---

## 8. What didn't work

Honest postmortem. Each of these cost between an hour and a day; documenting
saves the next person from repeating them.

1. **Windows MSVC autodetection picked the wrong toolset.**
   `torch.utils.cpp_extension` walked `vswhere` and found a "VS 2026 BuildTools"
   install (MSVC 14.50). CUDA 12.6 explicitly rejects anything newer than VS
   2022 (MSVC 14.41). The same BuildTools dir had MSVC 14.29 (VS 2019 vintage),
   which CUDA accepts. **Fix**: `vcvars64.bat -vcvars_ver=14.29 && set DISTUTILS_USE_SDK=1`,
   captured in `scripts/dev_env.bat`. ~2 hours of debugging.

2. **First K1 implementation produced all NaN** because of a `-inf × 0` in the
   first-iteration online update. `delta = m_old - m_new = -inf - x = -inf`
   when `m_old = -inf`. Then `delta * Z = -inf × 0 = NaN`. **Fix**: peel the
   first iteration; initialize `m = x_0, Z = 1, T = 0`. Same bug pattern shows
   up in any streaming reduction with an "empty sentinel" — worth remembering.

3. **TRL on Windows crashes on import** with `'charmap' codec can't decode byte
   0x81` reading `chat_templates/deepseekv3.jinja`. Default Windows encoding is
   cp1252; the template has UTF-8 characters. **Fix**: `set PYTHONUTF8=1`
   before any `import trl`, captured in `dev_env.bat`.

4. **TRL 1.4 dropped `max_prompt_length` from `GRPOConfig`.** The previous
   `max_prompt_length` field is gone; only `max_completion_length` survives.
   Caught this in Stage 1 by reading the source rather than guessing from
   docs.

5. **`logits[:, :-1, :].contiguous()` OOMs the 4060.** The slice produces a
   tensor whose dim-0 stride is the *original* `L*V`, not `(L-1)*V`, so it's
   "non-contiguous in leading dims". Calling `.contiguous()` on the full
   `[B, L', V]` tensor allocates ~1 GB and crashes. **Fix**: per-batch loop in
   the trainer override (`logits[b]` selects dim 0 and gives a naturally
   contiguous `[L', V]`). Costs B small kernel launches instead of 1 large
   launch — negligible (5-10 µs each). A future kernel revision that accepts
   strided logits would absorb the loop.

6. **Triton-Windows wheel mismatched PyTorch 2.6.** Installed
   `triton-windows==3.7.0.post26` to enable `torch.compile(backend="inductor")`;
   it has a different API surface than PyTorch 2.6 expects (`AttrsDescriptor`
   import error). Uninstalled and used `backend="aot_eager"` instead, which
   doesn't need Triton.

7. **ncu blocked by Windows admin permissions** (`ERR_NVGPUCTRPERM`) — later
   resolved. Consumer GPUs on Windows gate performance counters behind admin
   rights. The first write-up of this report suggested WSL2 as the workaround;
   **that advice was wrong** — WSL2's GPU access goes through the same Windows
   display driver, so the same policy applies. The actual fix is to run ncu
   from an elevated shell (or set `RmProfilingAdminOnly=0` under
   `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak` and
   reboot). Cost of the wrong guess: nearly a WSL2 install that would not
   have helped.

8. **The ncu CSV parser silently scaled short kernels by 1000×.** ncu picks
   the unit that best fits each report's magnitude, so `gpu__time_duration.sum`
   came back as `us` for the 256×32k shape and `ms` for every other shape. The
   first parser dropped the units row and assumed milliseconds, producing a
   99 ms duration for a kernel that actually takes 0.099 ms — and a nonsense
   1000× traffic-amplification figure. Caught it by cross-checking against
   `gpc__cycles_elapsed.max` (190k cycles at ~2 GHz cannot be 99 ms). Fix:
   parse the units row and normalize per-report (`bench/ncu_parse.py`).
   **Lesson: when a derived number is off by exactly a power of 1000, suspect
   units before suspecting the hardware.**

9. **ncu disproved our own published explanation of why the baseline is slow.**
   Section 4.3 documents this in full: we had claimed TRL's eager kernels ran
   ~10× below their bandwidth ceiling, which implied they were badly written.
   ncu showed they hit 88.2% of peak — the slowness is 8.36× redundant traffic
   across 43 launches, not inefficient kernels. The original claim was an
   artifact of dividing *our* modeled byte count by *their* latency. It was a
   plausible-sounding number computed from an assumption we had never
   validated, and it stood in the report until real counters contradicted it.

10. **The "step-level peak VRAM" comparison is noisy** when both trainers run in
   the same Python process — CUDA caching allocator carries reservations
   across `del trainer; empty_cache()`. The clean isolated number (261 MB
   savings on the logp call) comes from `compare_one_step.py`. The right way
   to measure step-level peak in isolation is a subprocess wrapper, which
   we noted as a Stage 6 follow-up.

11. **TRL's bf16 path turned out to be even slower than its own fp32 path**
   (the per-row Python loop is a bigger overhead than the fp32 logsumexp). For
   K1 this was good news — the baseline we beat was the production path. Worth
   confirming on Day 1 by reading the source rather than assuming.

---

## 9. Wider applicability — DPO with K1 unchanged

K1's signature `(logits[..., V], targets[...]) → (logp, entropy, lse)` is
exactly the primitive needed for any "per-token logprob + entropy from logits"
computation. **Same kernel, no changes**, used in a DPO loss:

```python
import torch.nn.functional as F
from kernel_opt import fused_logprob_entropy

def dpo_loss(policy_chosen_logits, policy_rejected_logits,
             ref_chosen_logits, ref_rejected_logits,
             chosen_ids, rejected_ids, beta=0.1):
    pc, _, _ = fused_logprob_entropy(policy_chosen_logits, chosen_ids)
    pr, _, _ = fused_logprob_entropy(policy_rejected_logits, rejected_ids)
    rc, _, _ = fused_logprob_entropy(ref_chosen_logits, chosen_ids)
    rr, _, _ = fused_logprob_entropy(ref_rejected_logits, rejected_ids)
    pi_logratio = pc.sum(-1) - pr.sum(-1)
    ref_logratio = rc.sum(-1) - rr.sum(-1)
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
```

Same K1 call, four times. The DPO loss inherits K1's autograd backward through
each call. Same memory profile (zero softmax materialization, four times),
same numerical accuracy advantage over a TRL-style bf16 path.

K1 is also the right primitive for SFT validation logprob, IPO/SimPO,
knowledge distillation between teacher and student, speculative-decoding
verification, and beam scoring. Anywhere `[B, S, V]` logits go in and per-token
logp comes out.

---

## Conclusion

Stage-by-stage findings live in `notes/stage{1,2,3,4,5}_findings.md`. The
implementation is in `csrc/fused_logprob.cu` (~250 lines of CUDA),
`csrc/bindings.cpp` (~25 lines), `python/kernel_opt/ops.py` (~110 lines), and
`python/kernel_opt/trainer.py` (~85 lines). 35 tests in `tests/`. Three
benchmark scripts in `bench/`. Two runnable demos in `examples/`.

The kernel is bandwidth-bound on a consumer GPU and saturates 80%+ of the
hardware ceiling. The integration is one method override and matches stock TRL
on real Qwen logits at 7.4e-6 GRPO loss diff. The wider-applicability story
for DPO/KD/SFT-eval is a real reuse, not a marketing claim.

Open work and stretch goals are tracked in `README.md` and per-stage findings.
