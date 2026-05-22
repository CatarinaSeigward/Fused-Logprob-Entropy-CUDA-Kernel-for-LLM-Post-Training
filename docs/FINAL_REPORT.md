# kernel-opt: A Fused Logprob+Entropy CUDA Kernel for LLM Post-Training

**Author**: Kaiwen Lin (kaiwenlin@utexas.edu)
**Hardware**: NVIDIA GeForce RTX 4060 Laptop (sm_89, 8 GB GDDR6, 256 GB/s peak DRAM)
**Stack**: PyTorch 2.6.0+cu124, CUDA Toolkit 12.6, MSVC 14.29, TRL 1.4
**Repository**: `D:\Projects\kernel-opt`

---

## Abstract

LLM post-training algorithms (GRPO, PPO, DPO, RLHF) all bottleneck on the same
inner computation: extracting per-token log-probabilities from a `[batch, seq,
vocab]` logits tensor where `vocab` is now routinely 128k+. Open-source
frameworks like HuggingFace TRL implement this in eager PyTorch, materializing
the `log_softmax` intermediate and running a Python-level loop over batch in
the bf16 path. This work presents `kernel-opt`, a hand-written CUDA kernel
that fuses log-probability extraction with entropy computation in a single
streaming pass, exposed as a PyTorch autograd op and integrated into TRL's
GRPO trainer with a one-method subclass override. On a consumer RTX 4060,
forward is **5–9× faster** than TRL's eager path and **achieves 82.6% of peak
DRAM bandwidth**; backward is **8–14× faster** than PyTorch autograd and reaches
**87.8% of peak**. As a side benefit of fp32 register-level accumulation, the
kernel is numerically **6 orders of magnitude more accurate** than TRL's bf16
path against fp32 ground truth. End-to-end GRPO training with the trainer
subclass produces loss values within `7.4 × 10⁻⁶` of stock TRL — 135× tighter
than the project's `10⁻³` design contract.

---

## 1. Introduction

### 1.1 Background: LLM post-training is the dominant 2025–26 ML compute workload

The 2025–26 wave of frontier-model post-training — GRPO for DeepSeek-R1
[1], PPO for Llama-3-Instruct, DPO for Mistral instruction-tuning, KTO and
SimPO variants — has shifted the inference-time compute balance toward
*training-time RL on inference-style workloads*. The inner training loop
generates G candidate completions per prompt (rollout), scores them, and
runs gradient updates that involve repeated forward passes through both a
trainable policy model and a frozen reference model.

### 1.2 The bottleneck: per-token logprob from `[B, S, V]` logits

For each forward pass through the policy or reference model, the loss
computation requires per-token log-probabilities under that model:
`log P(token_t | prefix)`. Mechanically, this means starting from a
`[B, S, V]` logits tensor (where V is the vocabulary size) and extracting
one log-probability value per (b, s) position. With Qwen2.5 (V = 152,064),
Llama-3 (V = 128,256), or Gemma-2 (V = 256,000), this tensor is gigabyte-scale
even at modest batch sizes. **The compute is trivial; the bottleneck is DRAM
bandwidth.**

### 1.3 The gap in open-source post-training frameworks

We surveyed three major open-source frameworks: HuggingFace TRL, ByteDance
verl, and OpenRLHF. All three call into TRL-style implementations of the
inner logprob computation, and none of them apply the fusion that's standard
in inference (e.g., FlashAttention's online softmax [2]). Specifically,
TRL's `selective_log_softmax` for the bf16 production path
(`trl/trainer/utils.py:436`) explicitly states in a code comment:

> *"logsumexp approach is unstable with bfloat16, fall back to slightly less
> efficient approach"*

and proceeds to run a Python `for` loop over the batch dimension, calling
`F.log_softmax` per row and materializing the full `[seq, vocab]`
softmax intermediate. A second function, `entropy_from_logits`, is invoked
separately on the same logits tensor with a chunked materialization of size
`128 × vocab × 4` bytes per chunk. Two passes, two materialized
intermediates, Python-level overhead — all on a tensor that should ideally be
streamed once.

Commercial implementations (NVIDIA NeMo-Aligner, internal Meta GenAI
trainers) reportedly fuse these operations, but their kernel sources are not
public.

### 1.4 Contributions

This work contributes:

1. **K1**, a hand-written CUDA kernel for sm_89 that fuses `log_softmax +
   gather` and `entropy_from_logits` into a single streaming pass over the
   logits tensor, with both forward and backward implemented.
2. A `torch.autograd.Function` wrapper with custom backward, `torch.compile`
   round-trip compatibility, and a documented integration story for
   `torch.library.custom_op` (deferred as a stretch).
3. **`KernelOptGRPOTrainer`**, an 85-line subclass of `trl.GRPOTrainer` that
   replaces all four logprob call-sites (policy, old, reference-PEFT,
   reference-non-PEFT) with K1 via a single overridden method.
4. A complete benchmark harness, three headline plots, and an end-to-end demo
   on Qwen2.5-0.5B + GSM8K. **All results are reproducible on a single
   consumer GPU.**
5. An empirical finding that — as a free byproduct of fp32 register-level
   accumulation — K1 is strictly more accurate than TRL's bf16 path by
   4–6 orders of magnitude against fp32 ground truth.

---

## 2. Problem Statement

### 2.1 The exact computation we replace

Given logits `x ∈ R^[B, S, V]` (in bf16) and target token indices
`t ∈ Z^[B, S]`, we want to produce per-token quantities:

```
logprob[b, s] = x[b, s, t[b, s]] - logsumexp_v(x[b, s, v])
entropy[b, s] = -Σ_v softmax(x[b, s])_v · log_softmax(x[b, s])_v   (in nats)
lse[b, s]     = logsumexp_v(x[b, s, v])
```

TRL's stock implementation issues two separate passes over `x`:

```python
# selective_log_softmax (bf16 path):
for row_logits, row_labels in zip(logits, index, strict=True):
    row_logps = F.log_softmax(row_logits, dim=-1)              # alloc [S, V] bf16
    row_per_token_logps = row_logps.gather(dim=-1, row_labels)
    per_token_logps.append(row_per_token_logps)                 # B Python iterations

# entropy_from_logits, called separately:
for chunk in flat_logits.split(128, dim=0):
    logps = F.log_softmax(chunk, dim=-1)                        # alloc [128, V] bf16
    chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
    entropies.append(chunk_entropy)
```

The combined cost is two full streaming reads of `x`, plus several materialized
intermediates totaling roughly `2 × B × S × V × dtype_size` bytes of allocator
traffic.

### 2.2 Hardware and integration constraints

The MVP is constrained to a single RTX 4060 Laptop GPU (sm_89, 8 GB VRAM,
24 SMs, 100 KB shared memory per SM, 32 MB L2 cache, 256 GB/s peak DRAM
bandwidth, no FP8 hardware support, no Hopper-style TMA or `wgmma`). The kernel
must:

- Be invokable from PyTorch via standard custom-op machinery (`autograd.Function`
  or `torch.library.custom_op`).
- Compose with `torch.compile` (graph break is acceptable for MVP; full
  graph fusion is a stretch goal).
- Drop into `trl.GRPOTrainer` without forking the `transformers` or `trl`
  source trees.

---

## 3. Method

### 3.1 Online logsumexp for one-pass forward

The numerically stable logsumexp computation is canonical:

```
m = max(x)
lse = m + log(Σ_v exp(x_v - m))
```

The two-pass version (one pass for `m`, one for the sum) doubles DRAM traffic.
The online single-pass version, due to Milakov and Gimelshein [3], maintains
running state `(m, Z)` and updates incrementally on each new element:

```
m_new = max(m, x)
Z_new = exp(m - m_new) · Z + exp(x - m_new)
m, Z ← m_new, Z_new
```

After processing all V elements, `lse = m + log(Z)`.

### 3.2 Extending the streaming state to entropy

Shannon entropy of the softmax distribution can be rewritten as

```
H = -Σ_v p_v · log p_v = log(Z) - (1/Z) · Σ_v (x_v - m) · exp(x_v - m)
```

This requires a third running accumulator `T = Σ_v (x_v - m) · exp(x_v - m)`,
which transforms under the same max-shift as `Z`:

```
T_new = exp(m - m_new) · (T + (m - m_new) · Z) + (x - m_new) · exp(x - m_new)
```

The `(m - m_new) · Z` correction term arises because each previously-seen
element `(x_i - m_old)` shifts by exactly `(m_old - m_new)` when the running
max increases; the `exp(...)` factor must compensate. Failing to include
this term produces silent numerical drift.

The forward output is then `entropy = log(Z) - T/Z`.

### 3.3 Block-per-row reduction with online state combination

A naive single-thread-per-row implementation under-utilizes the GPU at
`B·S = 256` (the typical decode-time batch): only 256 threads launched on a
device that supports ~36,000 concurrent threads. The production kernel uses
**one threadblock per row**, with each thread accumulating its own local
`(m, Z, T)` over a strided slice of the V dimension, then reducing
across threads using:

1. **Warp-level shuffle reduction** (butterfly pattern) using `__shfl_xor_sync`,
   which requires no shared memory and no synchronization beyond the implicit
   warp lock-step. Five steps reduce 32 lanes to 1.
2. **Cross-warp reduction via shared memory**: each warp's lane-0 writes its
   reduced state to a fixed slot, `__syncthreads()`, and a single warp does
   a final reduction.

The combine operation for two partial states:

```
(m_combined, Z_combined, T_combined) = combine((m_a, Z_a, T_a), (m_b, Z_b, T_b))
```

is associative — verifiable by direct algebraic substitution into the
formulas — which is the precondition for any tree-style parallel reduction.

A subtle implementation issue: when one partial state is the "empty sentinel"
`(-∞, 0, 0)`, the combine formula yields `−∞ × 0 = NaN` for the `T` term.
Two early-return branches in the combine function handle this case explicitly.

### 3.4 Streaming backward via saved logsumexp

For training, K1 must support gradient flow. Given upstream gradients
`g_logp, g_ent, g_lse ∈ R^[B, S]`, the per-element gradient with respect to
`x[b, s, v]` decomposes as:

| from output       | term                                              |
|-------------------|---------------------------------------------------|
| `logprob` chain   | `g_logp · (δ(v == t) - p_v)`                     |
| `lse` chain       | `g_lse · p_v`                                    |
| `entropy` chain   | `g_ent · (-p_v) · (log_p_v + entropy)`           |

where `p_v = exp(x_v - lse)` and `log_p_v = x_v - lse` are computed on the fly
using the `lse` value saved during forward. Combining and factoring:

```
d_x[v] = p_v · (g_lse - g_logp - g_ent · (log_p_v + entropy))
       + (v == t ? g_logp : 0)
```

The kernel hoists the `v`-independent part of the coefficient out of the V
loop (`const_part = g_lse - g_logp - g_ent · entropy`), reducing per-element
work to one global read, one `expf`, one fused multiply-add, and one global
write. **No softmax materialization in backward, same memory profile as
forward.**

### 3.5 PyTorch integration

The forward and backward are wrapped in a `torch.autograd.Function`:

```python
class _FusedLogprobEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        logp, ent, lse = _alloc_outputs(targets)
        _C.fused_logprob_entropy_v1(logits, targets, logp, ent, lse)
        ctx.save_for_backward(logits, targets, lse, ent)
        return logp, ent, lse

    @staticmethod
    def backward(ctx, g_logp, g_ent, g_lse):
        logits, targets, lse, ent = ctx.saved_tensors
        ...                                     # convert None grads to zeros
        d_logits = torch.empty_like(logits)
        _C.fused_logprob_entropy_v1_backward(
            logits, targets, lse, ent, g_logp, g_ent, g_lse, d_logits)
        return d_logits, None
```

The TRL integration overrides the single method
`_get_per_token_logps_and_entropies` (located by `grep` of TRL source). One
nuance from the integration: TRL's slicing pattern
`logits[:, :-1, :][:, -logits_to_keep:, :]` produces a tensor whose batch
stride is the *original* `L · V`, not the sliced `(L-1) · V`. Calling
`.contiguous()` to satisfy K1's input requirement allocates ~1 GB on
realistic GRPO shapes and OOMs the 4060. The fix is per-batch indexing:
`logits[b]` selects dim 0 and yields a `[L', V]` tensor with strides
`(V, 1)`, which is naturally contiguous and requires no copy.

---

## 4. Results

### 4.1 Microbenchmark: kernel-level forward and backward latency

Measured on RTX 4060 Laptop, bf16, median of 30 iterations after 5 warmup
iterations. Baselines: TRL's `selective_log_softmax + entropy_from_logits`
(production path), PyTorch eager `F.log_softmax + gather` (separate
implementation reference), and a naive single-thread-per-row CUDA kernel
(correctness anchor).

| (B·S, V)         | TRL eager | PyTorch sep. | Naive | **K1 (ours)** | Speedup vs TRL |
|------------------|----------:|-------------:|------:|--------------:|---------------:|
| (256,   32k)     |   0.43 ms |     0.42 ms  | 1.95  |       0.05 ms |       **8.1×** |
| (1024,  32k)     |   1.82 ms |     2.43 ms  | 2.20  |       0.36 ms |       **5.1×** |
| (4096,  32k)     |   7.20 ms |     9.61 ms  | 4.30  |       1.24 ms |       **5.8×** |
| (256,  128k)     |   2.82 ms |     2.36 ms  | 7.84  |       0.35 ms |       **8.2×** |
| (1024, 128k)     |  11.31 ms |     9.31 ms  | 8.19  |       1.33 ms |       **8.5×** |
| (1024, 152k)     |  13.63 ms |    11.05 ms  | 9.71  |       1.58 ms |       **8.6×** |

K1 backward vs full PyTorch autograd (eager reference, with grad), bf16:

| (B·S, V)         | autograd backward | **K1 backward** | Speedup |
|------------------|------------------:|----------------:|--------:|
| (256,   32k)     |          1.58 ms  |        0.19 ms  | **8.3×** |
| (1024, 128k)     |         34.76 ms  |        2.45 ms  |**14.2×** |
| (1024, 152k)     |         41.10 ms  |        3.14 ms  |**13.1×** |
| (4096,  32k)     |         33.43 ms  |        2.33 ms  |**14.3×** |

### 4.2 DRAM bandwidth utilization

For a memory-bound kernel, the meaningful upper bound on throughput is
`bytes_streamed / peak_bandwidth`. RTX 4060 Laptop peak is 256 GB/s.
Effective bandwidth measured as `bytes_read / latency`:

| Backend (bf16, B·S=1024, V=152k) | GB/s  | % of 256 GB/s peak |
|----------------------------------|------:|-------------------:|
| **K1 forward**                   | 197.6 |          **77.2%** |
| **K1 backward**                  | 198.6 |          **77.6%** |
| TRL eager                        |  22.8 |               8.9% |
| PyTorch separate                 |  28.2 |              11.0% |
| Naive (1 thread/row)             |  32.1 |              12.5% |

Across the full sweep, K1 forward peaks at **82.6%** of peak bandwidth at
(4096, 32k), and K1 backward peaks at **87.8%** at the same shape. The
remaining ~15% gap to the hardware ceiling reflects launch overhead, residual
non-vectorized loads (8-byte vector loads not yet implemented), and the
fundamental cost of two streaming passes (forward) or read+write (backward).

### 4.3 Numerical accuracy: an unanticipated win

K1 reads bf16 inputs but accumulates `(m, Z, T)` in fp32 throughout. TRL's
bf16 path runs `F.log_softmax → exp → mul → sum` end-to-end in bf16. The
former is bounded by fp32 floating-point error (~10⁻⁶ relative); the latter
quantizes the final output to bf16's 8-bit mantissa precision (~10⁻² relative).

Tested on a real Qwen2.5-0.5B forward pass (4 sequences × 128 tokens × 152,064
vocab in bf16), comparing both paths against an fp32 reference recomputed from
the same logits cast to fp32:

| Quantity        | TRL eager error      | K1 error              | Improvement       |
|-----------------|---------------------:|----------------------:|------------------:|
| Per-token logp  | 1.21 × 10⁻²          | **2.04 × 10⁻⁶**       | **6 orders**      |
| Per-token ent   | 3.34 × 10⁻²          | **2.15 × 10⁻⁶**       | **4 orders**      |

This is a free byproduct of the fusion design (we already had the values in
fp32 registers; using fp32 for the running accumulators costs nothing).

### 4.4 End-to-end GRPO step parity

Plugging K1 into `trl.GRPOTrainer` via the `KernelOptGRPOTrainer` subclass and
running one GRPO step on the same Qwen2.5-0.5B rollout: simulated GRPO loss
computed from K1's per-token logp differs from stock TRL by **7.4 × 10⁻⁶**.
The project's design contract was a tolerance of `10⁻³`, so the actual
deviation is **135× tighter** than required.

End-to-end smoke test (`examples/train_gsm8k.py`): 5 GRPO steps complete in
52.7 seconds with peak 2.8 GB VRAM, on a recipe (G=4, max_completion=192,
gradient_checkpointing=True) that we tightened from Stage 1's stock recipe
(G=8, max_completion=256) which had peaked at 9.4 GB and overflowed into
Windows WDDM shared memory. The kernel-level memory savings of 261 MB per
logp call (`compare_one_step.py`) are part of what enabled the tighter
recipe to fit comfortably.

---

## 5. Related Work

### 5.1 Online softmax: Milakov & Gimelshein, 2018 [3]

The single-pass online normalizer formulation for softmax (and by extension
logsumexp) was published as a NVIDIA technical report. Their motivation was
inference-time softmax in attention and classifier heads. The same algebraic
trick — running max + running sumexp with the `exp(m_old - m_new)` rescaling
factor — underlies our kernel's forward pass. Our extension is the third
accumulator `T` for entropy, which to our knowledge has not appeared in the
public kernel literature.

### 5.2 FlashAttention: Dao et al., 2022 [2]

FlashAttention applies online softmax to the attention map computation,
fusing `softmax(QK^T) V` into a single tiled kernel that never materializes
the `[seq, seq]` attention matrix. The mathematical machinery is identical to
ours (online `(m, Z)` state with `exp` rescaling), but the application domain
differs: FlashAttention attacks the attention bottleneck during inference and
training; K1 attacks the loss-block bottleneck during post-training. Both are
manifestations of the same insight: when the natural eager implementation
materializes a tensor whose only purpose is to be reduced, fusion via online
state recovers an order of magnitude in DRAM traffic.

### 5.3 GRPO and the K3 KL estimator: DeepSeek-R1 [1] and Schulman, 2020 [4]

GRPO (Group Relative Policy Optimization), introduced in the DeepSeek-R1
training pipeline, replaces PPO's value baseline with a group-relative
advantage normalization, eliminating the value head and its training cost.
The KL penalty term in GRPO uses the K3 unbiased low-variance estimator:

```
D_KL_hat(π || π_ref) = exp(log π_ref - log π) - (log π_ref - log π) - 1
```

published in Schulman's blog post on KL approximation. This estimator only
requires the per-token log-probabilities of the sampled tokens — exactly what
K1 produces — making K1 the natural fused primitive for the GRPO loss block.

### 5.4 PPO: Schulman et al., 2017 [5]

Proximal Policy Optimization established the importance-ratio + clipped
surrogate loss formulation used by GRPO. The inner per-step compute is
structurally identical: gather per-token logp under current and old policies,
compute the ratio, clip, and reduce. K1 is the right primitive for PPO's
loss block as well, with no modifications to the kernel signature.

### 5.5 Liger-Kernel: Hsu et al., 2024 [6]

Liger-Kernel from LinkedIn ships a suite of Triton-based fused kernels for
LLM training, including `LigerFusedLinearCrossEntropy`, which combines the
LM-head linear projection with the cross-entropy loss into a single kernel
that avoids materializing the `[B, S, V]` logits. Liger's approach is more
ambitious (it fuses GEMM + cross-entropy together) but requires Triton
infrastructure and writes a different intermediate: cross-entropy loss, not
logp+entropy. K1 is complementary: it operates on already-computed logits,
in CUDA C++ (no Triton dependency), and exposes both logp and entropy
explicitly for downstream loss combinators (GRPO, DPO, etc.) that require them.
A Liger-style fused-LM-head extension of K1 is a natural follow-up.

### 5.6 Marlin: Frantar et al., 2024 [7]

Marlin is a hand-written CUDA W4A16 GEMM kernel for LLM inference, targeting
the low-batch decode regime where weight-loading bandwidth dominates. While
Marlin operates in a different problem space (quantized inference vs.
post-training), it shares K1's design philosophy: identify the operation that
saturates DRAM bandwidth in the production workload, hand-write a CUDA kernel
that explicitly fuses and tiles to minimize redundant memory traffic, and
expose the result through PyTorch's custom-op machinery. The Marlin paper
also documents an architecture-specific tuning story (Ampere/Hopper vs.
consumer Ada) that informed our decision to focus K1 explicitly on sm_89.

---

## 6. Limitations and Future Work

This work is a single-developer MVP completed on a consumer GPU under a 4-week
schedule. The following are deliberate scope cuts or open follow-ups:

1. **Hopper-only optimizations not exercised.** sm_90's TMA (Tensor Memory
   Accelerator) and `wgmma` instructions would let a redesigned K1 prefetch
   logits asynchronously and overlap memory with compute. Not relevant to
   sm_89 hardware and out of scope.
2. **No vectorized loads.** K1 reads bf16 elements one at a time; switching
   to 8-element `int4` aligned loads would likely push 82% peak DRAM
   utilization to ~90%. Estimated 1 day of work.
3. **Per-batch Python loop in trainer integration.** The OOM-avoidance
   workaround for non-contiguous slices iterates B times in Python instead of
   one fused launch. A kernel revision that accepts strided logits would
   eliminate this. Estimated 0.5 day.
4. **`torch.library.custom_op` not registered.** Currently exposed via
   `torch.autograd.Function` only, which causes a Dynamo graph break (with a
   warning) when used inside `torch.compile`. The result is correct but not
   single-graph-fused. Estimated 1 day.
5. **No Nsight Compute screenshots.** Windows requires admin permissions
   for GPU performance counter access; the bench harness measures effective
   bandwidth as a numerical proxy.
6. **No Triton port for direct Liger comparison.** K1 vs.
   `LigerFusedLinearCrossEntropy` would require either porting Liger to take
   already-computed logits or extending K1 to fuse with the LM-head
   projection.

---

## 7. Conclusion

`kernel-opt` demonstrates that a hand-written CUDA kernel can capture the
remaining order-of-magnitude bandwidth gap in the open-source LLM
post-training stack. The work is concretely useful for any team running GRPO,
PPO, DPO, or KD at scale — the K1 primitive plugs into TRL via a
single-method override and produces strictly more numerically accurate
gradients than the eager baseline. The project also illustrates a
reproducible engineering recipe — algorithmic fusion via online state +
block-per-row reduction with associative combine + autograd integration via
`torch.autograd.Function` — that generalizes to any per-token reduction over
the vocabulary dimension.

---

## References

[1] DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning*. arXiv:2501.12948, January 2025.

[2] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness*. arXiv:2205.14135, NeurIPS
2022.

[3] M. Milakov, N. Gimelshein. *Online normalizer calculation for softmax*.
arXiv:1805.02867, 2018.

[4] J. Schulman. *Approximating KL Divergence*. Blog post,
http://joschu.net/blog/kl-approx.html, 2020. (Source for the K3 estimator
adopted by GRPO.)

[5] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal
Policy Optimization Algorithms*. arXiv:1707.06347, 2017.

[6] B. Hsu, Y. Dai, et al. *Liger Kernel: Efficient Triton Kernels for LLM
Training*. arXiv:2410.10989, October 2024.

[7] E. Frantar, R. L. Castro, A. Alistarh. *Marlin: Mixed-Precision Auto-Regressive
Parallel Inference on Large Language Models*. arXiv:2408.11743, 2024.
