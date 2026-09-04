# kernel-opt

**Fused logprob + entropy CUDA kernel for LLM post-training.** Drop-in
replacement for HuggingFace TRL's `selective_log_softmax` + `entropy_from_logits`.
Same primitive serves SFT eval, DPO, knowledge distillation, speculative-decode
verification — anywhere you need per-token logprob from `[B, S, V]` logits.

> One streaming pass over the logits tensor via online logsumexp. No softmax
> materialization. Forward + backward + autograd integration + a 50-line
> `trl.GRPOTrainer` subclass.

## Headline numbers

Measured on RTX 4060 Laptop (sm_89, 8 GB, 256 GB/s peak DRAM), bf16.
Bandwidth, traffic, occupancy and launch counts are **Nsight Compute measured**
(`bench/ncu/`); latencies are median-of-30 from `bench/bench_micro.py`.

| | K1 (ours) | baseline | factor |
|---|---:|---:|---:|
| **Forward latency** (B·S=1024, V=152k) | 1.58 ms | 13.6 ms (TRL eager) | **8.6×** |
| **Backward latency** (B·S=1024, V=152k) | 3.14 ms | 41.1 ms (PyTorch autograd) | **13.1×** |
| **DRAM traffic** vs theoretical minimum | **1.00×** | 8.36× (TRL eager) | **8.4× less traffic** |
| **Kernel launches** per call | **1** | 43 (TRL eager) | **43× fewer** |
| **Peak DRAM bandwidth** (ncu) | 68.8–91.4% | — | — |
| **Occupancy** (ncu) | 96–98% | 8.3% (naive 1-thread/row) | — |
| **Intermediate alloc per call** | **0 MB** | 298 MB (TRL eager) | — |
| **Error vs fp32 ground truth** (logprob) | 2.0e-06 | 1.2e-02 (TRL bf16) | **6 orders of magnitude** |

GRPO integration: simulated loss differs by **7.4e-06** from stock TRL — 135× tighter than the 1e-3 contract.

> **The headline isn't "a faster kernel."** Nsight Compute shows TRL's own kernels
> hit 88.2% of peak DRAM bandwidth — they're efficient. TRL is slow because it
> streams the logits tensor **~8 times across 43 kernel launches**. K1 does it
> once. See [REPORT.md §4.3](REPORT.md).

## Plots

![Traffic amplification and per-kernel bandwidth efficiency](bench/plots/traffic_amplification.png)

![Roofline — all kernels ~8x left of the ridge](bench/plots/roofline.png)

![K1 speedup vs TRL eager across (vocab × batch×seq)](bench/plots/speedup.png)

![Per-call intermediate allocation — K1 streams in fp32 registers](bench/plots/memory.png)

## Quickstart

### Install (Windows native, with the project's pinned dev env)

```powershell
# Activates MSVC 14.29 toolset (CUDA 12.6 compatible), DISTUTILS_USE_SDK=1, venv on PATH
cmd /c "scripts\dev_env.bat && uv pip install --no-build-isolation -e ."
```

Pinned to: `torch>=2.4` (tested 2.6.0+cu124), `trl==1.4.*`, CUDA toolkit 12.6, sm_89.
Linux users: drop the `dev_env.bat` wrapper and `pip install -e .` directly.

### Use as a drop-in op

```python
import torch
from kernel_opt import fused_logprob_entropy

logits = torch.randn(2, 128, 152064, device="cuda", dtype=torch.bfloat16, requires_grad=True)
targets = torch.randint(0, 152064, (2, 128), device="cuda", dtype=torch.int64)

logp, entropy, lse = fused_logprob_entropy(logits, targets)
# logp[b, s] = log_softmax(logits[b, s])[targets[b, s]]
# entropy[b, s] = Shannon entropy of softmax(logits[b, s]) in nats
# lse[b, s] = logsumexp(logits[b, s])  -- saved for backward, also useful externally

loss = -(logp + 0.01 * entropy).mean()
loss.backward()              # K1 backward, single streaming pass
print(logits.grad.shape)     # torch.Size([2, 128, 152064])
```

### Use in TRL GRPO (one-line swap)

```python
from kernel_opt import KernelOptGRPOTrainer  # was: from trl import GRPOTrainer
trainer = KernelOptGRPOTrainer(model=model, args=cfg, train_dataset=ds,
                               reward_funcs=[...], peft_config=lora)
trainer.train()
```

The subclass overrides exactly one method (`_get_per_token_logps_and_entropies`)
and that method is the hub for **all four** logprob call sites in TRL's GRPO
(policy / old / ref-PEFT / ref-non-PEFT). One override = full coverage.
Multimodal training paths fall back to the parent eager implementation.

End-to-end demo (5 steps of GRPO on Qwen2.5-0.5B + LoRA + GSM8K):
```powershell
cmd /c "scripts\dev_env.bat && python examples\train_gsm8k.py"
```
Completes in ~53s on a 4060. Peak VRAM 2.8 GB. Loss curve and reward are sane.

## Wider applicability — DPO loss in 12 lines

K1 is the right primitive for any "per-token logprob + entropy from logits"
computation. DPO using K1 unchanged:

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

Same kernel signature works for SFT eval, DPO/IPO/SimPO, KD, speculative-decode
verification, beam scoring. The K1 computation is genuinely a primitive, not a
GRPO-specific optimization.

## Project status

What ships (MVP, complete):
- K1 forward + backward, three dtypes (bf16/fp16/fp32)
- `torch.autograd.Function` wrapper; `torch.compile(backend="aot_eager")` round-trip works
- `KernelOptGRPOTrainer(trl.GRPOTrainer)` subclass
- 35 tests pass (`pytest tests/`)
- Benchmark harness + 5 plots, including an ncu-measured roofline
- Nsight Compute profiling pipeline (`run_ncu.ps1` → `ncu_parse.py` → `plot_roofline.py`)
- `examples/train_gsm8k.py` end-to-end demo
- `examples/compare_one_step.py` 1-step parity check

Out of scope (locked, see `PLAN.md`):
- Tensor Cores / fused LM-head matmul (would require Hopper-style mma; not on
  4060's MVP-budget for a single-developer 4-week project)
- Multi-GPU
- Triton port

Stretch / open follow-ups (see `notes/stage*_findings.md` for details):
- Register K1 as a `torch.library.custom_op` with a `meta` kernel — eliminates
  the Dynamo graph-break (currently graph-breaks at the pybind boundary;
  result is correct, just not single-graph-fused)
- Kernel stride support — would let `KernelOptGRPOTrainer` do one fused launch
  per call instead of B per-batch launches (limit currently is the OOM-avoidance
  workaround for non-contiguous slices in TRL's loss block; see `notes/stage4_findings.md`)
- Vectorized 8-element bf16 loads — ncu shows forward at 68.8–73.4% of peak vs
  backward's 90.5–91.4%; the forward's warp/block reduction is the gap, and
  vectorized loads are the obvious next lever

## Layout

```
kernel-opt/
  PLAN.md                       # the original spec
  REPORT.md                     # technical writeup (3 pages)
  notes/                        # per-stage findings + retrospective
    retrospective_stage1_2.md
    stage{1,2,3,4,5}_findings.md
  csrc/
    fused_logprob.cu            # naive + v1 forward + v1 backward kernels
    bindings.cpp
  python/kernel_opt/
    ops.py                      # autograd.Function + 3 public entry points
    trainer.py                  # KernelOptGRPOTrainer
  tests/
    test_logprob.py             # 32 tests (correctness + gradcheck)
    test_compile.py             # 3 tests (torch.compile round-trip)
  bench/
    bench_micro.py              # kernel-level sweep
    bench_backward.py           # backward-only sweep
    bench_step.py               # full GRPO step latency + VRAM
    plot_results.py             # speedup / bandwidth / memory PNGs
    ncu_targets.py              # one-backend-per-process ncu profiling targets
    run_ncu.ps1                 # batch ncu driver (8-report focused matrix)
    ncu_parse.py                # ncu CSV -> tidy table (+ model validation)
    plot_roofline.py            # roofline + traffic-amplification PNGs
    plots/, results/, ncu/
  examples/
    train_gsm8k.py              # E2E demo with our trainer
    compare_one_step.py         # 1-step parity check vs stock TRL
  scripts/
    dev_env.bat                 # Windows MSVC 14.29 + venv activation
  setup.py, pyproject.toml
```

## Acknowledgments / references

- HuggingFace [TRL](https://github.com/huggingface/trl) — GRPO trainer this kernel plugs into
- The "online logsumexp" trick, used identically in [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao 2022) for the softmax normalization step
- John Schulman's K3 KL estimator (`(q/p - 1) - log(q/p)`), the GRPO KL term
- DeepSeek-R1 paper for the GRPO recipe

---
**License**: MIT. **Author**: Kaiwen Lin (kaiwenlin@utexas.edu). Built on RTX 4060 Laptop.
