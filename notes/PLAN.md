# GRPO Kernel Pack — `kernel-opt`

**Target role**: NA AI infra (post-training / inference / framework teams).
NeMo-Aligner, HuggingFace TRL, ByteDance verl, OpenRLHF, vLLM, SGLang,
xAI / Anthropic / OpenAI post-training infra, Meta GenAI infra, torchao.

**Hardware**: RTX 4060 Laptop, sm_89 (Ada Lovelace), 8 GB VRAM. Single GPU.
**Effort**: ~18-25 working days end-to-end (see Stages). MVP-shaped.

---

## Context

GRPO (DeepSeek-R1's algorithm) is the dominant 2025-26 LLM post-training recipe.
Open-source frameworks (TRL, verl, OpenRLHF) implement the inner loss as eager
PyTorch ops over the `[batch, seq, vocab]` logits tensor — full-tensor reductions
with materialized softmax intermediates. Real, measurable bandwidth left on the
table.

The project ships **one hero CUDA kernel** plus a clean PyTorch op contract and a
one-class `trl.GRPOTrainer` subclass. The framing is **AI infrastructure**, not
"I made GRPO faster": memory-bound op fusion, profiling-driven, scaling-
characterized, `torch.compile`-compatible, with a generalization story (the same
kernel is the right primitive for SFT eval, DPO, KD, and speculative-decode
verification).

## What this project demonstrates (AI-infra interview rubric mapping)

| Hiring signal | Artifact |
|---|---|
| Profiling discipline | ncu roofline + bandwidth % for K1; per-op latency breakdown of the loss step |
| Memory-system thinking | peak-VRAM measurement; "fits 2× batch" framing |
| Framework integration | `torch.library.custom_op` + `autograd.Function` + `meta` kernel + `torch.compile` round-trip test |
| Scaling characterization | sweep over (vocab, batch×seq); regime analysis; extrapolation to Llama-3 / Gemma-2 / Qwen3 vocab sizes |

## Elevator pitch (resume bullet)

> Wrote `kernel-opt`, a fused logprob+entropy CUDA kernel for LLM post-training.
> Single streaming pass over `[B,S,V]` logits via online logsumexp; no softmax
> materialization. Drop-in via a 50-line `trl.GRPOTrainer` subclass; same
> primitive reusable for SFT eval, DPO, and KD. On RTX 4060: forward at [X]%
> peak DRAM bandwidth, [Y]× lower peak VRAM than HF TRL's eager path, [Z]%
> lower per-step latency on Qwen2.5-0.5B GRPO. `torch.compile`-compatible.
> Profiling via Nsight Compute (roofline + scaling sweep in `REPORT.md`).

## Scope

**In (MVP)**:
- One CUDA kernel: `fused_logprob_entropy` (K1) — forward + streaming backward
- `torch.library.custom_op` + `autograd.Function` + `meta` kernel
- `KernelOptGRPOTrainer(trl.GRPOTrainer)` subclass, **single-step loss matches eager** within tolerance
- Microbench sweep (vocab × B×S); per-op breakdown chart of one GRPO loss step; peak-VRAM comparison
- ncu reports for K1 forward + backward at the sweet-spot shape; roofline scatter
- `torch.compile` round-trip test
- README (pitch + 3 plots + quickstart) + REPORT (~3 pages: design, scaling, ncu, postmortem)
- README snippet showing K1 reused for DPO loss in <15 lines (claim of generalization)

**Stretch (only if MVP ships early)**:
- K2 `fused_grpo_loss` kernel (fuses 6 elementwise ops on `[B,S]`; small absolute win)
- 20-step E2E run with full loss-curve overlap plot
- Standalone DPO benchmark (`bench_dpo.py`) with numbers, not just a snippet
- CUDA Graph capture test
- Warp-per-row K1 variant for small V

**Out (locked, do not re-litigate)**:
- No autotuner, no Triton variant, no multi-GPU, no reward model, no PPO/GAE
- No fused LM-head matmul (would require Tensor Cores)
- No K3 group-advantage-norm kernel (32 floats per step; PyTorch is plenty fast)
- No CMake (setuptools only)
- No "RL convergence" claim (numerical-match only)

---

## The kernel — K1 `fused_logprob_entropy`

**Signature**:
```
fused_logprob_entropy(
    logits: [B, S, V] bf16/fp16,
    target_tokens: [B, S] int64,
    return_entropy: bool = True,
) -> (logprob: [B, S] fp32, entropy: [B, S] fp32, lse: [B, S] fp32)
```

**Forward**: single streaming pass over V per (b,s) row. Online logsumexp
(running max + running sumexp), entropy via one extra accumulator (essentially
free). Vectorized 8-element bf16 loads. **Block-per-row implementation** (one
threadblock per (b,s)); scales across V from 32k to 256k+ without code branches.
Never materializes `[B,S,V]` softmax.

**Backward (MVP)**: stores `lse` from forward, second streaming pass recomputes
`p = exp(logit - lse)` on the fly, applies grad: `d_logits = (p - onehot(t)) * d_logp`.
Same memory profile as forward.

**Backward (cut-line)**: if streaming backward drift > 1e-3 vs autograd, fall
back to materializing softmax in backward (calls existing PyTorch path). Loses
backward memory win; keeps forward win, which is the big one. Document the
trade-off in REPORT.

**Wider applicability** (the AI-infra angle): same kernel signature is the right
primitive for SFT validation logprob, DPO/IPO/SimPO loss (called twice), KD
(student/teacher logprob), speculative-decode verification, beam scoring. README
shows a <15-line DPO loss using K1 unchanged.

---

## End-to-end demo

- **Model**: Qwen2.5-0.5B-Instruct, LoRA rank 16 on attn projections
- **Reference**: Qwen2.5-0.5B-Instruct frozen, bf16, no grad
- **Optimizer**: 8-bit AdamW (`bitsandbytes`) on LoRA params only
- **Dataset**: GSM8K train, small subset
- **Reward**: rule-based (answer-match + format), no reward model
- **MVP contract**: subclass loads, one GRPO step runs end-to-end, loss value
  matches stock TRL within 1e-3 absolute. **Multi-step run is stretch.**
- **Fallback**: if 8 GB OOMs, drop to TinyLlama-1.1B or synthetic-target step

---

## Repo layout (lean)

```
D:\Projects\kernel-opt\
  README.md              # pitch + 3 plots + DPO snippet + quickstart
  REPORT.md              # ~3 pages: design, scaling, ncu, postmortem
  PLAN.md                # this file
  pyproject.toml
  setup.py               # CUDAExtension, sm_89 only

  csrc/
    fused_logprob.cu     # K1 forward + backward
    bindings.cpp         # torch.library + autograd.Function glue

  python/kernel_opt/
    __init__.py
    ops.py               # autograd.Function wrapper; meta kernel
    trainer.py           # KernelOptGRPOTrainer(trl.GRPOTrainer)

  tests/
    test_logprob.py      # forward + gradcheck vs eager
    test_compile.py      # torch.compile round-trip
    test_trainer_step.py # single GRPO step matches eager

  bench/
    bench_micro.py       # K1 sweep (vocab × B×S × dtype) → CSV
    bench_e2e.py         # GRPO step latency + per-op breakdown + peak VRAM
    plots/
    results/
    ncu/

  examples/
    train_gsm8k.py       # the smoke-test demo
```

**Build**: setuptools + `torch.utils.cpp_extension.CUDAExtension`, sm_89 only.
Pin: `torch>=2.4`, `trl` to whatever's current at Stage 1.

---

## Stages

Each stage has an exit criterion. Move on only when met. Day estimates are
working-day counts.

### Stage 1 — Foundation (3-4 days)

- Verify CUDA 12.6 + `torch.utils.cpp_extension` builds a hello-world kernel on Windows MSVC. **If >2 hrs of fight, switch to WSL2 Ubuntu.**
- Pin and install: `transformers trl peft bitsandbytes datasets accelerate`. (Windows + bitsandbytes can be flaky; verify GPU path works.)
- **Read TRL's `grpo_trainer.py` source.** Locate the per-token logprob computation. Confirm whether it materializes `log_softmax`. **If it already uses `F.cross_entropy`, K1's headline number drops to ~1.1-1.4× from bf16 vec + entropy fusion (entropy is the differentiator since `cross_entropy` doesn't return it). Update the README pitch numbers honestly.** This check is non-optional.
- Run stock `trl.GRPOTrainer` for 5 steps on Qwen2.5-0.5B + GSM8K to confirm the recipe fits 8 GB. If OOM, drop LoRA rank or G/seq, lock the recipe.

**Exit**: hello-world kernel builds; 5-step stock GRPO completes; TRL logprob path read; framing locked.

### Stage 2 — K1 forward + bench harness (5-7 days)

- Naive correct version: one thread per row, scalar V loop. Validate vs `F.log_softmax + gather` to <1e-5.
- Build `bench/bench_micro.py` against the naive version. **Every later perf change is measured by it.**
- Optimize: vectorized 8-element bf16 loads, block-per-row, online logsumexp + entropy in one accumulator. Run `ncu` after each change; record bandwidth %.
- Add scaling sweep: vocab ∈ {32k, 128k, 152k, 256k}, B*S ∈ {256, 1024, 4096, 8192}.

**Exit**: K1 forward correct; ≥80% peak DRAM bw on at least one shape; full forward sweep CSV in `bench/results/`.

### Stage 3 — K1 backward + autograd (2-4 days)

- K1 backward: streaming version first; if drift > 1e-3 vs autograd, fall back to materializing variant.
- `autograd.Function` wrapper. `torch.library.custom_op` registration with `meta` kernel.
- `gradcheck` on small shapes.
- `torch.compile` round-trip test.

**Exit**: `pytest tests/test_logprob.py tests/test_compile.py` passes.

### Stage 4 — TRL integration (3-4 days)

- `KernelOptGRPOTrainer(trl.GRPOTrainer)` subclass. Override the single method containing the logprob block (located in Stage 1).
- `test_trainer_step.py`: one GRPO step, custom vs stock, loss value matches within 1e-3.
- `examples/train_gsm8k.py` runs the demo end-to-end (single step or short loop).

**Exit**: 1-step training matches stock TRL.

### Stage 5 — Benchmark + scaling study (3-5 days)

- Full microbench sweep complete, plotted.
- Per-op latency breakdown (`bench_e2e.py`): stock TRL loss step decomposed into kernel-time bars; ours decomposed; side-by-side. **This is the headline plot.**
- Peak VRAM measurement, stock vs ours, with batch-doubling demo.
- ncu reports for K1 forward + backward at the sweet-spot shape; roofline scatter.
- Scaling extrapolation: project K1's win at Llama-3 (V=128k), Gemma-2 (V=256k), Qwen3 (V=152k).

**Exit**: 3 plots in `bench/plots/`; CSVs in `bench/results/`; ncu reports in `bench/ncu/`.

### Stage 6 — Writeup + polish (3-4 days)

- README: pitch, 3 plots inline, DPO snippet, quickstart that runs in <5 minutes.
- REPORT: ~3 pages — design decisions, scaling-study analysis, ncu screenshots, "what didn't work" honesty section.
- Code cleanup, docstrings on the 5 files anyone would read.
- Publish to GitHub; consider a small upstream PR (TRL benchmark or Liger-Kernel comparison port).

**Exit**: repo is shippable as portfolio.

---

## Report contract (what `REPORT.md` must contain)

What hiring managers actually read. Specced explicitly so the writeup isn't an
afterthought. Each item can be terse — REPORT is ~3 pages, not a thesis.

1. **One-paragraph framing**: why this op is the bottleneck; fused vs eager in 5 lines of code.
2. **Three headline numbers** repeated from README with full context (shapes, dtype, baseline).
3. **Scaling study**: vocab sweep + B×S sweep plots. **Identify the regime where the win is largest and explain why** (memory-bound, vocab dominates DRAM traffic).
4. **Roofline scatter** for K1 forward across all sweep shapes; annotate where bandwidth limit binds.
5. **Per-op breakdown stack chart** of one GRPO loss step, stock vs ours.
6. **Numerical-correctness table**: max abs error, gradcheck pass/fail, 1-step loss match.
7. **`torch.compile` compatibility status.**
8. **One "what didn't work" subsection.** Honesty here is a hiring signal — fake-perfect projects look fake.
9. **DPO snippet** showing K1 reused unchanged for a different loss (~10 lines of code, no separate benchmark).

---

## Risks (top 3)

| # | Risk | Mitigation |
|---|---|---|
| 1 | **TRL already uses `F.cross_entropy`** for logprob → K1 vs naive baseline is a strawman | Stage 1 read confirms; if so, headline number drops but K1 still wins via bf16 vectorization + free entropy + lse return; update README numbers honestly |
| 2 | **Windows MSVC + nvcc + cpp_extension build pain** | Stage 1 hello-world; switch to WSL2 if >2 hrs |
| 3 | **8 GB OOM on Qwen2.5-0.5B + LoRA + GRPO + ref model + KV cache** | Pre-decided fallback to TinyLlama-1.1B or synthetic-target step |

---

## Job targeting

**Direct fit**: NVIDIA NeMo-Aligner, HuggingFace TRL, ByteDance verl, OpenRLHF,
xAI/Anthropic/OpenAI post-training infra, Allen AI OLMo, Meta GenAI alignment.

**Adjacent fit**: vLLM / SGLang / TRT-LLM (kernel discipline, profiling, op-
contract skills transfer 1:1 to inference), torchao, Liger-Kernel, Unsloth.

**Tactical multiplier**: ship one upstream PR after the repo is public — a TRL
benchmark of fused logprob, or a Liger-Kernel port of K1 to Triton. A merged
50-line upstream PR multiplies repo credibility ~10×.

**Interview answers this project unlocks**:
- *"Tell me about a CUDA kernel you wrote."* → "Fused logprob + entropy for LLM post-training. The eager path materializes a `[B,S,V]` fp32 softmax intermediate then throws most of it away. Online logsumexp in one streaming pass kills the intermediate. Forward hits [X]% peak DRAM bw on 4060."
- *"How does this scale?"* → "Win grows with vocab. Bandwidth-bound at every shape; arithmetic intensity is ~constant. Project to ~Y× at Gemma's V=256k."
- *"How would you integrate a custom kernel into a training framework?"* → "torch.library.custom_op for the contract, autograd.Function for grad, meta kernel for shape inference so torch.compile composes. TRL integration is a 50-line subclass, no fork."
- *"Memory-bound vs compute-bound — how do you tell?"* → ncu roofline + arithmetic-intensity calculation. Show the chart from REPORT §4.

---

## Verification

1. `pytest tests/` — kernel forward/backward, gradcheck, compile round-trip, 1-step trainer match.
2. `python bench/bench_micro.py --sweep` → CSV in `bench/results/`.
3. `python bench/bench_e2e.py` → step-latency table + per-op breakdown plot + peak-VRAM table.
4. `python examples/train_gsm8k.py` → one GRPO step runs end-to-end.
5. `ncu --set full -o bench/ncu/k1_fwd python -c "..."` → roofline plot generation.

---

## Critical files (the four most load-bearing)

- `csrc/fused_logprob.cu` — the kernel, forward + backward
- `csrc/bindings.cpp` — `torch.library` + `autograd.Function` glue, `meta` kernel
- `python/kernel_opt/trainer.py` — TRL subclass; the integration story
- `bench/bench_e2e.py` — produces the headline per-op breakdown plot; if this plot is weak the project is weak
