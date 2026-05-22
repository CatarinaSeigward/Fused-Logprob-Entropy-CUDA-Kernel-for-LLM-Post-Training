# Stage 1 findings

Versions pinned at start of Stage 1 (2026-05-14):
- torch 2.6.0+cu124
- transformers 5.8.1
- trl 1.4.0
- peft 0.19.1
- numpy 2.4.4, ninja 1.13.0, pytest 9.0.3
- CUDA toolkit 12.6, MSVC 14.29.30133 (VS 2019 toolset, CUDA 12.6 compatible)

## Toolchain

Windows native build works. **No need to switch to WSL2.** The recipe:

- VS 2022 Community at `D:\Program Files\Microsoft Visual Studio\2022\Community\` is an empty shell (no `VC\Tools\MSVC\` directory) — `cpp_extension`'s autodetect skips it.
- A separate "Visual Studio 18 BuildTools" at `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\` has 4 MSVC toolsets (14.16, 14.29, 14.44, 14.50). `cpp_extension` autoselects the newest (14.50) which CUDA 12.6 explicitly rejects ("only VS 2017–2022 supported").
- Fix: explicitly select MSVC 14.29 via `vcvars64.bat -vcvars_ver=14.29`, and set `DISTUTILS_USE_SDK=1` so `cpp_extension` trusts the active VC env. Captured in `scripts/dev_env.bat`.
- From PowerShell: `cmd /c "scripts\dev_env.bat && <command>"`.

Hello-world CUDA kernel built and ran in 38s; both tests pass on the 4060 (sm_89, bf16 supported).

## TRL logprob + entropy path (the K1 baseline)

Confirmed in `trl/trainer/utils.py:436` (`selective_log_softmax`) and `trl/trainer/utils.py:483` (`entropy_from_logits`), called from `trl/trainer/grpo_trainer.py:1106-1124`.

### What TRL actually does per loss step

Once `logits = model(...).logits` produces a `[B, L-1, V]` tensor (after slicing):

1. `logits.div_(self.temperature)` (in-place, fine).
2. `logps = selective_log_softmax(logits, completion_ids)` — gathers per-token logprobs.
3. If `compute_entropy=True` (default for the policy model in GRPO), `entropies = entropy_from_logits(logits)` — a **separate pass** over the same logits.

### Why this is the right kernel to fuse

`selective_log_softmax` for **bf16/fp16** (the GRPO production path):

```python
# logits.dtype == torch.bfloat16
per_token_logps = []
for row_logits, row_labels in zip(logits, index, strict=True):  # Python loop over batch!
    row_logps = F.log_softmax(row_logits, dim=-1)               # materializes [L-1, V]
    row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
    per_token_logps.append(row_per_token_logps)
per_token_logps = torch.stack(per_token_logps)
```

The TRL comment explicitly says: *"logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach"*. So the bf16 production path is:
- A **Python for loop over batch dim** (kernel-launch overhead per row)
- Materializes a full `[L-1, V]` log_softmax tensor per row
- The total work is dominated by reads/writes of the `[B, L-1, V]` logits and intermediates

`entropy_from_logits` is even more wasteful — it always materializes both softmax and log_softmax in chunks of 128 rows:

```python
for chunk in flat_logits.split(128, dim=0):
    logps = F.log_softmax(chunk, dim=-1)            # alloc [128, V] fp32
    chunk_entropy = -(torch.exp(logps) * logps).sum(-1)  # second pass
    entropies.append(chunk_entropy)
```

For Qwen2.5 with V=152064: each chunk ≈ 128 × 152064 × 4 B = 78 MB of fp32 intermediate, allocated and freed per chunk.

### What K1 will replace

Both `selective_log_softmax(logits, ids)` and `entropy_from_logits(logits)` collapse into **one streaming pass** over `logits[B, L-1, V]`:

```
fused_logprob_entropy(logits, target_tokens) -> (logprob[B, L-1], entropy[B, L-1], lse[B, L-1])
```

Wins over TRL's bf16 path:
1. One streaming pass instead of two (logprob loop + entropy loop)
2. No Python-loop kernel-launch overhead
3. No intermediate `[L-1, V]` softmax allocations
4. **bf16 stability via fp32 register-level accumulation** of running max + running sumexp + running entropy accumulator (the same trick `F.cross_entropy` uses internally)

### Integration entry point

In `KernelOptGRPOTrainer(GRPOTrainer)`, override `_get_per_token_logps_and_entropies` (lines 1046–1125) to call our fused op. Single method override; no fork.

### Risk previously flagged ("TRL might already use F.cross_entropy")

Resolved. TRL does NOT use `F.cross_entropy`. The K1 baseline is a real, measurable win and the headline number should be substantial (especially for bf16 + entropy).

## TRL 1.4 API notes

- `GRPOConfig.max_prompt_length` was removed in TRL 1.4. Bound at tokenizer / dataset level instead. Only `max_completion_length` survives.
- `per_device_train_batch_size` counts SEQUENCES (generations), not prompts. Must be a multiple of `num_generations`. For `num_generations=G` and 1 prompt/step: set `per_device_train_batch_size=G`.
- Windows TRL needs `PYTHONUTF8=1` or it crashes loading `chat_templates/deepseekv3.jinja` with cp1252 codec error. Set in `scripts/dev_env.bat` for all training runs.

## Stage 1 GRPO smoke result (recipe needs tightening for E2E)

5 stock GRPO steps with: Qwen2.5-0.5B-Instruct, LoRA r=16 on q/k/v/o_proj, G=8, max_completion=256, bf16, AdamW8bit.

| metric | value |
|---|---|
| wall time | 90s (18s/step avg, 13s/step steady-state) |
| **peak allocated VRAM** | **9365 MB** |
| **peak reserved VRAM** | **10096 MB** |
| step completes | yes (Windows WDDM spills overflow to system RAM) |

**Stage 1 exit criterion is MET** — 5-step stock GRPO completes. But the recipe overflows the 4060's 8 GB and runs against system-RAM spillover, making per-step timing unreliable.

**Recipe to tighten before Stage 4 E2E** (try in this order until fits in ≤7 GB):
1. `gradient_checkpointing=True` (probably enough on its own; ~2× VRAM reduction on activations)
2. drop `num_generations` 8 → 4
3. drop `max_completion_length` 256 → 192
4. fall back to TinyLlama-1.1B (PLAN-allowed)

Don't tune now — for Stage 2 (K1 forward + microbench) the kernel work is independent of the E2E recipe. Microbench allocates its own logits tensor and doesn't go through TRL.

## Next

Stage 2: write naive correct K1 forward (`fused_logprob_entropy`), validate vs `selective_log_softmax(logits, ids)` + `entropy_from_logits(logits)`, build `bench/bench_micro.py` against it. Every later perf change is measured by that harness.
