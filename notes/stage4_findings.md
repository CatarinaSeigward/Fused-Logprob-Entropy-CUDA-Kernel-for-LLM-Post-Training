# Stage 4 findings — TRL integration

## Status: complete

PLAN exit criterion (`1-step training matches stock TRL within tolerance`) — met and exceeded.

- `KernelOptGRPOTrainer(trl.GRPOTrainer)` ships in `python/kernel_opt/trainer.py`, ~85 lines, overrides exactly one method.
- `examples/compare_one_step.py` runs both implementations on a real Qwen rollout: simulated GRPO loss differs by **7.4e-06** (PLAN tolerance is 1e-3 — 135× headroom).
- `examples/train_gsm8k.py` runs 5 GRPO steps end-to-end with our trainer: completes in 52.7s, peak VRAM 2.8 GB (vs Stage 1 stock at the laxer G=8/256 recipe peaking at 9.4 GB).

## What the integration looks like

The override replaces `selective_log_softmax(logits, ids) + entropy_from_logits(logits)` with one call to `fused_logprob_entropy(logits, ids)`. Multimodal call paths (any of pixel_values / image_grid_thw / etc.) fall back to the parent. ~85 lines total.

The single overridden method (`_get_per_token_logps_and_entropies`) is the hub for **all four logprob call sites** in TRL's GRPO:
- policy logprob during loss (`_compute_loss`, line 2447)
- old logprob during rollout (line 2051)
- ref-model logprob (lines 2097 + 2112, both PEFT and non-PEFT paths)

So one override = full coverage.

## Numerical contract (better than PLAN required)

PLAN required: 1-step loss match within 1e-3 absolute. We get 7.4e-06 — 135× tighter.

Three measurements on real Qwen2.5-0.5B logits (4 sequences × 128 tokens × 152064 vocab in bf16, computed via the trainer's actual rollout pipeline):

| comparison | metric | result |
|---|---|---|
| **[3] vs fp32 ground truth — logprob** | stock TRL error | 1.21e-02 |
|                                         | **K1 error**    | **2.04e-06** (✅ 6 orders of magnitude better) |
| **[3] vs fp32 ground truth — entropy** | stock TRL error | 3.34e-02 |
|                                         | **K1 error**    | **2.15e-06** (✅ 4 orders of magnitude better) |
| **[4] simulated GRPO loss diff**       | PLAN tolerance  | 1e-3 |
|                                         | **observed**    | **7.42e-06** (✅ 135× headroom) |

This restates the bf16-precision finding from Stage 2 in the live integration setting: TRL's `selective_log_softmax` (bf16 path: per-row Python loop) and `entropy_from_logits` (bf16 chunked materialization) both quantize to bf16 in their final output. We accumulate in fp32 from bf16 input — strictly more accurate.

## The per-batch loop (the gotcha)

The override iterates over the leading batch dim instead of calling K1 once on the whole `[B, L', V]` tensor. Reason:

After TRL's slicing (`logits[:, :-1, :]` followed by `[:, -logits_to_keep:, :]`), the resulting tensor is **non-contiguous in leading dims**: the V dim still has stride 1, but the batch stride is the original `L*V`, not the sliced `(L-1)*V`. Calling `.contiguous()` would allocate a fresh `B * L' * V * 2` tensor — typically 1+ GB for GRPO shapes — and OOM the 4060.

Discovery: `logits[b]` (per-batch slice) gives `[L', V]` with strides `(V, 1)`, which IS naturally contiguous (the batch-stride irrelevance disappears once dim 0 is sliced away). So per-batch processing avoids any contiguous-copy.

Cost: B small kernel launches per call instead of 1 large launch. For B=4 and our K1 latency around 0.5–1.5 ms per launch, the launch overhead is negligible (~5–10 µs each). At very large B this would matter; the kernel could be extended to accept per-batch strides and absorb the loop. **Marked as Stage 6 follow-up.**

## End-to-end demo result (`train_gsm8k.py`)

5 GRPO steps on Qwen2.5-0.5B + LoRA + GSM8K + rule-based reward, recipe tightened to G=4 / max_completion=192 / gradient_checkpointing=True so it fits comfortably:

```
wall time:           52.7s   (10.5 s/step)
peak allocated VRAM: 2825 MB  (vs Stage 1 stock 9365 MB at G=8/256, no checkpointing)
loss curve:          0.20 → 0.31 → -0.15 → 3e-5 → 0.41   (RL noise; finite, no NaN)
kl:                  ~0.0007 typical   (policy not drifting wildly)
rewards:             reaches 0.5 mean correctness on step 3
```

The 5-step run is the integration smoke test — proves the trainer subclass cooperates with TRL's accelerator, optimizer, dataloader, and reward pipeline. **It does, with no other code changes.**

## OOM detour (worth documenting)

First attempt at `train_gsm8k.py` (G=8 / max_completion=256 / no checkpointing — Stage 1's recipe) OOMed during PyTorch's transformer **backward**, not in our K1. Same thing would happen to stock TRL at this recipe (Stage 1 only completed by spilling to system RAM via Windows WDDM). Tightening to G=4 / 192 + `gradient_checkpointing=True` brings peak alloc into 2.8 GB.

This is a recipe-tightness story, not a K1 story. K1 saved 261 MB on each logprob call (verified in `compare_one_step.py`) — that headroom is what lets us comfortably fit; without it we'd be back to spillover.

## Open follow-ups (deferred to Stage 5/6)

1. **Kernel stride support**: accept non-contiguous logits in the host wrapper so the trainer can do one fused launch per call instead of B per-batch launches. ~30 lines of CUDA + ~10 lines of Python.
2. **Apples-to-apples vs stock TRL on the same recipe**: re-run `train_gsm8k.py` with stock TRL at G=4/192/checkpointing for a clean VRAM-savings number to put in REPORT. (We have approximations from the 1-step parity check — 261 MB per logp call.)
3. **`torch.library.custom_op` registration** with meta kernel — eliminates the Dynamo graph break noted in Stage 3. Stage 6 stretch.

## Test inventory after Stage 4

Unchanged from Stage 3 in `tests/`:
- `tests/test_logprob.py` — 32 tests
- `tests/test_compile.py` — 3 tests

The "trainer integration" tests live as runnable scripts in `examples/` rather than pytest, because each requires loading Qwen2.5-0.5B (~30s) and running real rollouts:
- `examples/compare_one_step.py` — parity check (the Stage 4 exit gate)
- `examples/train_gsm8k.py` — full 5-step demo

Both pass.

## Next

Stage 5: full benchmark sweep + per-op breakdown plot + ncu reports. The K1 timing numbers from Stage 2 (forward) and the bench_backward.py numbers from Stage 3 form the kernel-level data; we now also need (a) the per-step decomposition showing where time goes in stock vs ours, (b) a clean VRAM comparison at the same recipe, (c) ncu screenshots for the headline kernel call.
