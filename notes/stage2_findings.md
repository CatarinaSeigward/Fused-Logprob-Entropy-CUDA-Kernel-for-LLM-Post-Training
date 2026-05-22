# Stage 2 findings — K1 forward

## Status: complete

PLAN exit criterion (`K1 forward correct; ≥80% peak DRAM bw on at least one shape; full forward sweep CSV`) — met.

- All 24 correctness tests pass (`pytest tests/test_logprob.py`).
- Bench harness (`bench/bench_micro.py`) ships with three baselines (`ours_naive`, `trl_eager`, `pytorch_separate`) plus the production v1 (`ours_v1`).
- Full sweep CSV at `bench/results/bench_micro_*.csv`.

## Two kernels, same correctness contract

- `_naive_kernel` (one thread per row, scalar V loop) — correctness anchor. Stays in the build for validation.
- `_v1_kernel` (one block per row, warp-shuffle + shared-mem cross-warp reduction, online streaming `(m, Z, T)` accumulator) — production K1 forward.

## Headline numbers (bf16, RTX 4060)

| (B, S, V)         | v1 ms | TRL eager ms | **v1 speedup** | v1 DRAM % peak |
|-------------------|------:|-------------:|---------------:|---------------:|
| (1, 256, 32000)   | 0.053 |        0.430 |       **8.1×** |    120% (L2)   |
| (1, 1024, 32000)  | 0.359 |        1.818 |       **5.1×** |          71%   |
| (1, 4096, 32000)  | 1.239 |        7.197 |       **5.8×** |        82.6%   |
| (1, 256, 128256)  | 0.346 |        2.822 |       **8.2×** |          74%   |
| (1, 1024, 128256) | 1.326 |       11.306 |       **8.5×** |          77%   |
| (1, 256, 152064)  | 0.410 |        3.401 |       **8.3×** |          74%   |
| (1, 1024, 152064) | 1.576 |       13.629 |       **8.6×** |          77%   |

(Peak DRAM bw on RTX 4060 mobile = 256 GB/s.)

## VRAM savings (the headline plot story)

For one call to the loss block on `[B*S, V] = [1024, 152064]` bf16:

| baseline           | peak intermediate alloc |
|--------------------|------------------------:|
| `ours_v1`          |              **0.0 MB** |
| `trl_eager`        |               298.0 MB  |
| `pytorch_separate` |               894.0 MB  |

Per-call savings × steps × layers projects to multiple GB at training scale. This is the bar in headline plot #2 ("fits 2× batch").

## What's interesting (REPORT material)

1. **TRL's bf16 entropy is precision-limited**, ours is more accurate. TRL's `entropy_from_logits` runs the entire `F.log_softmax + exp + mul + sum` chain in bf16 — outputs quantize to values like 8.5000 / 8.5625. Our kernel reads bf16 but accumulates `(m, Z, T)` in fp32. Test `test_at_least_as_accurate_as_trl_bf16` asserts our error is strictly less than TRL's vs the fp32 reference. Free side benefit of fusing.
2. **The (1, 256, 32k) row hits 120% of peak DRAM** because the whole 16 MB logits tensor fits in the 4060's L2 cache (32 MB). Real bandwidth, just to L2 not DRAM. Worth annotating in the roofline plot.
3. **TRL slows down with B*S** more than we do. At (1, 4096, 32k) trl_eager is 7.2 ms (5.8× slower) because its bf16 path runs a Python `for` loop over the batch dim — 4096 Python iterations. Ours is one kernel launch.
4. **naive at small batch is 36× slower than v1** ((1, 256, 32k): 1.95 ms vs 0.053 ms) because we only launch B*S threads — 256 threads vs the 4060's ~36k concurrent capacity. v1 launches 256 blocks × 256 threads = 65k threads. This makes the case for block-per-row in REPORT.

## Optimization headroom not yet taken

- **Vectorized loads**: 8-element bf16 reads (`int4` aligned) would cut load instructions 8×. Likely pushes 77% → ~88-92% on V≥128k shapes.
- **Larger BLOCK_DIM (512 / 1024)**: more parallelism per row at small B*S, less benefit at large B*S.
- **Persistent kernel** for very small B*S to amortize launch overhead.

Holding off on these — the PLAN exit criterion is met and the headline numbers already tell a strong story. Returning to optimization only if Stage 3/4 work surfaces a real motivator.

## Next

Stage 3: K1 backward (streaming version using saved `lse`) + `autograd.Function` wrapper + `torch.library.custom_op` registration with `meta` kernel. Then `torch.compile` round-trip test.
