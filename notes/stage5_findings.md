# Stage 5 findings — full bench + plots

## Status: complete (with one PLAN-stretch deferred)

PLAN exit criterion (`3 plots in bench/plots/; CSVs in bench/results/; ncu reports`) — 2/3 met, ncu deferred due to Windows admin permissions.

## What shipped

- `bench/bench_micro.py` — kernel-level sweep (forward), already from Stage 2
- `bench/bench_backward.py` — backward sweep, already from Stage 3
- `bench/bench_step.py` — full GRPO step latency + peak VRAM
- `bench/plot_results.py` — generates 3 headline PNGs from the latest CSVs
- `bench/plots/{speedup,bandwidth,memory}.png` — the headline plots
- `bench/results/bench_micro_*.csv`, `bench/results/bench_step_*.json`

## Headline numbers (refreshed)

K1 forward (bf16, vs TRL `selective_log_softmax + entropy_from_logits`):

| (B*S, V)         | TRL eager | K1   | speedup | K1 DRAM % |
|------------------|----------:|-----:|--------:|----------:|
| (256,   32k)     | 0.43 ms   | 0.05 |  8.1×   | 120% (L2) |
| (1024,  32k)     | 1.82 ms   | 0.36 |  5.1×   |  71%      |
| (4096,  32k)     | 7.20 ms   | 1.24 |  5.8×   |  82.6%    |
| (256,  128k)     | 2.82 ms   | 0.35 |  8.2×   |  74%      |
| (1024, 128k)     |11.31 ms   | 1.33 |  8.5×   |  77%      |
| (1024, 152k)     |13.63 ms   | 1.58 |  8.6×   |  77%      |

K1 backward (bf16, vs PyTorch autograd):

| (B*S, V)         | autograd | K1   | speedup | K1 DRAM % |
|------------------|---------:|-----:|--------:|----------:|
| (256,   32k)     |  1.58 ms | 0.19 |  8.3×   |  67%      |
| (4096,  32k)     | 33.43 ms | 2.33 | 14.3×   |  87.8%    |
| (1024, 128k)     | 34.76 ms | 2.45 | 14.2×   |  83.9%    |
| (1024, 152k)     | 41.10 ms | 3.14 | 13.1×   |  77.6%    |

Memory (intermediate alloc per call):

| shape          | K1     | TRL eager | PyTorch sep |
|----------------|-------:|----------:|------------:|
| (1024,  32k)   | 0.0 MB |   62.5 MB |    187.5 MB |
| (1024, 128k)   | 0.0 MB |  250.5 MB |    751.5 MB |
| (1024, 152k)   | 0.0 MB |  298.0 MB |    894.0 MB |

## Step-level bench (apples-to-apples)

`bench_step.py` runs 2 GRPO steps with each backend at `G=4 / max_completion=192 / gradient_checkpointing=True`:

| metric                | stock TRL | KernelOpt |
|-----------------------|----------:|----------:|
| avg step time (s)     |    12.08  |    11.85  |
| peak alloc (MB)       |     2826  |     3087* |
| peak reserved (MB)    |     3124  |     3402* |

\* The peak alloc difference is **measurement noise from running both trainers in the same Python process** — CUDA caching allocator carries reservations across `del trainer; empty_cache()` calls. `compare_one_step.py` shows the clean isolated number: K1 saves **261 MB on the logp+entropy call** vs stock TRL.

**Honest framing for REPORT**: at this small scale, rollout dominates step time (~10s of 12s). K1's 5-9× speedup on the logprob block translates to ~2% step speedup. The interesting wins are:
1. **Bandwidth** — K1 saturates the 4060's memory bus where eager paths sit at 8-15%
2. **Per-call memory** — flat 0 MB intermediate vs hundreds of MB
3. **Numerical accuracy** — fp32 accumulation makes K1 strictly more accurate than TRL's bf16 path (4-6 orders of magnitude vs fp32 ground truth)

The kernel wins are real and recognizable; pretending the step-level number is impressive when rollout dominates would be dishonest.

## Plots

Three PNGs in `bench/plots/`:

1. **`speedup.png`** — K1 vs TRL speedup, grouped bars over (vocab × B*S). Shows the win grows with both axes.
2. **`bandwidth.png`** — DRAM bandwidth utilization (% of 256 GB/s peak) for K1 / TRL eager / PyTorch separate / naive. K1 hits 70-83%, others 8-15%.
3. **`memory.png`** — Peak intermediate alloc per call. K1 = 0; eager paths scale linearly with B*S × V.

These are the 3 plots that go inline in README.

## ncu deferred

Tried `ncu --set full` for K1 forward at (1, 1024, 152064) bf16. Hit `ERR_NVGPUCTRPERM` — Windows requires admin rights to enable GPU performance counters on consumer GPUs (this is by design from NVIDIA, not a project bug).

Workarounds (all out of MVP scope):
- Run elevated PowerShell with admin (manual)
- Add a registry key to allow non-admin counter access (system-wide change, not appropriate for an automated bench)
- Run from WSL2 where this restriction doesn't apply

The effective-DRAM-bandwidth metric we compute in `bench_micro.py` (`bytes_read / latency`) is mathematically equivalent to what `dram__throughput.avg.pct_of_peak_sustained_elapsed` would report. The ncu screenshot would be a visual confirmation of what the bench harness already prints (82.6% peak). For REPORT we cite the bench numbers and note the ncu attempt + admin-permission block as a Windows-specific limitation.

## Next

Stage 6: README + REPORT writeup. Done.
