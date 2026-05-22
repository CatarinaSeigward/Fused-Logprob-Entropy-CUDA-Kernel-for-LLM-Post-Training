# kernel-opt — Beginner's Manual

> Read-once tutorial for someone who's never written a CUDA kernel and wants
> to learn from this project. Two parts:
>
> - **Part A** — exact commands to reproduce the project from a clean machine.
> - **Part B** — every concept used in the codebase, explained from scratch.
>
> Aimed at a CS undergrad, ML engineer, or any working programmer who knows
> Python, has touched C++ once, and finds GPU programming intimidating. By
> the end you should be able to read every line of `csrc/fused_logprob.cu`
> and know why each one is there.

---

# Part A — Step-by-step reproduction

You'll run these commands in order. If anything fails, jump to the
troubleshooting table at the end of Part A (§A12) before retrying.

## A1. Prerequisites

| What | Minimum | What this project used | Why |
|---|---|---|---|
| GPU | NVIDIA Ampere (sm_80) or newer, 8 GB VRAM | RTX 4060 Laptop (sm_89, 8 GB) | bf16 native + enough VRAM for Qwen-0.5B |
| OS | Windows 11 or Linux | Windows 11 | Linux has fewer toolchain headaches |
| Disk | 15 GB free | — | venv (~6 GB) + Qwen weights (~1 GB) + datasets cache + build artifacts |
| System RAM | 16 GB | 32 GB | for HF dataset processing |
| Network | enough to download ~5 GB | — | PyTorch wheel (~2 GB), Qwen model (~1 GB), datasets, etc. |
| Patience | a couple of hours | — | first-time toolchain setup is real work |

You do **not** need: an A100/H100, multi-GPU setup, vLLM, or admin/root
access (except for Nsight Compute, which we work around).

## A2. Toolchain — Windows native (the most fragile path)

This is the path I used. Linux is easier; skip to A3 if you're on Linux.

### A2.1 Install CUDA 12.6 toolkit

1. Download from
   [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-12-6-0-download-archive)
   — pick `Windows / x86_64 / 11 / exe (local)`.
2. Run the installer. Accept defaults except: under **Custom Install**,
   ensure "CUDA → Development" and "CUDA → Runtime" are checked.
3. Verify in a fresh PowerShell:
   ```powershell
   nvcc --version    # should print CUDA 12.6
   nvidia-smi         # should show your GPU
   ```

### A2.2 Install Visual Studio with MSVC 14.29 (the version CUDA 12.6 accepts)

CUDA 12.6 only supports MSVC versions 14.16–14.41 (VS 2017 to VS 2022). Newer
MSVC will get rejected with `error C1189: -- unsupported Microsoft Visual
Studio version!`.

The cleanest setup: **Visual Studio 2019 Build Tools** (free) with the
"MSVC v142 - VS 2019 C++ x64/x86 build tools" workload. If you already have
VS 2022 with the C++ workload, that also works. If you have multiple MSVC
versions installed, the build script (§A7) will explicitly pick `14.29`.

Find your `vcvars64.bat`:
```powershell
# Typical paths:
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat
# or
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
```

Verify:
```powershell
cmd /c '"C:\path\to\vcvars64.bat" && where cl'
# Should print path ending in MSVC\14.29.xxxxx\bin\HostX64\x64\cl.exe
```

### A2.3 Install Python 3.12

**Use Python 3.12, not 3.13.** PyTorch wheels for 3.13 + Windows + CUDA
weren't fully landed at the time of this project, and `bitsandbytes` lacks
3.13 wheels. Get the installer from
[python.org](https://www.python.org/downloads/release/python-3128/).

### A2.4 Install `uv` (recommended)

`uv` is a fast Python package manager. Install via PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

(If you'd rather use `pip + venv`, you can; substitute commands accordingly.)

## A3. Toolchain — Linux

Much simpler:
```bash
sudo apt install build-essential
# Install CUDA 12.6 from NVIDIA's apt repo per https://developer.nvidia.com/cuda-downloads
# Install Python 3.12
sudo apt install python3.12 python3.12-venv
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No MSVC drama. No vcvars dance. The rest of the steps below replace
`scripts\dev_env.bat` invocations with direct commands.

## A4. Clone the repo and create a venv

```powershell
# (Windows PowerShell, or bash on Linux)
git clone <your-fork-url>  # or wherever the repo lives
cd kernel-opt

uv venv --python 3.12 .venv
# Activate (Windows PowerShell):
.venv\Scripts\Activate.ps1
# Activate (Linux/macOS):
# source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

## A5. Install PyTorch with CUDA support

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

This pulls ~2.4 GB. PyTorch built against CUDA 12.4 wheels works fine with the
CUDA 12.6 toolkit (CUDA 12.x is forward-compatible).

Verify:
```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print('bf16:', torch.cuda.is_bf16_supported())"
```

You should see something like:
```
2.6.0+cu124 12.4
True
NVIDIA GeForce RTX 4060 Laptop GPU
(8, 9)
bf16: True
```

`(8, 9)` means sm_89. Anything `>= (8, 0)` will work for this project.

## A6. Install build deps and the training stack

```powershell
uv pip install numpy ninja pytest matplotlib
uv pip install transformers trl peft datasets accelerate bitsandbytes
```

Verify (run `scripts\check_stack.py` if present, otherwise inline):
```powershell
python -c "import torch, transformers, trl, peft, datasets, bitsandbytes; print('all imports OK')"
```

## A7. Build the kernel

This is the most failure-prone step on Windows. The repo ships
`scripts\dev_env.bat` which sets up the right MSVC version + a critical
`DISTUTILS_USE_SDK=1` flag and then runs whatever command you pass.

```powershell
# Windows:
cmd /c "scripts\dev_env.bat && uv pip install --no-build-isolation -e ."
```

```bash
# Linux:
uv pip install --no-build-isolation -e .
```

What's happening: `setup.py` declares a `CUDAExtension` that compiles
`csrc/bindings.cpp` and `csrc/fused_logprob.cu` into a Python-importable
shared library `kernel_opt._C`. The compilation takes ~30–60 seconds on a
fast machine.

**Verify**:
```powershell
python -c "from kernel_opt import _C; print('built OK:', _C)"
# Expect: built OK: <module 'kernel_opt._C' from '...kernel_opt\\_C.cp312-win_amd64.pyd'>
```

If this fails: see §A12 troubleshooting.

## A8. Run the test suite

```powershell
# Windows
$env:PYTHONUTF8 = "1"
python -m pytest tests/ -v

# Linux:
PYTHONUTF8=1 pytest tests/ -v
```

Expected: **35 passed in ~10 seconds.** The `PYTHONUTF8=1` env var is needed
because TRL's chat-template files contain UTF-8 characters and Windows
defaults to cp1252 which crashes on import.

If even one test fails, **stop and debug** — the kernel doesn't work. The
tests are designed so that a single failure points at a specific math/memory
bug.

## A9. Run the microbenchmark + plots

```powershell
python bench/bench_micro.py
python bench/plot_results.py
```

`bench_micro.py` sweeps shapes (vocab × batch×seq), times K1 vs three
baselines, and writes a CSV to `bench/results/`. `plot_results.py` reads the
CSV and writes three PNGs to `bench/plots/`:
- `speedup.png` — K1 vs TRL eager speedup factor
- `bandwidth.png` — DRAM bandwidth utilization (% of 256 GB/s peak)
- `memory.png` — peak intermediate alloc per call

Open the PNGs. K1's bars should dominate.

## A10. Run the GRPO end-to-end demo

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python examples/train_gsm8k.py
```

This downloads Qwen2.5-0.5B-Instruct (~1 GB on first run), loads it with a
LoRA adapter, sets up TRL's `KernelOptGRPOTrainer` (our subclass), and runs
5 GRPO steps on GSM8K with a rule-based reward.

Expected: `~50 seconds` total, peak VRAM ~2.8 GB, loss values printed per
step, no crashes.

## A11. Run the parity check vs stock TRL

```powershell
python examples/compare_one_step.py
```

This loads Qwen2.5-0.5B once, gets a real GRPO rollout, then runs the logp
computation two ways: TRL's stock eager path, and ours. Compares both
against a fp32 ground truth recomputed from the same logits.

Expected output ends with:
```
Stage 4 exit criteria:
  ours <= stock vs fp32 ground truth (logp): PASS  (ours 2.04e-06 vs stock 1.21e-02)
  ours <= stock vs fp32 ground truth (ent):  PASS  (ours 2.15e-06 vs stock 3.34e-02)
  simulated GRPO loss diff < 0.001: PASS  (diff 7.42e-06)
```

If you see PASS on all three lines, you've successfully reproduced the
project's headline result.

## A12. Troubleshooting cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `'cl.exe' is not recognized` | MSVC not on PATH | Run via `cmd /c "scripts\dev_env.bat && <command>"` |
| `unsupported Microsoft Visual Studio version` | CUDA 12.6 picked an MSVC > 14.41 | Edit `scripts\dev_env.bat` to point at 14.29 / 14.41; or install VS 2019 BuildTools |
| `fatal error C1083: stddef.h: No such file` | Windows SDK not in include path | `vcvars64.bat` should fix this; verify it ran |
| `'charmap' codec can't decode byte` on `import trl` | Windows cp1252 default | `set PYTHONUTF8=1` in your shell |
| `ModuleNotFoundError: triton` from `torch.compile` | Triton not installed on Windows | Use `backend="aot_eager"` or run on Linux/WSL2 |
| `CUDA out of memory` in `train_gsm8k.py` | Recipe too big for 8 GB | Lower `NUM_GENERATIONS` (4 → 2) and/or `MAX_COMPLETION_LEN` (192 → 128) |
| `bitsandbytes` import errors on Windows | bnb wheel mismatch | Check Python is 3.12 (not 3.13); reinstall: `uv pip install --force-reinstall bitsandbytes` |
| `peak alloc reported 0 MB` in bench | `reset_peak_memory_stats()` race | Just re-run; first iteration sometimes shows 0 |
| `RuntimeError: Failed to initialize NumPy` warning | torch noticed numpy missing | Install numpy: `uv pip install numpy` |
| `ERR_NVGPUCTRPERM` running ncu | Windows requires admin for GPU counters | Run from admin PowerShell, or use WSL2 |

---

# Part B — Foundational knowledge

Each subsection is self-contained but they build on each other. Read in
order if you're new to GPU programming.

## B1. The big picture (no GPU code yet)

### What is "LLM post-training" and why does it matter for this kernel?

You probably know LLMs are pretrained on the internet (next-token
prediction). That gets a model that's fluent but not necessarily aligned with
human preferences or capable of multi-step reasoning. **Post-training** is
the second phase that does:
- **SFT** (supervised fine-tuning) on instruction–response pairs
- **RLHF** (reinforcement learning from human feedback) — train a reward
  model, then optimize the LLM to produce responses the reward model likes
- **GRPO / DPO / PPO** — the specific algorithms used in 2025-26 frontier
  labs

The hot loop of these algorithms is:
1. **Rollout**: sample G candidate completions per prompt
2. **Score**: compute a reward for each completion
3. **Loss**: compare current policy's probability of the completions to a
   reference policy's, weighted by reward
4. **Backward + optimizer step**: update the policy

Step 3 is where K1 lives. Specifically: "current policy's probability" means
**per-token log-probability**. For each token in each completion, you look
at the model's logits and ask "what log-probability did the model assign to
the token that was actually chosen?".

### Why is per-token logp from a `[B, S, V]` tensor a bottleneck?

Take Qwen2.5: vocabulary V = 152,064. A single training step might process
B·S = 1024 tokens (4 prompts × 8 generations × 32 tokens each, say). The
logits tensor is then `[1024, 152064] × 2 bytes (bf16) ≈ 312 MB`.

To extract one log-probability per token, you need to compute
`log_softmax(logits)[t]` for the chosen token `t` at each position. The
naive way is:
```python
log_probs = F.log_softmax(logits, dim=-1)        # alloc 312 MB (or 624 MB in fp32!)
selected = log_probs.gather(-1, tokens.unsqueeze(-1))
```

You materialized 312–624 MB of intermediate just to throw most of it away.
That's a lot of DRAM traffic for one number per row.

**K1 fuses these into a single streaming pass that never materializes the
softmax.** The "fusion" word means: instead of doing two operations with an
intermediate result stored in DRAM, do them as one combined operation that
keeps the intermediate in fast on-chip memory (registers / shared memory).

### Memory-bound vs compute-bound, in 30 seconds

Every kernel is bottlenecked either by:
- **Compute**: ALU/Tensor-Core throughput — how fast the GPU can do math
- **Memory**: DRAM bandwidth — how fast it can move bytes

On a 4060: compute peak ≈ 121 TFLOPs bf16, memory peak = 256 GB/s.
- Compute-bound kernel: doing many ops per byte (e.g., a big GEMM).
- Memory-bound kernel: doing few ops per byte (e.g., element-wise add,
  reductions).

K1 is firmly memory-bound. It does ~5 floating-point ops per byte read.
**The only way to make it faster is to reduce bytes read** — which is
exactly what fusion does.

### Why GPU?

CPU memory bandwidth is ~50 GB/s. GPU is 5× more. For a 312 MB tensor that
gets read twice (logp + entropy), that's a 12.5 GB read. CPU: 250 ms. GPU at
peak: 50 ms. Real PyTorch eager on GPU: ~13 ms. Our K1: ~1.3 ms. **The order
of magnitude that wins or loses is on-GPU; the CPU isn't even close.**

## B2. CUDA programming model

### The hardware

A modern NVIDIA GPU is a collection of **streaming multiprocessors (SMs)**.
The 4060 has 24 SMs. Each SM has:
- ~128 CUDA cores (scalar FP32/INT execution units)
- 4 Tensor Cores (matrix-multiply units; we don't use them in this project)
- 100 KB of "shared memory" (programmer-managed L1-equivalent)
- Registers (each thread gets its own slice)
- Schedules up to 1536 threads concurrently

Total concurrent threads on a 4060: 24 × 1536 ≈ **36,000**. If your kernel
launches fewer than this, the GPU is under-utilized.

### Threads → warps → blocks → grids

- **Thread**: smallest unit. Has its own registers and program counter.
- **Warp**: 32 threads that execute in lockstep (SIMT — single instruction,
  multiple threads). Inter-warp communication via fast shuffle instructions.
- **Block** (a.k.a. cooperative thread array): up to 1024 threads, share an
  allocation of shared memory, can `__syncthreads()` to wait for each other.
  A block runs entirely on one SM.
- **Grid**: collection of blocks. Blocks within a grid **cannot** wait for
  each other (no global sync inside a kernel launch).

When you launch a kernel, you specify `<<<gridDim, blockDim>>>`. For K1's v1
kernel: `<<< N=B*S, blockDim=256 >>>`. That's `N` blocks (one per row of
logits) of 256 threads each. Total threads = N × 256.

### Memory hierarchy

From fastest to slowest, with rough latencies:

| Level | Size | Latency | Bandwidth |
|---|---|---|---|
| Registers | ~64 KB/SM | 1 cycle | absurd |
| Shared memory | 100 KB/SM (4060) | ~30 cycles | TB/s aggregate |
| L1 cache | shared with SMEM | ~30 cycles | TB/s |
| L2 cache | 32 MB (4060) | ~200 cycles | ~1 TB/s |
| DRAM (HBM/GDDR) | 8 GB (4060) | ~500 cycles | 256 GB/s |

**Key insight for memory-bound kernels**: you want to read each byte from
DRAM exactly once, do all the work in registers + shared memory, and write
once. That's what K1 does.

### Kernel launch syntax

```cpp
__global__ void my_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// Host code:
int n = 1024 * 1024;
my_kernel<<<(n + 255) / 256, 256>>>(d_out, d_in, n);
```

`__global__` marks a function as a kernel (callable from host, runs on
device). `<<<gridDim, blockDim>>>` launches it. Each thread's
`threadIdx`/`blockIdx`/`blockDim`/`gridDim` are built-in variables.

### Why one block per row beats one thread per row

Suppose `B*S = 256, V = 128k`. A naive kernel that uses **one thread per row**
launches 256 threads — way under 36k. Each thread serially walks 128k
elements. The GPU is idle 99% of the time.

A **one block per row** kernel launches 256 blocks × 256 threads = 65k
threads. Each thread walks 500 elements (128k / 256). Now the GPU is
saturated and the per-row work is parallelized 256-way within each block.

This is the single biggest speedup in the project (~30× from naive to v1
forward).

## B3. Numerics: bf16, fp16, fp32 — and why fp32 accumulation matters

### IEEE 754 cheatsheet

A floating-point number is `(-1)^sign × 1.mantissa × 2^exponent`. Each
format trades range (exponent bits) for precision (mantissa bits):

| Format | Sign | Exp | Mantissa | Range | ~Decimal precision |
|---|---|---|---|---|---|
| **fp32** | 1 | 8 | 23 | ±10³⁸ | 7 digits |
| **fp16** | 1 | 5 | 10 | ±65,504 | 3.3 digits |
| **bf16** | 1 | 8 | 7 | ±10³⁸ | 2.5 digits |

bf16 has the **same range as fp32** but fewer mantissa bits. This is great
for ML because gradients can be tiny (1e-7) or huge (1e6) and bf16 doesn't
overflow/underflow like fp16 does. The cost: only ~3 significant decimal
digits.

### Why summing many bf16 values is dangerous

If you add 152,064 bf16 numbers naively, the rounding error per add is
~2⁻⁸ ≈ 0.4%. Over 152k adds, errors compound. Final relative error can be
several percent.

In fp32, mantissa is 23 bits → per-add error 2⁻²³ ≈ 10⁻⁷. Over 152k adds,
final error ~10⁻². Still small.

**The K1 design rule**: read inputs in bf16 (saves 2× DRAM traffic), but the
moment a value enters a register, **convert to fp32 and accumulate in
fp32**. Cost is essentially zero (registers are free); accuracy gain is 5
orders of magnitude.

This is exactly why K1 ends up MORE numerically accurate than TRL's bf16
path, which accumulates entirely in bf16. See `notes/stage2_findings.md` for
the measurement.

## B4. The math behind K1

### Naive cross-entropy

For a single row of logits `x ∈ R^V` and a target index `t ∈ {0..V-1}`:

```
log_softmax(x)[v] = x[v] - logsumexp(x)
log_prob_of_target = log_softmax(x)[t] = x[t] - logsumexp(x)
```

The expensive piece is `logsumexp(x) = log(Σ_v exp(x[v]))`. Naively
implementing the log-of-sum-of-exp overflows when any `x[v] > ~88` (because
`exp(88) ≈ 10³⁸`, the fp32 max).

### The classical max-shift trick (two passes)

```
m = max(x)
lse = m + log(Σ_v exp(x[v] - m))    # exp arguments are now <= 0, no overflow
```

This requires **two passes** over `x`: one to find `m`, one to compute the
sum. Two passes = 2× DRAM traffic. We can do better.

### The online (one-pass) version (Milakov & Gimelshein, 2018)

Maintain running `(m, Z)` state. Initialize `m = -∞, Z = 0`. For each new
element `x`:

```
m_new = max(m, x)
Z_new = exp(m - m_new) · Z + exp(x - m_new)
m, Z ← m_new, Z_new
```

**Why this works algebraically**: when the running max `m` increases to
`m_new`, every previously-summed `exp(x_i - m_old)` term needs to be
rescaled by `exp(m_old - m_new)` to express it in the new `m_new` reference
frame. The first term in `Z_new` does that rescaling; the second term adds
the new element.

After processing all V elements: `lse = m + log(Z)`.

### The first-iteration trap (where I lost an hour)

If you initialize `m = -∞, Z = 0`, the first iteration's update is:
```
m_new = max(-∞, x) = x
delta = m - m_new = -∞ - x = -∞
Z_new = exp(-∞) · 0 + exp(0) = 0 · 0 + 1 = 1     # math
                              = NaN · 0 + 1 = NaN   # IEEE float
```

`-∞ × 0 = NaN` in IEEE 754. The fix: peel off the first iteration. Initialize
from `x[0]` directly: `m = x[0], Z = 1`. Then loop from `v = 1`.

This bug pattern shows up in any streaming reduction — remember it.

### Extending to entropy

Shannon entropy `H = -Σ_v p_v · log p_v`, where `p_v = softmax(x)_v`. With
algebra (substitute `log p_v = x_v - lse`):

```
H = lse - Σ_v p_v · x_v
  = log(Z) - (1/Z) · Σ_v (x_v - m) · exp(x_v - m)
```

Define a third running accumulator
`T = Σ_v (x_v - m) · exp(x_v - m)`. When `m` updates, `T` needs the same
rescaling as `Z`, plus a correction term:

```
T_new = exp(m - m_new) · (T + (m - m_new) · Z) + (x - m_new) · exp(x - m_new)
```

The `(m - m_new) · Z` correction comes from this: every old `(x_i - m_old)`
term in `T_old` shifts by `(m_old - m_new)` when re-expressed in the
`m_new` frame, contributing an extra `(m_old - m_new) · exp(x_i - m_old)`
that sums to `(m_old - m_new) · Z_old`.

Final: `entropy = log(Z) - T / Z`.

### Combining two partial states (for parallel reduction)

If thread A computed `(m_a, Z_a, T_a)` over half the row and thread B
computed `(m_b, Z_b, T_b)` over the other half, we can merge them:

```
m  = max(m_a, m_b)
d_a = m_a - m,   d_b = m_b - m       # both <= 0
Z  = exp(d_a) · Z_a + exp(d_b) · Z_b
T  = exp(d_a) · (T_a + d_a · Z_a) + exp(d_b) · (T_b + d_b · Z_b)
```

This combine operation is **associative**, which is what allows tree-style
reduction (warp shuffle + shared memory). You can verify associativity by
direct algebraic substitution.

### The empty sentinel pitfall

Sometimes a thread has no work (e.g., when `V` doesn't divide block_dim
evenly). Its state is `(m=-∞, Z=0, T=0)`. Combining with a real state:
- `m = max(real_m, -∞) = real_m`
- `d_empty = -∞ - real_m = -∞`, so `exp(d_empty) = 0`
- `T contribution = 0 · (0 + (-∞) · 0) = 0 · NaN = NaN`

Fix: explicit early return in the combine function:
```cpp
if (b.m == -INFINITY) return a;
if (a.m == -INFINITY) return b;
```

Same `-∞ × 0 = NaN` pattern as the first-iteration trap.

## B5. K1 forward kernel — code walkthrough

Open `csrc/fused_logprob.cu` alongside this section.

### The state struct and combine

```cpp
struct PartialState {
    float m;
    float Z;
    float T;
};

__device__ __forceinline__ PartialState combine(PartialState a, PartialState b) {
    if (b.m == -INFINITY) return a;     // empty-sentinel guard
    if (a.m == -INFINITY) return b;
    float m_new = fmaxf(a.m, b.m);
    float d1 = a.m - m_new;
    float d2 = b.m - m_new;
    float s1 = __expf(d1);              // device intrinsic, fast approximate
    float s2 = __expf(d2);
    PartialState r;
    r.m = m_new;
    r.Z = s1 * a.Z + s2 * b.Z;
    r.T = s1 * (a.T + d1 * a.Z) + s2 * (b.T + d2 * b.Z);
    return r;
}
```

`__device__` marks this as device-only (callable from kernels, not host).
`__forceinline__` tells nvcc to inline aggressively (we want zero call
overhead). `__expf` is the fast approximate exp on hardware (~few cycles vs
20+ for `expf`).

### The main kernel

```cpp
template <typename T, int BLOCK_DIM>
__global__ void fused_logprob_entropy_v1_kernel(
    const T* __restrict__ logits,           // [N, V] flattened
    const int64_t* __restrict__ targets,    // [N]
    float* __restrict__ logprob,            // [N]
    float* __restrict__ entropy,            // [N]
    float* __restrict__ lse_out,            // [N]
    int V)
{
```

Templated on input dtype (`T` = bf16/fp16/fp32) and `BLOCK_DIM` (=256). The
`__restrict__` keyword is a promise to the compiler that pointers don't
alias, enabling more aggressive optimization.

```cpp
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;     // tid mod 32, the warp-relative index
    int warp = tid >> 5;     // tid / 32, the warp index within the block
```

One block per row. Each thread knows its `lane` (0-31) and `warp` (0-7 for
BLOCK_DIM=256).

```cpp
    const T* row_ptr = logits + (int64_t)row * V;
    int64_t tgt = targets[row];

    PartialState s = {-INFINITY, 0.0f, 0.0f};
    __shared__ float s_target_logit;
    if (tid == 0) s_target_logit = 0.0f;
    __syncthreads();
```

Each thread starts with empty state. Shared scalar `s_target_logit` will be
written by exactly one thread (the one whose strided sweep lands on
`v == tgt`).

```cpp
    for (int v = tid; v < V; v += BLOCK_DIM) {
        float x = to_float<T>(row_ptr[v]);   // bf16/fp16 → fp32
        if (v == tgt) s_target_logit = x;    // exactly one thread does this
        PartialState elem = {x, 1.0f, 0.0f}; // single-element state at max=x
        s = combine(s, elem);
    }
```

This is the **streaming pass**: each thread walks its strided slice of V,
accumulates its local `(m, Z, T)`. No shared memory traffic in this loop —
all per-thread registers.

The `(x, 1.0f, 0.0f)` initialization is the "single-element state": for one
element with value `x`, we have `m = x`, `Z = exp(x - x) = 1`, `T = 0 ·
exp(0) = 0`. The combine function then merges this with the running state.

```cpp
    // Phase 2a: warp-level reduction via shuffles.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        PartialState other;
        other.m = __shfl_xor_sync(0xffffffff, s.m, offset);
        other.Z = __shfl_xor_sync(0xffffffff, s.Z, offset);
        other.T = __shfl_xor_sync(0xffffffff, s.T, offset);
        s = combine(s, other);
    }
```

**Warp shuffle reduction** (butterfly pattern). After 5 iterations
(offset = 16, 8, 4, 2, 1), all 32 lanes in the warp hold the warp-wide
reduced state.

`__shfl_xor_sync(mask, value, offset)` makes thread `lane` exchange
`value` with thread `lane XOR offset`. With offset=16, lanes 0-15 swap with
lanes 16-31. With offset=8, halves of halves swap. Etc.

The `0xffffffff` mask says "all 32 lanes participate" (used to be needed
for correct behavior; modern PTX requires it).

```cpp
    // Phase 2b: cross-warp reduction via shared memory.
    __shared__ float s_m[N_WARPS], s_Z[N_WARPS], s_T[N_WARPS];
    if (lane == 0) {
        s_m[warp] = s.m; s_Z[warp] = s.Z; s_T[warp] = s.T;
    }
    __syncthreads();
```

Lane 0 of each warp writes its warp's reduced state to shared memory.
`__syncthreads()` makes all threads wait until every warp has written.

```cpp
    if (warp == 0) {
        s = (lane < N_WARPS)
            ? PartialState{s_m[lane], s_Z[lane], s_T[lane]}
            : PartialState{-INFINITY, 0.0f, 0.0f};
        // Same butterfly reduce
        for (int offset = 16; offset > 0; offset >>= 1) { ... }
        if (lane == 0) {
            float lse_val = s.m + __logf(s.Z);
            logprob[row] = s_target_logit - lse_val;
            entropy[row] = __logf(s.Z) - s.T / s.Z;
            lse_out[row] = lse_val;
        }
    }
}
```

Only warp 0 does the second reduction. Lanes 0-7 carry valid data (one per
warp); lanes 8-31 use the empty sentinel. After the second butterfly,
lane 0 has the final state and writes the three outputs.

That's the entire forward kernel. ~50 lines of CUDA, ~80% peak DRAM
bandwidth.

## B6. K1 backward — gradient derivation and kernel

### Why backward at all?

Training a model means computing the gradient of the loss with respect to
the parameters. If our op is in the middle of the network, PyTorch's
autograd engine will at some point ask: *"given the gradient with respect to
my outputs, what are the gradients with respect to my inputs?"*. If we don't
provide a backward, autograd can't propagate through us.

### The math

K1 produces three outputs `(logprob, entropy, lse)`, all functions of the
input `x ∈ R^V`. PyTorch will hand us upstream gradients
`(g_logp, g_ent, g_lse)`, each a scalar per row. Our job: compute
`d_x ∈ R^V` per row.

By the chain rule, each output contributes additively to `d_x`:

| Output      | ∂(output)/∂x_v                  | Contribution to d_x[v]                    |
|-------------|---------------------------------|-------------------------------------------|
| `logprob`   | `δ(v == t) - p_v`               | `g_logp · (δ(v == t) - p_v)`              |
| `lse`       | `p_v`                           | `g_lse · p_v`                             |
| `entropy`   | `-p_v · (log_p_v + entropy)`    | `g_ent · (-p_v) · (log_p_v + entropy)`    |

(Derivation of `∂entropy/∂x_v`: `entropy = lse - Σ_w p_w x_w`. Differentiating
the sum w.r.t. `x_v` and simplifying gives `-p_v · (log_p_v + entropy)`. The
algebra is in `notes/stage3_findings.md`.)

Combining and factoring:

```
d_x[v] = p_v · (g_lse - g_logp - g_ent · (log_p_v + entropy))
       + (v == t ? g_logp : 0)
```

Where `p_v = exp(x_v - lse)` and `log_p_v = x_v - lse`. We have `lse` and
`entropy` saved from forward (via `ctx.save_for_backward`); `p_v` is
recomputed on the fly. **No softmax materialization.**

### The kernel

```cpp
template <typename T, int BLOCK_DIM>
__global__ void fused_logprob_entropy_v1_backward_kernel(
    const T* __restrict__ logits,        // re-read in backward
    const int64_t* __restrict__ targets,
    const float* __restrict__ lse,       // saved
    const float* __restrict__ entropy,   // saved
    const float* __restrict__ g_logp,    // upstream
    const float* __restrict__ g_ent,
    const float* __restrict__ g_lse,
    T* __restrict__ d_logits,            // output, same dtype as logits
    int V)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const T* row_logits = logits + (int64_t)row * V;
    T* row_dlogits = d_logits + (int64_t)row * V;

    int64_t tgt = targets[row];
    float row_lse = lse[row];
    float row_ent = entropy[row];
    float row_g_logp = g_logp[row];
    float row_g_ent = g_ent[row];
    float row_g_lse = g_lse[row];
```

Load all per-row scalars once into registers. Each thread will reuse them
across many iterations of the V loop.

```cpp
    // Hoist the v-independent part of the coefficient out of the loop:
    //   coeff(v) = (g_lse - g_logp - g_ent * entropy) - g_ent * log_p_v
    float const_part = row_g_lse - row_g_logp - row_g_ent * row_ent;
```

Optimization: the `(g_lse - g_logp - g_ent · entropy)` part doesn't depend
on `v`. Compute it once, save 3 ops per V iteration.

```cpp
    for (int v = tid; v < V; v += BLOCK_DIM) {
        float x = to_float<T>(row_logits[v]);
        float log_p_v = x - row_lse;
        float p_v = __expf(log_p_v);
        float coeff = const_part - row_g_ent * log_p_v;
        float dval = p_v * coeff;
        if (v == tgt) dval += row_g_logp;
        row_dlogits[v] = from_float<T>(dval);
    }
}
```

That's the entire backward loop. **No reduction is needed** because the
gradient is pointwise (each `d_x[v]` only depends on `x[v]` plus per-row
saved scalars). This is why backward hits 87.8% peak DRAM bw — even closer
to the ceiling than forward.

### Wrapping forward + backward in autograd.Function

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
        logits, targets, lse, entropy = ctx.saved_tensors
        # Convert any None grads to zeros (downstream may not use all 3 outputs)
        if g_logp is None: g_logp = torch.zeros_like(lse)
        if g_ent  is None: g_ent  = torch.zeros_like(lse)
        if g_lse  is None: g_lse  = torch.zeros_like(lse)
        g_logp = g_logp.to(torch.float32, copy=False).contiguous()
        g_ent  = g_ent.to(torch.float32, copy=False).contiguous()
        g_lse  = g_lse.to(torch.float32, copy=False).contiguous()

        d_logits = torch.empty_like(logits)
        _C.fused_logprob_entropy_v1_backward(
            logits, targets, lse, entropy, g_logp, g_ent, g_lse, d_logits)
        return d_logits, None      # (grad for logits, None for targets)
```

Two methods: `forward` saves whatever backward will need; `backward`
consumes upstream gradients and returns input gradients (one per input;
non-tensor inputs get `None`).

The `if g_X is None: g_X = zeros` pattern handles the case where the user
only consumed one of the three outputs. Autograd then passes `None` for the
unused outputs' gradients.

## B7. Connecting CUDA to PyTorch

### pybind11 in 5 minutes

pybind11 is a header-only C++ library that exposes C++ functions as Python.
The bindings file (`csrc/bindings.cpp`):

```cpp
#include <torch/extension.h>

namespace kernel_opt {
void fused_logprob_entropy_v1(
    torch::Tensor logits, torch::Tensor targets,
    torch::Tensor logprob, torch::Tensor entropy, torch::Tensor lse_out);
// ... declarations of other host functions ...
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_logprob_entropy_v1",
          &kernel_opt::fused_logprob_entropy_v1,
          "v1 K1 forward: one-block-per-row, warp+block reduction.");
    // ... other m.def calls ...
}
```

`PYBIND11_MODULE` declares the Python module entry point. `m.def("name",
&func, "docstring")` exposes one C++ function. `TORCH_EXTENSION_NAME` is a
macro that expands to whatever name `setup.py` declared (here `_C`).

After the build, you can do `from kernel_opt import _C; _C.fused_logprob_entropy_v1(...)`
in Python.

### torch.utils.cpp_extension

`torch.utils.cpp_extension.CUDAExtension` is a setuptools extension that
knows how to compile `.cpp` and `.cu` sources, link against PyTorch's C++
libraries, and produce an importable Python extension. From `setup.py`:

```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="kernel_opt._C",
        sources=["csrc/bindings.cpp", "csrc/fused_logprob.cu"],
        extra_compile_args={
            "cxx":  ["/O2", "/std:c++17"],
            "nvcc": ["-O3", "--use_fast_math", "-std=c++17",
                     "-gencode=arch=compute_89,code=sm_89"],
        },
    ),
]
setup(name="kernel_opt", ext_modules=ext_modules,
      cmdclass={"build_ext": BuildExtension}, ...)
```

`-gencode=arch=compute_89,code=sm_89` tells nvcc to generate machine code for
sm_89 (Ada Lovelace). On a different GPU you'd change the number (sm_80
Ampere, sm_90 Hopper, etc.).

`--use_fast_math` enables fast approximate intrinsics like `__expf`. On a
memory-bound kernel like ours, this doesn't speed anything up directly but
keeps the math out of the bottleneck.

### torch.compile compatibility (briefly)

`torch.compile` traces your Python function and produces a fused graph. When
it encounters our pybind C extension call, it can't trace through the C++
code, so it issues a "graph break": the surrounding Python is compiled, but
our op is called eagerly at the boundary. This is correct but not maximally
optimal — a fully-fused graph would be faster.

The fix (deferred to a stretch goal) is to register the op via
`torch.library.custom_op` with a `meta` kernel:

```python
@torch.library.custom_op("kernel_opt::fused_logprob_entropy", mutates_args=())
def _op(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return fused_logprob_entropy_forward(logits, targets)

@_op.register_fake
def _meta(logits, targets):
    out_shape = targets.shape
    return (torch.empty(out_shape, dtype=torch.float32, device=logits.device),
            torch.empty(out_shape, dtype=torch.float32, device=logits.device),
            torch.empty(out_shape, dtype=torch.float32, device=logits.device))
```

`meta` (also called "fake") kernels return correctly-shaped dummy tensors
without doing the actual computation, allowing Dynamo to do shape inference
through the op. With this registration, no graph break.

## B8. Integrating into a real framework (TRL)

### The right pattern: subclass + override

HuggingFace's `Trainer` (and TRL's `GRPOTrainer`) is a class with many
methods. To customize behavior, **subclass and override**, never fork.

```python
class KernelOptGRPOTrainer(GRPOTrainer):
    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                           logits_to_keep, ...):
        # Our K1-based implementation, with multimodal fallback to parent.
        if is_multimodal: return super()._get_per_token_logps_and_entropies(...)
        # ... text-only fast path with K1 ...
```

The user replaces `from trl import GRPOTrainer` with `from kernel_opt import
KernelOptGRPOTrainer` and gets K1 transparently.

### Finding the right method to override

This is detective work. Open `trl/trainer/grpo_trainer.py` in your editor.
Search for the eager-implemented operation you want to replace
(`selective_log_softmax` in our case). Trace upward to find the smallest
method that contains it and is called as a unit. For TRL's GRPO that method
turned out to be `_get_per_token_logps_and_entropies` (lines 1046–1125 in
v1.4) and it's the hub for **all four** logp computations in the trainer
(policy / old / ref-PEFT / ref-non-PEFT). One override = full coverage.

### The non-contiguous-slice trap (real bug we hit)

TRL's loss block does:
```python
logits = model(input_ids).logits         # [B, L, V] contiguous
logits = logits[:, :-1, :]               # [B, L-1, V] -- non-contig in leading dims!
logits = logits[:, -logits_to_keep:, :]  # still non-contig
```

The slicing keeps the *original* dim-0 stride (`L*V`), which doesn't match
the natural stride for the new shape (`(L-1)*V`). Calling `.contiguous()`
on this allocates a fresh `B × L' × V × 2` byte tensor — typically 1+ GB
for GRPO shapes — and OOMs the 4060.

The fix: per-batch indexing. `logits[b]` selects dim 0 and gives
`[L', V]` with strides `(V, 1)`, which is naturally contiguous. Iterate
over `b in range(B)` and call K1 once per batch element. Cost: B small
kernel launches instead of 1 big one (~5-10 µs each, negligible).

The right fix is to extend the kernel to accept a stride parameter. We
deferred that — see `notes/stage4_findings.md`.

## B9. Measurement and profiling

### Why `time.time()` is wrong for CUDA

CUDA kernel launches are **asynchronous**. Python returns immediately after
the launch; the GPU is still working. `time.time()` measures the launch
latency (~5 µs), not the kernel runtime.

### CUDA events

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
my_kernel(...)
end.record()
torch.cuda.synchronize()    # wait for everything to finish
elapsed_ms = start.elapsed_time(end)    # in milliseconds
```

CUDA events are GPU-side timestamps. `.synchronize()` blocks the host until
the GPU is done. `.elapsed_time(other)` returns milliseconds.

### Warmup matters

The first kernel launch is slow because:
- nvcc's PTX gets JIT-compiled to SASS
- L2 cache is cold
- The driver may spin up

Always discard the first 5-10 measurements. The bench harness uses 5 warmup
+ 30 timed iterations and reports the median.

### Effective DRAM bandwidth

```
bandwidth_GBps = bytes_read / latency_seconds / 1e9
```

For K1 forward at (B*S=1024, V=152k, bf16):
```
bytes_read = 1024 × 152064 × 2 = 311 MB
latency    = 1.58 ms
bandwidth  = 311 MB / 1.58 ms = 197 GB/s
% of 256 GB/s peak = 77%
```

This is exactly what `bench/bench_micro.py` computes and prints.

### Memory profiling

```python
torch.cuda.reset_peak_memory_stats()
my_kernel(...)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()    # peak bytes since reset
```

`max_memory_allocated()` returns the high-water mark of *PyTorch-allocated*
memory. Be aware: PyTorch's caching allocator carries reservations across
calls in the same process, so peak alloc may include leftover reservations
from earlier work. To measure cleanly, run in a fresh subprocess.

### Nsight Compute (and why it's hard on Windows)

`ncu` (Nsight Compute CLI) gives per-kernel metrics like
`dram__throughput.avg.pct_of_peak_sustained_elapsed` (the "official"
bandwidth percentage), `sm__warps_active.avg.pct_of_peak_sustained_active`
(occupancy), various stall reasons, and a roofline plot.

Usage:
```
ncu --set full -o my_profile python my_script.py
ncu-ui my_profile.ncu-rep   # open the GUI
```

On Windows consumer GPUs, `ncu` requires admin to access GPU performance
counters (`ERR_NVGPUCTRPERM`). On Linux or WSL2 it works without elevation.

For this project's MVP we used the bench-harness-measured effective
bandwidth (which is mathematically equivalent to ncu's `dram__throughput`
metric) and skipped the screenshots.

### The Roofline model

A roofline plot has two axes:
- X: arithmetic intensity (FLOPs per byte)
- Y: throughput (FLOPs/sec)

The "roof" has two slopes:
- A diagonal line at the slope of peak DRAM bandwidth (left side)
- A horizontal ceiling at peak compute throughput (right side)

The crossover point is the "ridge". A kernel left of the ridge is
memory-bound; right of the ridge is compute-bound. The 4060's ridge:

```
ridge_FLOPs_per_byte = peak_compute / peak_bandwidth
                     = 121 TFLOPs / 256 GB/s
                     ≈ 470 FLOPs/byte
```

K1's arithmetic intensity is ~5 FLOPs/byte. We are ~100× left of the ridge,
firmly memory-bound. **Adding more compute to the kernel cannot make it
faster.** The only path to more throughput is reducing bytes per output.

## B10. What you'll learn next (beyond this project)

This project deliberately stayed in the memory-bound element-wise+reduction
regime. To go further into modern GPU programming, here's what to learn,
roughly in order:

1. **Tensor Cores (mma.sync)** — for GEMM-heavy kernels. Hand-write a
   minimal `mma.sync m16n8k16` BF16 GEMM. Read the CUTLASS docs.
2. **Asynchronous memory: `cp.async` (Ampere+) and TMA (Hopper)** — overlap
   DRAM → shared memory transfers with compute. FlashAttention 2/3 is the
   reference design.
3. **Triton** — Python-language CUDA alternative from OpenAI. Liger-Kernel
   is all Triton. Often gets 90% of hand-written perf with 10% of the code.
   Worth porting one of your existing kernels to as a follow-up.
4. **PTX inline assembly** — for the last 10% of perf. `lop3.b32` for
   bit-tricks, `ldmatrix` for tensor-core feeds, etc.
5. **CUDA Graphs** — record a sequence of kernels once, replay with one
   launch overhead. Critical for low-batch inference (vLLM uses this
   heavily).
6. **NCCL / multi-GPU** — `all_reduce`, `all_gather`, etc. Necessary for any
   real training infrastructure work.

A natural next project: extend K1 to fuse the LM-head linear projection into
the same kernel (Liger-style `FusedLinearCrossEntropy`). That's a real
Tensor Cores + memory-bound combo and would force you to learn (1) and (2).

## B11. Self-test — 20 questions

If you can answer all 20 in 30 seconds each, the material is internalized.
Hover over the link at the end of each for the relevant section.

**Setup & toolchain**
1. Why Python 3.12 and not 3.13? *(A2.3)*
2. Why MSVC 14.29 specifically? *(A2.2)*
3. What does `DISTUTILS_USE_SDK=1` do? *(A12 / scripts/dev_env.bat)*
4. Why is `PYTHONUTF8=1` needed before `import trl`? *(A12)*

**LLM training context**
5. What does GRPO do in 3 sentences? *(B1)*
6. Why is the per-token logprob computation a bottleneck? *(B1)*
7. What's the difference between memory-bound and compute-bound? *(B1, B9)*

**GPU model**
8. What's a warp? Why does it matter? *(B2)*
9. What's the memory hierarchy on a modern NVIDIA GPU? *(B2)*
10. Why does "one block per row" beat "one thread per row" for K1? *(B2, B5)*

**Numerics**
11. Why is bf16 used in training but fp32 used for accumulation? *(B3)*
12. What's the IEEE precision of bf16? Why does it matter for V=152k? *(B3)*

**The K1 algorithm**
13. What's the online logsumexp update rule? *(B4)*
14. Why does naive initialization of (m=-∞, Z=0) give NaN on the first
    iteration? How do we fix it? *(B4)*
15. What's the third accumulator T for, and where does the `(m - m_new) · Z`
    correction come from? *(B4)*
16. Why does the empty sentinel `(−∞, 0, 0)` need explicit handling in
    `combine()`? *(B4)*

**The K1 kernel**
17. Walk through the warp shuffle butterfly reduction. Why 5 steps? *(B5)*
18. Why does the backward kernel not need a reduction? *(B6)*
19. What does `ctx.save_for_backward` do, and what do we save? *(B6)*

**Integration & measurement**
20. Why is `time.time()` wrong for CUDA timing? What do we use instead? *(B9)*

If you got fewer than 15: re-read the corresponding sections. If you got
all 20: you're ready to pick up Tensor Cores or Triton next.

---

## Where to find more

- `PLAN.md` — original project spec and scope decisions
- `REPORT.md` — implementation-depth technical writeup
- `docs/FINAL_REPORT.md` — academic-paper style writeup with related work
- `notes/retrospective_stage1_2.md` — narrative retrospective on stages 1-2
- `notes/stage{1,2,3,4,5}_findings.md` — execution logs per stage, including
  every bug we hit and how we debugged it
- `csrc/fused_logprob.cu` — the kernel itself, ~250 lines, heavily commented
- `python/kernel_opt/ops.py` — the Python autograd wrapper
- `python/kernel_opt/trainer.py` — the TRL subclass
- `tests/test_logprob.py` — 32 tests; reading these is itself a tutorial in
  what kinds of correctness you need to check for a reduction kernel
- `bench/bench_micro.py` — the benchmark harness used throughout

Good luck. Email the author at kaiwenlin@utexas.edu if you find a bug or
have a question.
