# Stage 1-2 复盘：你到底做了什么 + 需要懂的基础

> 写给项目作者本人。读完应该能：(a) 跟人讲清楚前两周做了什么、为什么这么做；(b) 在面试里被追问任何一个技术决定时不会卡壳；(c) 知道自己还有哪些洞要补。

---

## TL;DR

- **Stage 1**（环境 + 框架锁定）：Windows 上把 nvcc + MSVC + cpp_extension 这条最容易翻车的链路打通，读 TRL 源码确认了 K1 要替换的 baseline 是真的（不是 strawman），跑了一次 stock GRPO 5 步确认 recipe 路径可行。
- **Stage 2**（K1 forward）：写了 naive 单线程 kernel 做正确性锚点，再写了 v1 block-per-row + warp/block reduction kernel 做生产版本。**v1 比 TRL bf16 production path 快 5–9×，达到 82.6% 峰值 DRAM 带宽，零中间内存分配，且数值上比 TRL 更精确**。
- **意外收获**：TRL 的 `entropy_from_logits` 全程在 bf16 累加，输出精度受限到 bf16 的 ~3 位有效数字。我们 fp32 累加更准，这是 fusing 的免费副产品，写进 REPORT。

---

## 第一部分：我们到底做了什么

### Stage 1 的 4 件事

1. **建项目专用 venv**（uv + Python 3.12）。系统 Python 是 3.13，PyTorch wheel 还没全跟上，3.12 是 ML 栈最稳的版本。把 torch / transformers / trl / peft / bitsandbytes / datasets / accelerate 全装进 `.venv`。
2. **打通 Windows 原生 build chain**。这一步差点要切 WSL2。问题：`torch.utils.cpp_extension` 自动找 MSVC 时挑了你机器上一个有问题的 VS 2026 BuildTools（MSVC 14.50），CUDA 12.6 拒绝它（"only VS 2017–2022 supported"）。解法：在同一个 BuildTools 里强制选 14.29 工具集（VS 2019 vintage，CUDA 12.6 兼容），并设 `DISTUTILS_USE_SDK=1` 让 cpp_extension 信任已激活的 VC 环境。固化在 `scripts/dev_env.bat`。
3. **读 TRL 源码确认 K1 卖点是真的**。这是 Stage 1 最重要的一件事——如果 TRL 已经用了 `F.cross_entropy`（PyTorch 自带的融合 cross entropy），K1 就只是去 PK PyTorch 内置 kernel，加速比会很小，整个项目的卖点崩掉。读完发现 TRL 的 bf16 path 是 **Python for 循环逐行 `F.log_softmax` + `gather`**（注释里自己说"slightly less efficient approach"），entropy 是**完全独立**的另一个 chunked pass。两个 pass 都物化了 `[L-1, V]` 的 softmax 中间张量。**K1 有充足 baseline 可打**。
4. **跑 5 步 stock GRPO** 在 Qwen2.5-0.5B + GSM8K，确认 recipe 路径能跑通（虽然 VRAM 顶到 9.4 GB，靠 Windows WDDM 把 1.4 GB 往系统 RAM 溢出。这是 Stage 4 才需要解决的事）。

### Stage 2 的 4 件事

1. **写 naive K1 forward kernel**（`csrc/fused_logprob.cu` 中的 `_naive_kernel`）。一个线程负责一行 logits，串行扫 V 维。**故意写慢**——它的作用是当正确性锚点，永远留在代码库里，让后面的优化版本有得对比。
2. **写校验测试**（`tests/test_logprob.py`，12 个）：vs PyTorch fp32 ground truth、vs TRL bf16 eager、极值不溢出、均匀分布熵 = log(V)。
3. **写 bench harness**（`bench/bench_micro.py`）。4 个对照（v1 / naive / TRL eager / pytorch separate），sweep over (vocab × B*S × dtype)，CUDA event 计时，输出 CSV + 终端 pretty table。**这个 harness 是 Stage 2/3 任何 perf 改动的统一裁判**——后面每个版本都用同一套 shape grid 测，谁也不能耍滑。
4. **写 v1 优化 kernel**（`_v1_kernel`）：
    - 一个 CUDA block 负责一行（不是一个 thread）；block_dim=256
    - 每个 thread 在 V 维上 strided 扫，本地累加 `(m, Z, T)` 三元组
    - Warp 内用 `__shfl_xor_sync` 蝶式归约
    - 跨 warp 用 shared memory 二级归约
    - Target logit 通过 shared scalar 广播
    - 整个过程**单流式 pass over logits**，零中间内存

写完跑 bench：在 (1, 1024, 152k) bf16 形状下 v1 是 1.58 ms / TRL 是 13.6 ms / pytorch separate 是 11.1 ms。**v1 8.6×。**

---

## 第二部分：关键 tradeoff 清单

每条都是"我们做了 A，没做 B，因为 C"。面试如果被追问，按这个表答。

| 决定 | 没做的 | 为什么 |
|---|---|---|
| **uv 建 venv，Python 3.12** | conda / 系统 Python 3.13 | conda 慢且笨重；3.13 没 bitsandbytes Windows wheel；3.12 是 ML 栈最稳版本 |
| **Windows 原生 build（MSVC 14.29）** | 切 WSL2 | 跑通了就不用切；切 WSL 就要重装整个 toolchain，再花一天。**有"2 小时不通就切"的硬时间盒** |
| **setuptools + cpp_extension** | CMake | PyTorch BuildExtension 自动处理 MSVC/nvcc flag，CMake 在 Windows 上跟 nvcc 配会再花一天 |
| **K1 输出 (logprob, entropy, lse) 三个张量** | 只输出 logprob | (a) entropy 几乎免费（多一个累加器）；(b) TRL 单独算 entropy 是大头开销，融合掉是真节省；(c) lse 是 backward 必需的 saved value |
| **block-per-row, block_dim=256** | thread-per-row（naive） | naive 在 V=128k 时一个线程要串行 128k 次 expf，4060 36k 个并发线程容量根本喂不饱；block-per-row 把单行 V 维并行化 |
| **256 threads/block** | 512 / 1024 | 256 = 8 warp；shared mem 占用小、occupancy 高；后续可以 sweep 但 MVP 不做 |
| **没做 vectorized 8-element bf16 loads** | `int4` aligned vector loads | 已经到 82.6% peak DRAM bw，PLAN exit 标准是 ≥80%。再优化是边际收益递减。MVP 不做 |
| **fp32 累加 (m, Z, T)** | bf16 累加 | bf16 只有 8 位 mantissa（~3 位有效数字），累加 V=152k 个数后误差爆炸。fp32 几乎免费且准确 |
| **MVP K1 backward 用 saved lse 流式重算** | 物化 softmax | PLAN 里的设计选择；流式重算同样的内存 profile。如果 Stage 3 翻车再降级到物化版本（cut-line 已写在 PLAN） |
| **K1 输入接受 bf16/fp16/fp32**（template） | 只支持 bf16 | 模板成本极小；fp32 路径用于 ground-truth 测试，fp16 给可能用 fp16 训练的人 |
| **保留 naive kernel 在生产代码里** | 删掉 | 0 维护成本；它是后续任何"v1 出 bug 了"调试时的对照参照物 |
| **bench 跳过 >6 GB 估算的 shape** | 强行跑然后 OOM | OOM 后 CUDA context 状态可能脏，影响后续测量；保守跳过更稳 |

---

## 第三部分：基础学习者需要知道的基础知识

按"理解我们这个 kernel 必需"的顺序排。每一节都包含：**概念** → **为什么这个项目需要它** → **进一步看什么**。

### 1. Memory-bound vs Compute-bound（最重要的心智模型）

**概念**：GPU kernel 的瓶颈分两种——
- **Compute-bound**：算术指令是瓶颈。比如大矩阵乘法，每个数据被反复用，算的多读的少。
- **Memory-bound**：从 DRAM 读数据是瓶颈。比如 element-wise 加法、归约，每个数据只用一次，读完就丢。

判断方法：算 **arithmetic intensity** = 浮点操作次数 / 字节数。AI = ops / bytes。
- 如果 AI 低（每读 1 字节只算 < 几个操作），是 memory-bound
- 如果 AI 高（每读 1 字节算 > 几十个操作），是 compute-bound

**为什么这个项目需要**：K1 是典型 memory-bound——读一行 V 个 bf16 = 2V 字节，做 ~10V 个浮点操作（max, exp, mul, add, exp, ...），AI ≈ 5 ops/byte。RTX 4060 的算力 ≈ 121 TFLOPs bf16 / 256 GB/s = ~470 ops/byte 才算 compute-bound。我们离那个数 100× 远，所以**目标不是算的快，是把内存带宽吃满**。

这就是为什么"82.6% 峰值 DRAM bw"是核心 KPI 而不是"减少了多少 FLOPs"。

**进一步**：Roofline 模型（一张图把这两种 bound 画出来）。Williams et al. 2008 那篇原 paper，或者搜 "roofline model GPU"。

### 2. DRAM 带宽 = kernel 质量天花板

**概念**：GPU 显存（GDDR6 / HBM）有物理带宽上限。RTX 4060 mobile = 256 GB/s。任何 memory-bound kernel 跑得再快，最快也就是"输入字节数 / 带宽"。

**为什么这个项目需要**：
- bench 里我们算 `bandwidth_gbps = bytes_read / latency`，跟 256 GB/s 比，得到"% 峰值"
- 这是 kernel 工程师的"达标线"——80% 是合格，90%+ 是好，95% 是天花板（无法 100% 因为有调度 overhead）
- naive 跑 12% peak → 说明 kernel 写错了（一个线程喂不饱 GPU），不是硬件问题

**关键洞察**：如果你的 kernel 已经到了 ~90% peak DRAM bw，**继续优化算法逻辑没用**——你被物理硬件卡住了。剩下的优化要么是减少要读的字节数（比如 fusing 让中间结果留在寄存器），要么是用更高级的硬件（HBM、L2 cache、TMA）。

**进一步**：`ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed` 直接读这个数。

### 3. Online logsumexp（K1 的算法核心）

**问题**：要算 `log(sum(exp(x_i)))`，朴素写法 `log(sum(exp(x)))` 在 x_i > 88 时溢出（fp32 max ≈ e^88）。

**经典解法**：减去 max。
```
m = max(x)
lse = m + log(sum(exp(x_i - m)))
```
现在 `x_i - m ≤ 0`，`exp(...) ≤ 1`，不会溢出。但需要**两遍**扫 x：一遍找 max，一遍算 sumexp。

**Online 版本**（流式，一遍）：维护"截至当前的 (m_n, Z_n)"，新元素来时增量更新：
```
m_{n+1} = max(m_n, x)
Z_{n+1} = exp(m_n - m_{n+1}) * Z_n + exp(x - m_{n+1})
```
最终 `lse = m + log(Z)`。

**为什么这个项目需要**：每行 V=152k 个数，**两遍扫 = 两遍读 DRAM**——直接砍掉一半带宽预算。Online 一遍扫，吃满硬件带宽。这就是 FlashAttention 用的同一个 trick（softmax 也是 normalize by sumexp）。

**进一步**：FlashAttention paper（Dao 2022）section 3.1 有一模一样的推导。我们 K1 的 `(m, Z, T)` 三元组是把熵也加进同一套 online 框架。

### 4. Online entropy（我们自己推的部分）

熵公式：`H = -sum(p_i * log p_i) = log(Z) - (1/Z) * sum((x_i - m) * exp(x_i - m))`

引入第三个累加器 **T = sum (x_i - m) * exp(x_i - m)**。和 Z 一样需要 m 变化时 rescale：

```
m_new = max(m, x)
delta = m - m_new       # ≤ 0
v_new = x - m_new       # ≤ 0
T_new = exp(delta) * (T + delta * Z) + v_new * exp(v_new)
Z_new = exp(delta) * Z + exp(v_new)
```

**最关键的细节**：`delta * Z` 这一项。当 m_new 比 m_old 大时，所有旧元素的相对位置 (x_i - m) 都向下平移了 |delta|，所以原来的 T 既要被 `exp(delta)` 缩放（因为 exp 系数变了），还要补一个 `delta * Z` 项（因为乘的那个 (x_i - m) 也变了）。

**初始化坑**：`m = -inf`, `Z = 0`, `T = 0` 看似自然，但首次 `delta = -inf`, `delta * Z = -inf * 0 = NaN`。**修法**：剥离首次迭代——直接用第一个元素初始化 `m = x_0, Z = 1, T = 0`，从第二个开始 online update。我在 `_naive_kernel` 第一版踩了这个坑，跑出全 NaN，三秒就改掉了，但这种 -inf × 0 的 NaN 问题在很多 streaming reduction kernel 里都会冒出来，记住这个 pattern。

### 5. 为什么 fp32 累加 bf16 输入：精度

**bf16 格式**：1 sign + 8 exp + 7 mantissa。和 fp32 同等指数范围，但只有 8 位精度（≈ 3 位十进制）。

**问题**：把 152k 个 bf16 数累加，每加一次相对误差 ~2^-8 ≈ 0.4%，累加 152k 次后误差爆炸到几个数量级。

**解法**：读进来是 bf16，**进 register 后立刻转 fp32**，整个累加用 fp32，只在最后写回时考虑是否回 bf16。fp32 mantissa 有 23 位，加 152k 个数误差 ~ 152k * 2^-23 ≈ 1.8%——还好。

**这就是我们为什么比 TRL 准**：TRL 的 `entropy_from_logits` 全程 bf16，输出值都是 bf16 量化的（8.5000, 8.5625...）。我们 fp32 中间精度高一个数量级。**这是 fusing 的免费 bonus，因为我们本来就要在寄存器里聚合中间值**——既然在寄存器里，就用 fp32 装。

### 6. GPU 执行模型（你已经懂，但要确认这几个数）

复习一遍，确保概念干净：
- **Thread**：最小执行单位
- **Warp**：32 个 thread 锁步执行（lock-step），所有 warp 内 sync 操作（shfl, ballot）只在这 32 个之间
- **Block**：1-1024 个 thread 的集合，共享一块 shared memory，可以 `__syncthreads()`。一个 block 跑在一个 SM 上
- **Grid**：所有 block 的集合，跨 block **不能**直接通信

**RTX 4060 关键数字**（背下来，面试有用）：
- 24 SMs × 1536 active threads / SM = ~36k concurrent threads
- 每 SM 100 KB shared mem
- 32 MB L2 cache
- 256 GB/s DRAM bandwidth
- 121 TFLOPs bf16 (with Tensor Cores; 没用 TC 的话 ~30 TFLOPs)

**为什么 naive 慢**：naive 启动 B*S 个 thread。当 B*S=256 时只有 256 个 thread 在跑，4060 能并发 36k 个，**利用率 0.7%**。block-per-row 启动 256 block × 256 thread = 65k thread，把 SMs 全部喂满。

### 7. Warp shuffle 归约（蝶式 / butterfly）

**指令**：`__shfl_xor_sync(mask, value, lane_offset)`。在一个 warp 的 32 个 thread 之间交换寄存器。无需 shared memory，无需 sync，零开销。

**蝶式归约模式**（reduce 32 个值到 1 个）：
```
for offset in [16, 8, 4, 2, 1]:
    other = __shfl_xor_sync(0xffffffff, mine, offset)
    mine = combine(mine, other)
```
每一步两两配对。5 步后所有 32 个 lane 都拿到全 warp 的归约结果。

**为什么这个项目需要**：v1 kernel 每个 thread 维护本地 `(m, Z, T)`，需要把 256 个 thread 的 `(m, Z, T)` 归约成 1 个。先 warp 内 shfl 归约（32→1，5 步），再跨 warp 用 shared mem。

**关键**：`combine` 操作必须**结合（associative）**，否则 reduce order 影响结果。online (m, Z, T) 的 combine 是结合的——证明在 FlashAttention paper 附录里。

### 8. 跨 warp 归约 via shared memory

8 个 warp 各自 reduce 完后，每个 warp 的 lane 0 把本地结果写到 shared memory，然后 `__syncthreads()`，再让 warp 0 把这 8 个结果再 shfl 归约一次。

```cpp
__shared__ float s_m[N_WARPS], s_Z[N_WARPS], s_T[N_WARPS];
if (lane == 0) { s_m[warp] = m; s_Z[warp] = Z; s_T[warp] = T; }
__syncthreads();

if (warp == 0) {
    // 重新读、再 reduce
    ...
}
```

**注意**：第二轮 reduce 时只有前 8 个 lane 有数据（因为只有 8 个 warp），其余 lane 用"empty 哨兵"填充。我们用 `(m=-INFINITY, Z=0, T=0)` 作为 empty。

### 9. Empty 哨兵的 combine 问题（kernel 工程容易踩的坑）

`combine(real, empty)` 应该返回 real。但用 (m=-inf, Z=0, T=0) 当 empty，公式里 `delta * Z = -inf * 0 = NaN`。

**修法**（我们 kernel 里的写法）：在 `combine` 函数顶上加两个早返：
```cpp
if (b.m == -INFINITY) return a;
if (a.m == -INFINITY) return b;
```

很简单但**面试常考**——题目类似"你怎么处理 reduce 边界？"。

### 10. CUDA event 计时（不是 Python time）

**为什么不用 `time.time()`**：CUDA kernel 是异步启动的，Python 调用 `kernel_launch()` 立刻返回，实际计算还在后台跑。`time.time()` 测的是 launch 开销不是计算时间。

**正确做法**：
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out = my_kernel(...)
end.record()
torch.cuda.synchronize()  # 等所有 kernel 跑完
elapsed_ms = start.elapsed_time(end)
```

**Warmup**：第一次跑 kernel 通常慢（JIT、cache miss、driver init），跑 5-10 次 warmup 再测，取中位数。

我们 `bench_micro.py` 里就是这套：5 warmup + 30 iter，取中位数。

### 11. ncu / Nsight Compute（你还没用，但 Stage 5 要用）

NVIDIA 出的 kernel profiler。用法：
```
ncu --set full -o out_report python my_script.py
```
然后用 `ncu-ui` 打开 `out_report.ncu-rep` 看图形界面。

**面试时最重要的几个 metric**：
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — 你这个 kernel 跑到了峰值 DRAM 带宽的百分之多少（**最重要**）
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy
- `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed` — ALU 占用
- `smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed` — long scoreboard stall（一般是等 DRAM）

**roofline plot**：ncu-ui 自带一个 roofline 视图，把你的 kernel 标在"compute bound vs memory bound"图上，一眼看出在哪条线下面。

我们 Stage 5 会跑 ncu 截图，放进 REPORT。

### 12. PyTorch cpp_extension 的工作原理

`torch.utils.cpp_extension.CUDAExtension` 是个 setuptools 扩展，做 4 件事：
1. 找 nvcc 和 host C++ 编译器（Windows 是 MSVC 的 cl.exe）
2. 编 .cu / .cpp 成 .pyd / .so
3. 链接 PyTorch 的 C++ 库（libtorch）
4. 注册 PyBind11 module

**Windows 上脆**：找 MSVC 用 vswhere，挑了错的；nvcc 跟 MSVC 版本要匹配（CUDA 12.6 ↔ MSVC 14.41 及以下）；要设 `DISTUTILS_USE_SDK=1` 否则它以为 VC env 没激活。我们用 `dev_env.bat` 一次性解决这堆。

**Linux 上几乎免配**：apt install build-essential 就行。这就是为什么 PLAN 里把 WSL2 fallback 列为 Stage 1 必备硬时间盒。

---

## 第四部分：你目前还没完全懂的，未来会需要的

诚实地列：

1. **PTX inline assembly**。我们没用，但优化到 90%+ 时常需要手写 `lop3`、`ldmatrix`、`mma.sync` 等。Marlin / FlashAttention 大量用。
2. **Tensor Cores**。我们整个项目避开了 Tensor Cores（这是 PLAN 故意的取舍——节省 1-2 周学习时间）。但简历上面试官几乎一定会问。建议项目结束后用一个周末写个最小 mma.sync GEMM 玩一下。
3. **`cp.async`** / **TMA**（Hopper 才有）：异步 DMA 把 DRAM → shared mem 的延迟和计算重叠。Hopper / Blackwell 上现代 kernel 必用。我们 K1 是 element-wise 流式，没用上，但 GEMM 类必用。
4. **Triton**。Python 写 kernel，编出来速度接近手写 CUDA。Liger-Kernel 全用 Triton。**我建议 Stage 6 后用 Triton 重写 K1 一次**，对照速度，作为 stretch 目标。能讲两套技术栈是面试加分。
5. **CUDA Graphs**。把多个 kernel launch 录制成 graph，重放时省掉 launch overhead。vLLM / TRT-LLM 重度依赖。我们 PLAN 里把 Graph 兼容性列为 stretch。
6. **NCCL / 通信**。多卡训练必须懂。我们项目单卡，跳过。

---

## 第五部分：现在你能干净地讲清楚的问题（自检）

如果以下问题你都能 30 秒内答出来，Stage 1+2 就内化了：

1. K1 替换 TRL 的什么函数？为什么这俩值得 fuse？
2. 为什么 naive kernel 慢？（"thread per row 不喂满 GPU"）
3. 为什么 v1 快？（"block per row + warp/block reduction，单流式 pass，82% peak DRAM bw"）
4. Online logsumexp 是什么？为什么需要它？
5. `delta * Z` 是干嘛的？为什么会出 -inf × 0 = NaN？怎么修？
6. 为什么 bf16 输入但 fp32 累加？
7. Memory-bound vs compute-bound 怎么判断？K1 是哪种？
8. RTX 4060 的峰值 DRAM 带宽是多少？我们跑到了百分之多少？
9. Warp shuffle 归约的步数是几？为什么是 5 步？
10. 跨 warp 归约怎么做？为什么需要 shared memory？

如果有答不出的，回去翻对应代码 + 这份文档对应小节。

---

## 下一步预告（Stage 3 在干什么）

K1 backward。挑战在两点：
- **数学**：cross-entropy backward = `softmax - onehot(target)`，但我们没物化 softmax。要用 forward 时存的 `lse`，第二次流式扫 logits 时**重算** softmax 的逐元素值，乘上上游梯度，写回 logits 的梯度。
- **工程**：把 forward+backward 包进 `torch.autograd.Function`，PyTorch 才能在反传时自动调到我们的 backward。再用 `torch.library.custom_op` 注册（带 `meta` kernel for shape inference），`torch.compile` 才能正确处理我们这个 op。

完事后是 Stage 4：把 K1 塞进 `KernelOptGRPOTrainer(trl.GRPOTrainer)` 子类，跑一步 GRPO 验证 loss 跟 stock TRL 对得上。
