// K1 fused logprob + entropy.
//
// Replaces TRL's `selective_log_softmax(logits, ids)` + `entropy_from_logits(logits)`
// (two separate Python-loop passes over the [B,S,V] logits tensor) with a
// single streaming pass per row.
//
// Two kernels live here:
//   - _naive_kernel : one-thread-per-row reference; correctness anchor; never
//                     fast. Stays for validation.
//   - _v1_kernel    : one-block-per-row, warp-shuffle + shared-mem reduction.
//                     The production K1 forward.
//
// Math (online streaming, numerically stable):
//   For row of length V with running max m, running Z = sum exp(x - m),
//   running T = sum (x - m) * exp(x - m):
//
//     m_new   = max(m, x)
//     delta   = m - m_new            (<= 0)
//     scale   = exp(delta)
//     v_new   = x - m_new            (<= 0)
//     T <- scale * (T + delta * Z) + v_new * exp(v_new)
//     Z <- scale * Z + exp(v_new)
//     m <- m_new
//
//   At the end:  lse = m + log(Z)
//                logprob[t] = logits[t] - lse
//                entropy    = log(Z) - T / Z      (in nats)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace kernel_opt {

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}
template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}
template <>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template <typename T>
__global__ void fused_logprob_entropy_naive_kernel(
    const T* __restrict__ logits,    // [N, V] (N = B * S)
    const int64_t* __restrict__ targets,  // [N]
    float* __restrict__ logprob,     // [N]
    float* __restrict__ entropy,     // [N]
    float* __restrict__ lse_out,     // [N]
    int N,
    int V)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const T* row_ptr = logits + (int64_t)row * V;
    int64_t tgt = targets[row];

    // Peel the first iteration so we don't hit -inf * 0 = nan in the
    // `delta * Z` term when m starts at -infinity.
    float x0 = to_float<T>(row_ptr[0]);
    float m = x0;
    float Z = 1.0f;
    float T_acc = 0.0f;       // (x0 - m) * exp(x0 - m) = 0 * 1 = 0
    float target_logit = (tgt == 0) ? x0 : 0.0f;

    for (int v = 1; v < V; ++v) {
        float x = to_float<T>(row_ptr[v]);
        if (v == tgt) target_logit = x;

        float m_new = fmaxf(m, x);
        float delta = m - m_new;          // <= 0
        float scale = __expf(delta);
        float v_new = x - m_new;          // <= 0
        float exp_v_new = __expf(v_new);

        // T update first (still uses old Z)
        T_acc = scale * (T_acc + delta * Z) + v_new * exp_v_new;
        Z = scale * Z + exp_v_new;
        m = m_new;
    }

    float lse_val = m + __logf(Z);
    logprob[row] = target_logit - lse_val;
    entropy[row] = __logf(Z) - T_acc / Z;
    lse_out[row] = lse_val;
}

// ============================================================================
// v1: block-per-row, warp-shuffle + shared-mem reduction.
// ============================================================================
//
// Combine rule for two partial states (m1, Z1, T1) and (m2, Z2, T2):
//
//     m  = max(m1, m2)
//     d1 = m1 - m,    d2 = m2 - m            (both <= 0)
//     s1 = exp(d1),   s2 = exp(d2)
//     Z  = s1 * Z1   + s2 * Z2
//     T  = s1*(T1 + d1*Z1) + s2*(T2 + d2*Z2)
//
// Empty sentinel: (m=-INFINITY, Z=0, T=0). When combined with anything, the
// scale collapses to 0 *and* the `delta * Z` term is -INFINITY * 0 = NaN, so
// we explicitly suppress the empty side's contribution.
//
// Target logit: handled out-of-band — one thread (the one whose strided
// schedule lands on `tgt`) writes to a shared scalar; everyone reads it
// post-syncthreads. No race because exactly one thread visits each v.

struct PartialState {
    float m;
    float Z;
    float T;
};

__device__ __forceinline__ PartialState combine(PartialState a, PartialState b) {
    if (b.m == -INFINITY) return a;
    if (a.m == -INFINITY) return b;
    float m_new = fmaxf(a.m, b.m);
    float d1 = a.m - m_new;
    float d2 = b.m - m_new;
    float s1 = __expf(d1);
    float s2 = __expf(d2);
    PartialState r;
    r.m = m_new;
    r.Z = s1 * a.Z + s2 * b.Z;
    r.T = s1 * (a.T + d1 * a.Z) + s2 * (b.T + d2 * b.Z);
    return r;
}

template <typename T, int BLOCK_DIM>
__global__ void fused_logprob_entropy_v1_kernel(
    const T* __restrict__ logits,    // [N, V]
    const int64_t* __restrict__ targets,  // [N]
    float* __restrict__ logprob,     // [N]
    float* __restrict__ entropy,     // [N]
    float* __restrict__ lse_out,     // [N]
    int V)
{
    static_assert(BLOCK_DIM % 32 == 0 && BLOCK_DIM <= 1024,
                  "BLOCK_DIM must be a positive multiple of 32, <= 1024");
    constexpr int N_WARPS = BLOCK_DIM / 32;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    const T* row_ptr = logits + (int64_t)row * V;
    int64_t tgt = targets[row];

    // Phase 1: per-thread streaming over strided V.
    PartialState s = {-INFINITY, 0.0f, 0.0f};
    __shared__ float s_target_logit;
    if (tid == 0) s_target_logit = 0.0f;
    __syncthreads();

    for (int v = tid; v < V; v += BLOCK_DIM) {
        float x = to_float<T>(row_ptr[v]);
        if (v == tgt) s_target_logit = x;  // exactly one thread writes
        PartialState elem = {x, 1.0f, 0.0f};   // single-element state at max=x
        s = combine(s, elem);
    }

    // Phase 2a: warp-level reduction via shuffles.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        PartialState other;
        other.m = __shfl_xor_sync(0xffffffff, s.m, offset);
        other.Z = __shfl_xor_sync(0xffffffff, s.Z, offset);
        other.T = __shfl_xor_sync(0xffffffff, s.T, offset);
        s = combine(s, other);
    }

    // Phase 2b: cross-warp reduction via shared memory.
    __shared__ float s_m[N_WARPS], s_Z[N_WARPS], s_T[N_WARPS];
    if (lane == 0) {
        s_m[warp] = s.m; s_Z[warp] = s.Z; s_T[warp] = s.T;
    }
    __syncthreads();

    if (warp == 0) {
        // Only lanes [0, N_WARPS) carry valid data.
        s = (lane < N_WARPS)
            ? PartialState{s_m[lane], s_Z[lane], s_T[lane]}
            : PartialState{-INFINITY, 0.0f, 0.0f};
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            PartialState other;
            other.m = __shfl_xor_sync(0xffffffff, s.m, offset);
            other.Z = __shfl_xor_sync(0xffffffff, s.Z, offset);
            other.T = __shfl_xor_sync(0xffffffff, s.T, offset);
            s = combine(s, other);
        }
        if (lane == 0) {
            float lse_val = s.m + __logf(s.Z);
            logprob[row] = s_target_logit - lse_val;
            entropy[row] = __logf(s.Z) - s.T / s.Z;
            lse_out[row] = lse_val;
        }
    }
}

// ============================================================================
// v1 backward: streaming, one block per row, reuses saved (lse, entropy).
// ============================================================================
//
// Math (derivation in notes/retrospective_stage1_2.md and notes/stage3_findings.md):
//
//   p_v     = exp(x_v - lse)            [softmax, recomputed on the fly from saved lse]
//   log_p_v = x_v - lse
//   d_x[v]  = p_v * (g_lse - g_logp - g_ent * (log_p_v + entropy))
//             + (v == t ? g_logp : 0)
//
// Single streaming pass over V per row. No [B,S,V] intermediate. Same
// memory profile as forward.

template <typename T, int BLOCK_DIM>
__global__ void fused_logprob_entropy_v1_backward_kernel(
    const T* __restrict__ logits,      // [N, V]   re-read in backward
    const int64_t* __restrict__ targets,  // [N]
    const float* __restrict__ lse,     // [N]      saved from forward
    const float* __restrict__ entropy, // [N]      saved from forward
    const float* __restrict__ g_logp,  // [N]      upstream gradient
    const float* __restrict__ g_ent,   // [N]      upstream gradient
    const float* __restrict__ g_lse,   // [N]      upstream gradient
    T* __restrict__ d_logits,          // [N, V]   output, same dtype as logits
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

    // Pre-combine the constant part of the coefficient that doesn't depend on v:
    //   coeff(v) = (g_lse - g_logp - g_ent * entropy) - g_ent * log_p_v
    //           = const_part - g_ent * log_p_v
    float const_part = row_g_lse - row_g_logp - row_g_ent * row_ent;

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

void fused_logprob_entropy_v1_backward(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    torch::Tensor entropy,
    torch::Tensor g_logp,
    torch::Tensor g_ent,
    torch::Tensor g_lse,
    torch::Tensor d_logits)
{
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
    TORCH_CHECK(lse.dtype() == torch::kFloat32 && entropy.dtype() == torch::kFloat32,
                "saved lse / entropy must be fp32");
    TORCH_CHECK(g_logp.dtype() == torch::kFloat32 && g_ent.dtype() == torch::kFloat32 &&
                g_lse.dtype() == torch::kFloat32,
                "upstream gradients must be fp32");
    TORCH_CHECK(g_logp.is_contiguous() && g_ent.is_contiguous() && g_lse.is_contiguous(),
                "upstream gradients must be contiguous");
    TORCH_CHECK(d_logits.dtype() == logits.dtype() && d_logits.is_contiguous() &&
                d_logits.sizes() == logits.sizes(),
                "d_logits must match logits dtype/shape and be contiguous");

    int V = (int)logits.size(-1);
    int64_t N = logits.numel() / V;
    TORCH_CHECK(targets.numel() == N && lse.numel() == N && entropy.numel() == N &&
                g_logp.numel() == N && g_ent.numel() == N && g_lse.numel() == N,
                "leading-dim count mismatch");

    constexpr int BLOCK_DIM = 256;
    int blocks = (int)N;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (logits.dtype() == torch::kBFloat16) {
        fused_logprob_entropy_v1_backward_kernel<__nv_bfloat16, BLOCK_DIM>
            <<<blocks, BLOCK_DIM, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
                targets.data_ptr<int64_t>(),
                lse.data_ptr<float>(), entropy.data_ptr<float>(),
                g_logp.data_ptr<float>(), g_ent.data_ptr<float>(), g_lse.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(d_logits.data_ptr()),
                V);
    } else if (logits.dtype() == torch::kFloat16) {
        fused_logprob_entropy_v1_backward_kernel<__half, BLOCK_DIM>
            <<<blocks, BLOCK_DIM, 0, stream>>>(
                reinterpret_cast<const __half*>(logits.data_ptr()),
                targets.data_ptr<int64_t>(),
                lse.data_ptr<float>(), entropy.data_ptr<float>(),
                g_logp.data_ptr<float>(), g_ent.data_ptr<float>(), g_lse.data_ptr<float>(),
                reinterpret_cast<__half*>(d_logits.data_ptr()),
                V);
    } else if (logits.dtype() == torch::kFloat32) {
        fused_logprob_entropy_v1_backward_kernel<float, BLOCK_DIM>
            <<<blocks, BLOCK_DIM, 0, stream>>>(
                logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
                lse.data_ptr<float>(), entropy.data_ptr<float>(),
                g_logp.data_ptr<float>(), g_ent.data_ptr<float>(), g_lse.data_ptr<float>(),
                d_logits.data_ptr<float>(),
                V);
    } else {
        TORCH_CHECK(false, "logits dtype must be bf16, fp16, or fp32");
    }
}

// ============================================================================
// Host-side dispatch (forward).
// ============================================================================
//
// Accepts logits with arbitrary leading dims; flattens all but the last dim
// to a single batch axis N.

void fused_logprob_entropy_v1(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor logprob,
    torch::Tensor entropy,
    torch::Tensor lse_out)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
    TORCH_CHECK(logprob.dtype() == torch::kFloat32 &&
                entropy.dtype() == torch::kFloat32 &&
                lse_out.dtype() == torch::kFloat32, "outputs must be float32");

    int V = (int)logits.size(-1);
    int64_t N = logits.numel() / V;
    TORCH_CHECK(targets.numel() == N, "targets numel must match leading dims of logits");
    TORCH_CHECK(logprob.numel() == N && entropy.numel() == N && lse_out.numel() == N,
                "output tensor numels must equal leading-dim count");

    constexpr int BLOCK_DIM = 256;
    int blocks = (int)N;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (logits.dtype() == torch::kBFloat16) {
        fused_logprob_entropy_v1_kernel<__nv_bfloat16, BLOCK_DIM>
            <<<blocks, BLOCK_DIM, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
                targets.data_ptr<int64_t>(),
                logprob.data_ptr<float>(), entropy.data_ptr<float>(),
                lse_out.data_ptr<float>(), V);
    } else if (logits.dtype() == torch::kFloat16) {
        fused_logprob_entropy_v1_kernel<__half, BLOCK_DIM>
            <<<blocks, BLOCK_DIM, 0, stream>>>(
                reinterpret_cast<const __half*>(logits.data_ptr()),
                targets.data_ptr<int64_t>(),
                logprob.data_ptr<float>(), entropy.data_ptr<float>(),
                lse_out.data_ptr<float>(), V);
    } else if (logits.dtype() == torch::kFloat32) {
        fused_logprob_entropy_v1_kernel<float, BLOCK_DIM>
            <<<blocks, BLOCK_DIM, 0, stream>>>(
                logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
                logprob.data_ptr<float>(), entropy.data_ptr<float>(),
                lse_out.data_ptr<float>(), V);
    } else {
        TORCH_CHECK(false, "logits dtype must be bf16, fp16, or fp32");
    }
}

void fused_logprob_entropy_naive(
    torch::Tensor logits,        // [..., V]
    torch::Tensor targets,       // [...]
    torch::Tensor logprob,       // [...]
    torch::Tensor entropy,       // [...]
    torch::Tensor lse_out)       // [...]
{
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
    TORCH_CHECK(logprob.dtype() == torch::kFloat32, "logprob must be float32");
    TORCH_CHECK(entropy.dtype() == torch::kFloat32, "entropy must be float32");
    TORCH_CHECK(lse_out.dtype() == torch::kFloat32, "lse must be float32");

    int V = (int)logits.size(-1);
    int64_t N = logits.numel() / V;
    TORCH_CHECK(targets.numel() == N, "targets numel must match leading dims of logits");
    TORCH_CHECK(logprob.numel() == N && entropy.numel() == N && lse_out.numel() == N,
                "output tensor numels must equal leading-dim count");

    int threads = 128;
    int blocks = (int)((N + threads - 1) / threads);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (logits.dtype() == torch::kBFloat16) {
        fused_logprob_entropy_naive_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            logprob.data_ptr<float>(),
            entropy.data_ptr<float>(),
            lse_out.data_ptr<float>(),
            (int)N, V);
    } else if (logits.dtype() == torch::kFloat16) {
        fused_logprob_entropy_naive_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            logprob.data_ptr<float>(),
            entropy.data_ptr<float>(),
            lse_out.data_ptr<float>(),
            (int)N, V);
    } else if (logits.dtype() == torch::kFloat32) {
        fused_logprob_entropy_naive_kernel<float><<<blocks, threads, 0, stream>>>(
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            logprob.data_ptr<float>(),
            entropy.data_ptr<float>(),
            lse_out.data_ptr<float>(),
            (int)N, V);
    } else {
        TORCH_CHECK(false, "logits dtype must be bf16, fp16, or fp32");
    }
}

}  // namespace kernel_opt
