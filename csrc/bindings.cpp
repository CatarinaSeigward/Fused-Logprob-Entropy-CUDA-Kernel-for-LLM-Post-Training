#include <torch/extension.h>

namespace kernel_opt {
void fused_logprob_entropy_naive(
    torch::Tensor logits, torch::Tensor targets,
    torch::Tensor logprob, torch::Tensor entropy, torch::Tensor lse_out);
void fused_logprob_entropy_v1(
    torch::Tensor logits, torch::Tensor targets,
    torch::Tensor logprob, torch::Tensor entropy, torch::Tensor lse_out);
void fused_logprob_entropy_v1_backward(
    torch::Tensor logits, torch::Tensor targets,
    torch::Tensor lse, torch::Tensor entropy,
    torch::Tensor g_logp, torch::Tensor g_ent, torch::Tensor g_lse,
    torch::Tensor d_logits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_logprob_entropy_naive",
          &kernel_opt::fused_logprob_entropy_naive,
          "Naive one-thread-per-row reference (correctness anchor).");
    m.def("fused_logprob_entropy_v1",
          &kernel_opt::fused_logprob_entropy_v1,
          "v1 K1 forward: one-block-per-row, warp+block reduction.");
    m.def("fused_logprob_entropy_v1_backward",
          &kernel_opt::fused_logprob_entropy_v1_backward,
          "v1 K1 backward: streaming, recomputes softmax from saved lse.");
}
