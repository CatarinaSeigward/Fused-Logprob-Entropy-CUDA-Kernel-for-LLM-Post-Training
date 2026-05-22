"""Tiny script for ncu to profile. Calls K1 once at a representative shape."""
import torch
from kernel_opt import fused_logprob_entropy_forward

torch.manual_seed(0)
B, S, V = 1, 1024, 152064  # Qwen vocab, realistic GRPO B*S
logits = torch.randn(B, S, V, device="cuda", dtype=torch.bfloat16)
targets = torch.randint(0, V, (B, S), device="cuda", dtype=torch.int64)

# Warmup so JIT / autotune aren't profiled
for _ in range(3):
    fused_logprob_entropy_forward(logits, targets)
torch.cuda.synchronize()

# The call ncu profiles
out = fused_logprob_entropy_forward(logits, targets)
torch.cuda.synchronize()
print("done", out[0].shape)
