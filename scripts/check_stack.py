"""Stage 1 sanity check: import all training-stack libs and confirm CUDA + bf16."""
import torch
import transformers
import trl
import peft
import datasets
import bitsandbytes

for m in [torch, transformers, trl, peft, datasets, bitsandbytes]:
    print(f"{m.__name__:20s} {m.__version__}")

print()
print("cuda available:", torch.cuda.is_available())
print("device:        ", torch.cuda.get_device_name(0))
print("capability:    ", torch.cuda.get_device_capability(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())

# bitsandbytes 8-bit Adam smoke
from bitsandbytes.optim import AdamW8bit
p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
opt = AdamW8bit([p], lr=1e-4)
loss = (p ** 2).sum()
loss.backward()
opt.step()
print("bnb 8bit AdamW step: OK")
