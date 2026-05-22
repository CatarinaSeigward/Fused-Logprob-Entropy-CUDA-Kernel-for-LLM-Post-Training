from . import _C  # noqa: F401
from .ops import (  # noqa: F401
    fused_logprob_entropy,
    fused_logprob_entropy_forward,
    fused_logprob_entropy_naive,
)

# Lazy import so users without trl can still use the kernels.
def __getattr__(name):
    if name == "KernelOptGRPOTrainer":
        from .trainer import KernelOptGRPOTrainer
        return KernelOptGRPOTrainer
    raise AttributeError(f"module 'kernel_opt' has no attribute {name!r}")

__version__ = "0.0.1"
