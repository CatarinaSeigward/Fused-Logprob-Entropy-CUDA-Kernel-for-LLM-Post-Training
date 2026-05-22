from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-gencode=arch=compute_89,code=sm_89",
    "--ptxas-options=-v",
]

cxx_flags = ["/O2", "/std:c++17"]

ext_modules = [
    CUDAExtension(
        name="kernel_opt._C",
        sources=[
            "csrc/bindings.cpp",
            "csrc/fused_logprob.cu",
        ],
        include_dirs=["csrc/include"],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    ),
]

setup(
    name="kernel_opt",
    version="0.0.1",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
