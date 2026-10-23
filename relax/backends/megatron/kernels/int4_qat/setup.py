# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Get CUDA arch list
# NOTE: This setup script is CUDA-only as it compiles .cu kernel files via CUDAExtension.
# Non-CUDA backends (NPU, XPU, PPU) should provide their own kernel implementations.
env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
arch_list = []
if env_arch:
    # Explicit cross-compilation targets take priority over the build host's GPU.
    arch_list = [a.strip().replace("+PTX", "") for a in env_arch.replace(";", " ").split() if a.strip()]
elif torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        arch_list.append(f"{major}.{minor}")
    arch_list = sorted(set(arch_list))

# Fallback when neither explicit targets nor a build-host GPU are available.
if not arch_list:
    arch_list = ["8.0", "8.6", "8.9", "9.0"]

setup(
    name="fake_int4_quant_cuda",
    ext_modules=[
        CUDAExtension(
            name="fake_int4_quant_cuda",
            sources=["fake_int4_quant_cuda.cu"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-Xcompiler",
                    "-fPIC",
                ]
                + [
                    f"-gencode=arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}"
                    for arch in arch_list
                ]
                + ["-gencode=arch=compute_90a,code=sm_90a"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
