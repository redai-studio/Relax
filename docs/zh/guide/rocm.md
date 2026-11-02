# AMD ROCm

Relax 可以通过基于 `docker/Dockerfile.rocm` 构建的 ROCm Docker 镜像运行在 AMD Instinct GPU 上。训练脚本、启动器和配置方式与 NVIDIA GPU 完全相同，区别只在于镜像提供了一套 ROCm 版本的底层软件栈：

- **基础镜像**：SGLang 官方 ROCm 10 镜像（Ubuntu 24.04、ROCm 10 SDK、ROCm 版 PyTorch 2.11、aiter kernel），按 digest 固定版本。
- **SGLang 0.5.21**：与 CUDA 13 镜像使用相同的 Relax 补丁、Rust radix-tree cache 和 DP-aware router，并额外应用 `docker/patch/rocm/` 中的一项 ROCm 修复。
- **Megatron**：使用与 CUDA 13 镜像相同的 Megatron-Bridge commit 和补丁序列构建。
- **Transformer Engine、apex 和 flash-attn**：使用预编译的 ROCm wheel，构建时会校验 SHA-256 摘要。

## 支持的 GPU

构建参数 `GPU_ARCH` 用于选择 GPU 架构，每个取值对应一个 SGLang 基础镜像和一组预编译 wheel。

| `GPU_ARCH` | GPU | SGLang 基础镜像 |
| --- | --- | --- |
| `gfx950`（默认） | MI350X、MI355X | `lmsysorg/sglang:v0.5.21-rocm10-mi35x` |
| `gfx942` | MI300X、MI325X | `lmsysorg/sglang:v0.5.21-rocm10-mi30x` |

两个目标由同一个 Dockerfile 构建。`gfx942` 变体尚未在 MI300 系列硬件上验证，用于长时间任务之前，建议先做一次短时间的试运行。

## 构建镜像

请在仓库根目录下执行构建。构建过程不需要 GPU，但需要从 GitHub、PyPI 和 Rust 工具链服务器下载源码和软件包，最终镜像约 50 GB。

```bash
cd Relax

# MI350X / MI355X
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t relax:dev-rocm-gfx950 .

# MI300X / MI325X
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm --build-arg GPU_ARCH=gfx942 -t relax:dev-rocm-gfx942 .
```

如果构建机需要通过代理访问网络，可以使用 Docker 标准的代理参数（`--build-arg HTTP_PROXY=...`、`HTTPS_PROXY`、`NO_PROXY`）。

如果某个预编译 wheel 与记录的摘要不一致，构建会在 `sha256sum -c SHA256SUMS` 这一步失败。这说明上游发布的文件已被替换；请先审查新文件，再更新 `docker/Dockerfile.rocm` 中的摘要。

## 启动容器

挂载代码仓库以及模型和数据目录，并把 ROCm 设备传入容器：

```bash
docker run -it --init \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc host --network host \
  --security-opt seccomp=unconfined \
  -v $(pwd)/Relax:/root/Relax \
  -v /path/to/models_and_data:/root/exp \
  relax:dev-rocm-gfx950 /bin/bash
```

`/dev/kfd`、`/dev/dri` 加上 `video` 和 `render` 用户组，使容器能够访问 AMD GPU，作用相当于 NVIDIA 环境下的 `--gpus all`。`--ipc host` 为 PyTorch 和 colocate 训练提供进程间共享内存，`--network host` 让 Ray 和 SGLang 引擎直接使用宿主机网络。`--init` 会启动一个轻量的 init 进程，回收已退出的 Ray worker；否则长时间运行的任务中会不断积累 defunct 进程。

在 Apptainer 沙箱中运行 agent 工具的示例（例如 [Mini-SWE-Agent](https://github.com/redai-studio/Relax/tree/main/examples/mini_swe_agent)）还需要加上 `--privileged`，以便 Apptainer 挂载其镜像。

进入容器后，确认 PyTorch 能识别到 GPU：

```bash
rocm-smi
python -c "import torch; print(torch.cuda.device_count(), torch.version.hip)"
```

ROCm 版 PyTorch 沿用 `torch.cuda` API，因此第一个数字是 GPU 数量，第二个是 HIP 版本。

## 运行训练

使用与[快速上手](./quick-start.md)相同的脚本和环境变量启动训练。例如，Qwen3-4B 模型和 DAPO-Math 数据位于 `/root/exp` 时：

```bash
cd /root/Relax
export EXP_DIR=/root/exp
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

镜像中已在 `/root/Megatron-LM` 预置打好补丁的 Megatron 代码，这也是 `scripts/entrypoint/local.sh` 中 `MEGATRON` 的默认路径，因此无需额外设置路径。本地启动器默认使用 8 张 GPU，如需改变，请设置 `NUM_GPUS`。

## ROCm 运行时设置

镜像会设置以下环境变量。除非正在调试它们所控制的行为，否则请保留这些设置：

| 变量 | 取值 | 作用 |
| --- | --- | --- |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `1` | ROCm 7.2 及以上版本通过 dma-buf 在进程间共享显存，在 colocate 训练中会导致导出的显存泄漏。使用 legacy IPC 可以避免该问题。 |
| `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` | `0` | 让不申请 GPU 的 Ray actor 仍能看到 GPU，避免这些 actor 中的设备检查失败。 |
| `SGLANG_RUST_BUILD_MODE` | `never` | 直接使用镜像中已编译好的 Rust radix-tree cache，而不是在每个新容器中重新编译。 |

SGLang 基础镜像还通过 `SGLANG_USE_AITER=1` 启用了 aiter kernel。
