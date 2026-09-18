# NPU 训练指导

## 概述

本文档介绍了在华为昇腾的算力节点上使用Relax训练框架，对业界主流的开源三方大模型进行训练的详细过程。训练使用的算力资源是910C。

## 模型支持

| 模型            | 训练场景 | Sync | Async | 训练所需最小卡数 | 参考脚本                                                      |
| --------------- | -------- | ---- | ----- | ---------------- | ------------------------------------------------------------- |
| Qwen3-4B        | DAPO     | √    | √     | 910C 2卡         | `scripts/training/text/run-qwen3-4B-4xnpu-colocate.sh`        |
| Qwen3.5-9B      | DAPO     | √    | √     | 910C 2卡         | `scripts/training/text/run-qwen35-9B-4xnpu-colocate.sh`       |
| Qwen3.5-35B-A3B | DAPO     | √    | √     | 910C 8卡         | `scripts/training/text/run-qwen35-35B-A3B-16xnpu-colocate.sh` |

## 特性支持

| 模型            | 训练场景 | 多模态 | MTP | CP  | 训练所需最小卡数 | 参考脚本                                                       |
| --------------- | -------- | ------ | --- | --- | ---------------- | -------------------------------------------------------------- |
| Qwen3.5-9B      | DAPO     | -      | -   | √   | 910C 8卡         | `scripts/training/text/run-qwen35-9B-16xnpu-cp.sh`             |
| Qwen3.5-9B      | DAPO     | -      | √   | -   | 910C 4卡         | `scripts/training/text/run_qwen35_9B_mtp_8xnpu_thd.sh`         |
| Qwen3.5-35B-A3B | SFT      | √      | √   | -   | 910C 4卡         | `scripts/training/sft/run_qwen35-35B-pokemon-sft-mtp-8xnpu.sh` |

## 环境准备

### 前置准备

- 资源类型：`Ascend910 Snt9b23`
- 驱动版本：`Software Version 25.5.1`
- 固件版本：`Firmware Version 7.8.0.6.201`
- 基础镜像：`quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11`

### 环境检查

`npu-smi info`                    # 在每个实例节点上运行此命令可以看到NPU卡状态

`npu-smi info -l | grep Total`    # 在每个实例节点上运行此命令可以看到总卡数，用来确认对应卡数已经挂载

`npu-smi info -t board -i 1 | egrep -i "software|firmware"`   #查看驱动和固件版本

### 安装方法

`docker/Dockerfile.npu` 采用多阶段构建：

- `base`：系统依赖 + CANN 工具链层；
- `train`：安装 torch_npu、MindSpeed、Megatron、sglang-npu 和 sgl-kernel-npu；
- `relax`：安装 Relax 的 Python 运行时依赖，不复制或安装 Relax 源码。训练任务运行时挂载目标分支代码。

通过 Makefile 一次构建并推送完整 Ascend 镜像，标签形如 `ascend-dev-YYYYMMDD-<hash8>`：

```bash
REGISTRY=<registry> make docker-ascend
```

910C DinD 使用传统 builder 时：

```bash
REGISTRY=<registry> ASCEND_DOCKER_BUILDKIT=0 make docker-ascend
```

可配置变量：`BASE_IMAGE`（默认 `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11`）、`SOC_VERSION`（默认 `ascend910_9391`）、`REGISTRY`、`DO_PUSH`（默认 `1`，设为 `0` 时不推送并检查本地镜像）、`ASCEND_DOCKER_BUILDKIT`（默认 `1`）。远端已存在同名镜像时会跳过构建。

（可选，内部 QS 镜像）复用 `ml-engine/tools/relax-ci` 的 `docker/Dockerfile.qs`（纯 Python 依赖，架构无关，ARM64 可直接构建）。由 CI 先 checkout relax-ci，再指向其 `Dockerfile.qs`：

```bash
# 需先 checkout relax-ci，ASCEND_QS_DOCKERFILE 指向其 docker/Dockerfile.qs
REGISTRY=<registry> \
  ASCEND_QS_DOCKERFILE=<relax-ci>/docker/Dockerfile.qs \
  ASCEND_QS_BASE_IMAGE=<registry>/relax:ascend-dev-YYYYMMDD-<hash8> \
  make docker-qs-ascend
```

## 启动配置

### 容器启动

- 启动命令：

```
export work_dir="自定义挂载的工作目录"     # 容器内挂载的目录，如果挂载SFS可使用挂载目录
export container_work_dir="自定义挂载到容器内的工作目录"
export container_name="自定义容器名称"
export image_name="镜像名称"

docker run \
--shm-size=200gb   --cap-add=SYS_PTRACE   \
--device=/dev/davinci0   \
--device=/dev/davinci1   \
--device=/dev/davinci2   \
--device=/dev/davinci3   \
--device=/dev/davinci4   \
--device=/dev/davinci5   \
--device=/dev/davinci6   \
--device=/dev/davinci7   \
--device=/dev/davinci8   \
--device=/dev/davinci9   \
--device=/dev/davinci10   \
--device=/dev/davinci11   \
--device=/dev/davinci12   \
--device=/dev/davinci13   \
--device=/dev/davinci14   \
--device=/dev/davinci15   \
--device=/dev/davinci_manager   \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc   \
--name ${container_name}   \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info   \
-v /var/log/npu/:/usr/slog   \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v ${work_dir}:${container_work_dir} \
-u root \
-itd ${image_name}  /bin/bash
```

- 参数说明：

  > ⦁	--shm-size：表示共享内存，用于多进程间通信

  > ⦁	--device=/dev/davinciX：表示容器挂载的卡号

  > ⦁	driver及npu-smi 需同时挂载至容器

### 训练启动

- 启动命令：

```
# Fully async mode
bash scripts/training/text/run-qwen3-4B-8xgpu-async-npu.sh
```

- 脚本说明：

  > ⦁	环境变量：
  > ASCEND_RT_VISIBLE_DEVICES #指定卡号
  > HCCL_NPU_SOCKET_PORT_RANGE #配置HCCL在NPU侧使用的通信端口
  > HCCL_HOST_SOCKET_PORT_RANGE #配置HCCL在Host侧使用的通信端口
  > PYTORCH_NPU_ALLOC_CONF #优化内存使用，减少碎片化
  > EXP_DIR #模型、数据集地址

  > ⦁	环境入口：
  > source entrypoint：/root/Relax/scripts/entrypoint/local-npu.sh

  > ⦁	启动配置：
  > PERF_ARGS，启用 recompute（重计算） 以节省显存，同时改用动态 batch size 自适应调整。`--recompute-granularity full/--recompute-method uniform/--recompute-num-layers 1/--use-dynamic-batch-size`

  > OPTIMIZER_ARGS，内存优化，优化器状态 offload 到 CPU 并重叠通信以隐藏延迟。`--optimizer-cpu-offload/--overlap-cpu-optimizer-d2h-h2d/--use-precision-aware-optimizer`

  > SGLANG_ARGS，硬件适配：切换device为npu `--sglang-device npu`、attention backend 为 ascend（华为自研加速库）`--sglang-attention-backend ascend`、禁用 radix cache `--sglang-disable-radix-cache`;

  > SGLANG_ARGS，性能优化：开启数据并行（DP）优化 lm_head 和 attention `--sglang-enable-dp-lm-head/--sglang-enable-dp-attention`、配置 CUDA graph 预热 batch sizes `--sglang-cuda-graph-bs 4 8 16 32 64 96 128`

  > MISC_ARGS，显示启用FlashAttention实现 `--use-flash-attn`

### MTP 特性说明

开启 MTP 训练时，`MTP_NUM_LAYERS`（MTP 头层数，脚本透传给训练参数 `--mtp-num-layers`）只能为 `1`：由于 Qwen3.5 原始 checkpoint 中仅包含 1 层 MTP 权重（`mtp_num_hidden_layers=1`），MTP 相关启动脚本（`scripts/training/text/run_qwen35_9B_mtp_8xnpu_thd.sh`、`scripts/training/sft/run_qwen35-35B-pokemon-sft-mtp-8xnpu.sh`）已加入校验，当该参数被设置为非 `1` 的值时脚本会报错退出

## 下一步

- [ ] 性能优化：Qwen3.5-35B-A3B 等
