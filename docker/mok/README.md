# Megatron + MoK 增量镜像

在 CUDA 13 Blackwell 训练镜像上增加 Megatron MoK 适配、640 维尾块支持和调度器优化。MoK wheel 在增量镜像内编译。

## 构建

从 Relax 仓库根目录构建 `docker/Dockerfile.cu13` 的 Blackwell 镜像，再用它作为 `BASE_IMAGE`。已有相同源码构建的镜像时，可以直接使用对应的 digest。

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.cu13 --target relax \
  --build-arg GPU_ARCH=blackwell -t relax-cu13-blackwell:f78809f6 .
docker build --platform linux/amd64 --build-arg BASE_IMAGE=relax-cu13-blackwell:f78809f6 \
  -t relax-mok:pr10-sm103 docker/mok
```

增量构建上下文为 `docker/mok`。基底提供 Python 3.12、Torch `2.13.0+cu130`、CUDA 13.0 NVCC、TE 2.18、Bridge、HybridEP、Git、Make 和 patch。`PYTHON_BIN` 默认指向解释器目录 `/opt/sglang/bin`，`MEGATRON_ROOT` 默认指向 `/root/Megatron-LM`。MoK 目标架构为 Blackwell `SM103`。

构建先校验 CUDA 13 基底中 Megatron 补丁涉及的 7 个文件的 SHA256，用 `--fuzz=0` 应用补丁，再校验结果。基底为 `f78809f6` 应用 `Dockerfile.cu13` 的 Megatron 补丁后的源码。

## 来源与补丁

| 组件               | 固定来源                                                                            | 增量                                                                          |
| ------------------ | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Megatron           | Core `f78809f6` 上移植 NVIDIA/Megatron-LM PR8497（`c42e8216`）                      | TP2/SP/ETP1、native shared、固定 token 补齐、offload 缓存清理及 EP 进程组恢复 |
| MoK                | `QiZhangNV/mixture-of-kittens@ee7bdaba515fd3d557105d71809ec74cd523d5a1`             | 640 维尾块支持                                                                |
| 调度器             | `cursor/mixture-of-kittens` PR10，commit `79b320ab5236dd0d031ea580063571271924bdae` | 优化重复扫描；保留 `schedule_entrypoint` 绑定名称                             |
| ThunderKittens     | `HazyResearch/ThunderKittens@1c3920d993404dd49a6d4c7267ea11d583bd5c68`              | `tcgen05.cuh` MN-major 修复                                                   |
| Transformer Engine | 基底 TE 2.18                                                                        | bounded Adam 修复，限制临时 tensor 句柄数量                                   |

补丁顺序及目标文件 SHA256 见 `manifest.json`。`docker/Dockerfile.cu13` 已应用 TE 的 bounded Adam 补丁，MoK 构建直接校验补丁后的 TE 源码。

`build.py` 使用 `pip wheel --no-deps --no-build-isolation` 编译并安装 MoK。CUDA driver stub 仅用于链接，不写入运行时 `LD_LIBRARY_PATH`。普通 `docker build` 不需要 GPU，只检查源码、安装路径及 SM103 cubin。

## 运行检查

构建后，在有 GPU 的容器中检查实际导入路径及扩展加载：

```bash
docker run --rm --gpus all --entrypoint python relax-mok:pr10-sm103 \
  /opt/relax/mok-build/verify.py --runtime
```

所有 Ray worker 和 driver 应使用同一镜像。将 `MEGATRON` 指向镜像内的适配源码，并移除旧实验源码覆盖路径，避免 `PYTHONPATH` 遮住已安装的 MoK wheel。

在 SM103 GPU 上验证训练时，检查 TP2/SP、EP8、ETP1 下的前向、反向和参数更新，并覆盖实际 token 数小于 `fixed_num_local_tokens` 的 microbatch。

## OpenR1MM SFT 示例

仓库仅新增一个薄示例，复用现有 32 卡 SFT 入口，默认 TP2/PP4/CP1/EP8/ETP1、GBS128、32K 长度上限，冻结视觉编码器和投影层：

```bash
export MODEL_DIR=/shared/models/Qwen3.8-Flash-Next
export DATA_DIR=/shared/data/openr1mm-sft.jsonl
export SAVE_DIR=/shared/checkpoints
export MEGATRON=/root/Megatron-LM
bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-openr1mm-32xgpu-blackwell.sh
```

`DATA_DIR` 直接接收已准备好的 `messages/images` 格式 SFT JSONL，图片路径须对所有 worker 可见。设置 `MOE_BACKEND=hybridep` 可切换为同配置参考后端；`DRY_RUN=1` 仅输出配置，不启动 Ray。

`MOK_BACKEND_CONFIG` 可传入 JSON 覆盖 MoK 配置。默认 `shared_expert_mode=native`，`fixed_num_local_tokens` 按 TP2/SP 和物理 padding 上限计算；改变长度时同步设置 `SEQ_LENGTH`、`MAX_TOKENS_PER_GPU`。`macrobatch_size` 可按负载另行指定，不固化实验专用值。

当前机器每节点只有一个 8 卡 NVLink 域，因此此示例使用节点内 EP8。跨节点训练使用 NCCL/EFA，但 MoK 专家组仍需显存互访，不能直接在这些机器上扩展至跨节点 EP16/EP32。
