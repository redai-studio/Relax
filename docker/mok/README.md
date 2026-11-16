# Megatron + MoK 增量镜像

将 Megatron 适配、MoK 640 维尾块支持和调度器优化封装到现有 CUDA 13 训练镜像。MoK wheel 在新镜像内编译，运行时无需挂载实验目录或复用旧 `.so`。

## 构建

从 Relax 仓库根目录执行，`BASE_IMAGE` 填写当前完整训练镜像的引用，建议使用不可变 digest：

```bash
export BASE_IMAGE='填写完整 CUDA 13 训练镜像的引用'
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t relax-mok:pr10-sm103 docker/mok
```

构建上下文仅为 `docker/mok`。基底需提供 Python 3.12、Torch `2.13.0+cu130`、CUDA 13.0 NVCC、TE 2.18、Bridge、HybridEP、Git、Make 和 patch；构建不升级这些依赖。默认解释器为 `/opt/sglang/bin/python`、Megatron 路径为 `/root/Megatron-LM`，可分别通过 `PYTHON_BIN` 和 `MEGATRON_ROOT` 构建参数调整。目标架构为 Blackwell `SM103`。

构建仅检查 Megatron 补丁涉及的 8 个文件的前后 SHA256，再以 `--fuzz=0` 应用补丁；不要求其他文件与实验镜像完全一致。补丁目标不匹配时需要先适配基底源码，不能假定当前 `Dockerfile.cu13` 产物与原始训练镜像一致。

## 来源与补丁

| 组件               | 固定来源                                                                                            | 增量                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Megatron           | 镜像源码快照；移植基于 NVIDIA/Megatron-LM PR8497，commit `c42e8216d2031ab3ca191e56c9d370b347c2cf02` | TP2/SP/ETP1、固定输入补齐与容量检查、native shared、参数属性及进程组缓存管理 |
| MoK                | `QiZhangNV/mixture-of-kittens@ee7bdaba515fd3d557105d71809ec74cd523d5a1`                             | 640 维尾块支持                                                               |
| 调度器             | `cursor/mixture-of-kittens` PR10，commit `79b320ab5236dd0d031ea580063571271924bdae`                 | 优化重复扫描；保留 `schedule_entrypoint` 绑定名称                            |
| ThunderKittens     | `HazyResearch/ThunderKittens@1c3920d993404dd49a6d4c7267ea11d583bd5c68`                              | `tcgen05.cuh` MN-major 修复                                                  |
| Transformer Engine | 基底 TE 2.18                                                                                        | bounded Adam 修复，限制临时 tensor 句柄数量                                  |

补丁顺序及目标文件 SHA256 见 `manifest.json`。TE 补丁统一保存在 `patches/transformer-engine-fused-adam-bounded.patch`，主 `docker/Dockerfile.cu13` 也引用此文件；已经修复的 TE 源码会跳过应用，未知版本会拒绝。

`build.py` 使用 `pip wheel --no-deps --no-build-isolation` 编译并安装 MoK。CUDA driver stub 仅用于链接，不写入运行时 `LD_LIBRARY_PATH`。普通 `docker build` 不需要 GPU，只检查源码、安装路径及 SM103 cubin。

## 运行检查

构建后，在有 GPU 的容器中检查实际导入路径及扩展加载：

```bash
docker run --rm --gpus all --entrypoint python relax-mok:pr10-sm103 \
  /opt/relax/mok-build/verify.py --runtime
```

所有 Ray worker 和 driver 应使用同一镜像。将 `MEGATRON` 指向镜像内的适配源码，并移除旧实验源码覆盖路径，避免 `PYTHONPATH` 遮住已安装的 MoK wheel。

此封装已验证补丁重放；当前环境无 Docker CLI，尚未实际构建新镜像。导入检查不替代新镜像的训练及数值验证。

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
