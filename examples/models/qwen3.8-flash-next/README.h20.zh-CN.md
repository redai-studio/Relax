# Qwen3.8 Flash Next：40 卡 H20 SFT

[English](./README.h20.md)

本 recipe 用于 5 节点、每节点 8 张 H20 的多模 SFT：完整 48 层 + MTP1，
TP4/PP5/DP2/EP4，布局 `Et*8|t*11|t*11|t*12|t*6mL`，64K packing、GBS1024。
vision/projection/PLE/indexer 冻结，冻结的视觉组件仍参与前向计算。
PLE 表放在 CPU；训练参数保留在 GPU，使用完整 uniform 激活重计算。

## 配置与启动

所有节点使用兼容的 Relax 镜像，具备 Qwen3.8 模型/processor 支持和 ClearML 凭据。
脚本通过 Python 运行时定位 Megatron-LM，也可用 `MEGATRON` 指定其他 checkout。
模型、数据、输出和源码必须在所有节点以相同路径可访问。

```bash
export MODEL_DIR="/path/to/hf-model"
export DATA_DIR="/path/to/train.jsonl"
export SAVE_DIR="/path/to/run-output"
```

`DATA_DIR` 接受 JSONL 文件或包含 `train.jsonl` 的目录，数据使用 `messages`、
可选 `tools` 和受支持的多模消息格式。所有 worker 必须能访问媒体路径。
recipe 不安装依赖，也不修改镜像。

专用多节点 TorchJob 在**每个节点**从选定 checkout 执行：

```bash
bash scripts/entrypoint/spmd-multinode.sh examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

任务环境须提供 `MASTER_ADDR`（与 head 的 `POD_NAME` 相同的主机名）、`POD_NAME`、
`HOST_IP` 和 `WORLD_SIZE=5`，每节点默认 8 张卡。标准入口会清理残留进程并启动 Ray。
已有 Ray 集群则只在 head 执行：

```bash
bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

脚本通过 source 完成一次标准 Ray Jobs 初始化，并在初始化前拒绝重复的活动 Relax
训练任务。默认阻塞提交；异步提交设置 `RAY_NO_WAIT=1`。日志保存在 `${SAVE_DIR}/logs/`。

## 参数与预览

配置集中在单个训练脚本。可通过环境变量覆盖：`NUM_STEPS`（10）、`GBS`（1024）、
`SEQ_LENGTH` / `MAX_TOKENS_PER_GPU`（65536）、`LR` / `MIN_LR`（2e-5 / 2e-6）、
`LR_DECAY_ITERS`（2023）、`LR_WARMUP_FRACTION`（0.05）、`SEED`（1234）、
`EXP_NAME` 和 `PROJECT_NAME`。GBS 应能被 dense DP2 整除，额外 CLI 参数会透传。
每次启动从 HF 初始化，当前 recipe **不保存或恢复训练 checkpoint**。

`RELAX_DIR` 可选择包含本 recipe 的其他完整 checkout。TorchJob 应从该 checkout
同时调用 SPMD 入口与 recipe。`RAY_DASHBOARD_ADDRESS` 指定 Ray Jobs HTTP 地址，
默认为 head 本机 `http://127.0.0.1:8265`；旧 `RAY_JOBS_ADDRESS` 名称仍作为后备。

设置三个必需路径后，可以预览配置，不执行 Ray/GPU 初始化或创建输出文件：

```bash
DRY_RUN=1 bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

预览默认使用 `/root/Megatron-LM`，不会导入该模块；其他镜像布局通过 `MEGATRON`
覆盖。标准入口的编译缓存设置仍然有效。
