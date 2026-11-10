# OpenSearch-VL：SFT 与评测

本例使用完整的图文工具轨迹训练 Qwen3.5-4B / Qwen3.6-27B。数据准备不按长度过滤；训练在 32K token 上限下跳过超长样本，训练两轮，只在最后保存并导出模型，开启 ClearML。

- [数据准备](data-preparation.md)：原始七域数据、ZIP 图片读取、切分与真实 Processor 审计。
- [训练后模型评测](evaluation.md) / [English](evaluation.en.md)：SimpleVQA英文test和VDR testmini的无工具问答与MaaS判分；工具Agent为可选模式。
- [评测方案对比与工具准备](evaluation-setup.md)：其他GitHub项目的协议区别、工具依赖来源与准备方式。
- [Qwen3.5-4B 验证记录与阻塞项](results/qwen3.5-4b.md)：区分历史训练产物、上传验证版本和正式 benchmark。

## 训练入口

环境需已有匹配的 PyTorch、Megatron、SGLang、Ray 和 ClearML 配置；以下路径由运行环境提供，不将内网地址或密钥写进仓库：

```bash
export MODEL_DIR=/path/to/models
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3.5-4B"
export DATA_DIR=/path/to/data
export EXP_DIR=/path/to/experiments
export SAVE_DIR="${EXP_DIR}/opensearchvl/my-run/outputs"
export PROJECT_NAME=Relax/sft/opensearch-vl
export EXP_NAME=qwen3.5-4b-opensearch-vl-32k-8xgpu
export NUM_EPOCHS=2
export GLOBAL_BATCH_SIZE=32
export MAX_TOKENS_PER_GPU=32768
bash scripts/entrypoint/ray-job.sh examples/opensearch_vl/scripts/run-qwen3.5-4B-8xgpu.sh relax
```

在已配置的 8 GPU Ray 环境运行，提交默认阻塞等待。27B 改用 `run-qwen3.6-27B-8xgpu.sh` 和对应 HF 目录。TP/PP、整层重算及 optimizer CPU offload 由 `TP_SIZE / PP_SIZE / RECOMPUTE / OPTIMIZER_OFFLOAD_FRACTION` 调整，按实际显存和稳定吞吐选择，不能直接套用其他卡型的结果。

正式训练每 100 步做验证集 loss/PPL；这与 SimpleVQA/VDR 工具评测是不同环节。`BENCHMARK_ONLY=1` 是训练性能测试，默认只跑 8 步、不保存模型，也不是任务准确率评测。

## 最终产物

训练脚本最后保存 Megatron 权重 `${SAVE_DIR}/${EXP_NAME}/relax`，并导出 `${SAVE_DIR}/${EXP_NAME}/final-hf-<step>`；不保存 optimizer/RNG，也不在中途保存 checkpoint。

公共训练脚本自身不执行 QS 上传。QS 启动器会在后面串联最终目录改名、`model_tools` 上传、文件完整性/训练关联校验和服务验证；这些内部下载/上传启动配置留在 Git 外，由 `relax-ce` 的 `qs-opensearch-sft` skill 管理。成功后才清理 `${EXP_DIR}/opensearchvl/` 下本任务的临时目录，收据与评测日志保留在独立 results 目录。

评测必须核对实际模型版本：`kind=upload-validation, epochs=0` 只说明上传链路验证成功，不能作为 Relax SFT 模型。正式版本应记录 `kind=trained-final`、完整 branch/commit、训练 Job/Trial，以及最终训练轮数。

## SwanLab 指标同步

QS 启动命令安装 `swanlab==0.7.16`，将认证文件安全复制到容器的 `~/.swanlab/.netrc`（权限600），再设置 `ENABLE_SWANLAB=1`。认证文件不放入代码库。训练同时保留 ClearML，并通过 TensorBoard 标量同步到 SwanLab。

`SWANLAB_PROJ_NAME` 设置不含 `/` 的项目名；该固定SDK版本使用 `SWANLAB_MODE=cloud`。设置 `SWANLAB_LOG_DIR` 保存日志和看板链接，`SWANLAB_WORKSPACE` 选择目标workspace。运行器关闭自动环境、依赖、硬件及终端日志采集，只同步标量和明确设置的配置。Ray worker hook 在创建指标服务的 TensorBoard writer 时初始化 SwanLab，记录完整分支、commit、训练Job和Trial。其他worker不会主动创建实验。

训练后可直接用本地最终HF导出运行无工具评测，无需上传模型。模型来源需标为 `kind=trained-final, origin=local-final-export`，验证权重索引、分片和文件hash，记录实际训练轮数、分支、commit、Job/Trial；MaaS配置通过私有文件或环境输入。评测结果写入持久结果目录后，仅清理本任务的数据和缓存，保留未上传的最终HF权重。
