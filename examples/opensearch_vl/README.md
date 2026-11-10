# OpenSearch-VL：SFT 与评测

本例使用完整的图文工具轨迹训练 Qwen3.5-4B / Qwen3.6-27B。数据准备不按长度过滤；训练在 32K token 上限下跳过超长样本，训练两轮，只在最后保存并导出模型，开启 ClearML。

- [数据准备](data-preparation.md)：原始七域数据、ZIP 图片读取、切分与真实 Processor 审计。
- [训练后模型评测](evaluation.md) / [English](evaluation.en.md)：SimpleVQA 英文 test 和 VDR testmini 的工具 Agent 评测。
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
