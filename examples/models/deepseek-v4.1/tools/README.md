# DeepSeek V4.1 权重准备

[English](./README.en.md)

[convert_to_bf16.py](./convert_to_bf16.py) 将官方 DeepSeek-V4.1-Flash 的混合 FP8/FP4 HF checkpoint 转换为 Relax 纯文本训练和 rollout 使用的 BF16 HF checkpoint。SFT、RL、BSHD、THD 共用这一格式，无需为 THD 再转换一份。

## 来源和转换范围

此前使用的 `DeepSeek-V4.1-bf16` 来自官方 `deepseek-ai/DeepSeek-V4.1-Flash`，本地下载记录的 revision 为 `8b3cec70c307624fb098c3a04bdbfc118e7abda9`。旧实验归档中的 `convert_deepseek_v4_1_to_bf16-6f52e0b.py` 与 [Miles 提交 `6f52e0b`](https://github.com/radixark/miles/blob/6f52e0b3021484e8d6ad0d43c1074100adc8e4b9/tools/convert_deepseek_v4_1_to_bf16.py) 中的脚本相同；同目录 `convert-v41-bf16.log` 记录了成功转换。

[Miles 文档](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/docs/models/deepseek/deepseek-v4-1-flash.md#L44-L53) 提到的 `dsv41_cut_cast.py` 不是当前 main 中可直接运行的文件。上面的历史提交补齐了专用转换器。本工具基于该实现，保留转换语义，移除 Miles 包依赖，改为逐张量转换、流式复制原始数据，并增加预检查和显式恢复。

| 原始内容 | 输出处理 |
| --- | --- |
| Dense FP8 + 32×32 E8M0 scale | 反量化为 BF16，删除对应普通 scale |
| Routed experts 的 packed FP4 + group-32 E8M0 scale | 反量化为 BF16，删除对应普通 scale |
| 已有 BF16、FP32 辅助权重 | 原样复制，保留 dtype |
| 冻结 Engram `embed.weight` / `embed.scale` | 保留 FP8/E8M0 原始字节；不转 BF16 |
| `mtp.*`、`vision.*`、`aligner.*`、`image_*`、`.ffn.gate.bias_vl` | 丢弃 |
| 配置 | 展开 `text_config`，移除量化配置、视觉配置及 `dspark_*`，设置 `num_nextn_predict_layers=0` |

工具保留全部文本层及原层号，不提供减层重映射。它也不保留视觉头，不能用于准备 frozen-vision 多模态训练模型。已是 BF16 的四层模型无需运行此转换。

## 操作流程

在与仓库匹配的 Relax CU13 环境、仓库根目录执行。转换只使用 PyTorch 和已有 Relax 日志模块，不需要安装 Miles；默认在 CPU 上运行，也不需要启动 Ray。

先准备一个固定 revision 的完整原始模型。下面的 revision 对应此前使用的基础版，不是 `Flash-1017`：

```bash
export MODEL_DIR=/path/to/models
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision 8b3cec70c307624fb098c3a04bdbfc118e7abda9 \
  --local-dir "$MODEL_DIR/DeepSeek-V4.1-Flash"
```

已有完整下载目录可以跳过下载。源目录与输出目录必须独立；新转换建议使用一个新的输出名称，避免碰到正在训练使用的 BF16 目录。

先读取配置、索引和所有分片头部，检查量化格式、scale 配对、文本权重及 Engram 是否齐全，并估算输出大小：

```bash
python examples/models/deepseek-v4.1/tools/convert_to_bf16.py \
  --model-dir "$MODEL_DIR/DeepSeek-V4.1-Flash" \
  --save-dir "$MODEL_DIR/DeepSeek-V4.1-bf16-new" \
  --check-only
```

`--check-only` 不读取权重 payload，也不创建输出目录。上述原始版本预计输出 46,968 个张量、44 个分片，权重数据为 1,306,598,355,408 字节（约 1.31 TB / 1.19 TiB）；需要在保留原始模型之外安排这部分配额及少量元数据余量。

确认后执行转换；日志放在输出目录之外：

```bash
set -o pipefail
python examples/models/deepseek-v4.1/tools/convert_to_bf16.py \
  --model-dir "$MODEL_DIR/DeepSeek-V4.1-Flash" \
  --save-dir "$MODEL_DIR/DeepSeek-V4.1-bf16-new" \
  --device cpu \
  2>&1 | tee convert-deepseek-v4.1-bf16.log
```

每次只转换一个张量；Engram 等不需要反量化的数据按 64 MiB 块复制。CPU 内存仍需容纳当前反量化张量及其临时展开数据。需要 GPU 转换时可改为 `--device cuda`，通过 `CUDA_VISIBLE_DEVICES` 选择空闲卡；转换并不需要多卡训练环境。

中断后，保持源快照、脚本版本和输出目录不变，在相同命令后加 `--resume`。它检查 `conversion.json` 中的源路径、配置/索引哈希和脚本哈希，以及已完成分片的头部和大小，跳过已完成分片，重做未完成的 `.tmp` 分片。不要并发写入同一个输出目录。该检查不是对全部权重 payload 的校验和验证，源快照应保持只读。

所有分片完成后才发布最终 index，并把 `conversion.json` 的 `complete` 标为 `true`。输出同时包含展开后的 `config.json`、tokenizer、chat template、generation config 和原模型许可证（源目录有这些资产时复制）。转换完成后，训练脚本设置：

```bash
export HF_CKPT="$MODEL_DIR/DeepSeek-V4.1-bf16-new"
```

训练仍直接从 HF 权重加载，不需要预先生成 Megatron `torch_dist` checkpoint。保留这份 BF16 base；续训和未来独立部署导出都需要其中的冻结 Engram 数据。

## 训练结果导出为 HF 的现状

这里的工具只负责**原始量化 HF → BF16 HF**。训练保存的 Megatron `torch_dist` checkpoint 可用于续训；这与导出可独立加载的 HF 推理模型是两个步骤。

截至 Miles 提交 `fa13e2e`，官方有通用 [DCP → HF exporter](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/tools/convert_torch_dist_to_hf.py) 及 [V4.1 参数映射](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/miles/backends/megatron_utils/megatron_to_hf/deepseekv4_1.py)，但尚未给出完整 V4.1 部署导出 recipe。该映射跳过冻结 Engram table/scale，而 exporter 的资产复制也跳过原始 safetensors；仅运行通用 exporter 不会自动得到包含冻结表的完整独立模型。

Relax 当前也尚未接通这条完整导出路径：[V4.1 Bridge](../../../../relax/models/deepseekv41/bridge.py) 明确设置 `SUPPORTS_HF_PRETRAINED_EXPORT=False`，现有 CPU exporter 的模型映射不包含 V4.1。不能直接套用其他模型的通用导出命令。

完整导出需要将训练 checkpoint 的参数映射回 V4.1 HF 名称，从**同一次训练的 BF16 base** 原样补齐冻结 Engram 四个张量，保留必要的 FP32 参数，重建 index/config/tokenizer，并验证重新加载后的数值。在线同步到 SGLang 成功不等于这些离线导出步骤已经实现。

导出的 BF16 HF 与官方原始 FP8/FP4 部署包也不同；若希望恢复低精度部署权重，还需要单独的重新量化及推理精度验证，不能只把原始 `quantization_config` 加回去。
