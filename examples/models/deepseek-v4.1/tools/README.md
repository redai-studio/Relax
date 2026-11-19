# DeepSeek V4.1 权重准备

[English](./README.en.md)

[convert_to_bf16.py](./convert_to_bf16.py) 将官方 DeepSeek-V4.1-Flash 的混合 FP8/FP4 HF checkpoint 转换为 BF16 HF 基座。准备 Flash-1017 冻结视觉塔的 SFT 时使用 `--keep-vision` 保留图文结构。默认模式用于已有纯文本训练和 rollout。

## 来源和转换范围

此前使用的 `DeepSeek-V4.1-bf16` 来自官方 `deepseek-ai/DeepSeek-V4.1-Flash`，本地下载记录的 revision 为 `8b3cec70c307624fb098c3a04bdbfc118e7abda9`。

| 原始内容 | 默认纯文本模式的输出处理 |
| --- | --- |
| Dense FP8 + 32×32 E8M0 scale | 反量化为 BF16，删除对应普通 scale |
| Routed experts 的 packed FP4 + group-32 E8M0 scale | 反量化为 BF16，删除对应普通 scale |
| 已有 BF16、FP32 辅助权重 | 原样复制，保留 dtype |
| 冻结 Engram `embed.weight` / `embed.scale` | 保留 FP8/E8M0 原始字节；不转 BF16 |
| `mtp.*`、`vision.*`、`aligner.*`、`image_*`、`.ffn.gate.bias_vl` | 丢弃 |
| 配置 | 展开 `text_config`，移除量化配置、视觉配置及 `dspark_*`，设置 `num_nextn_predict_layers=0` |

## 准备保留视觉塔的 Flash-1017

使用完整的 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 快照。当前下载的 revision 为 `08c8657f02291e11b6f6b63e0e16014f8e748518`，与 `1bae0e1088993b9b82de08f59de5be5a34e42170` 的权重和配置相同，新增了 `chat_template.jinja`。在训练环境的 Relax 仓库根目录执行。`SOURCE_1017` 指向已下载的快照，`BASE_1017` 指向所有训练节点均可访问的新目录。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_1017=/path/to/DeepSeek-V4.1-Flash-1017
export BASE_1017=/path/to/DeepSeek-V4.1-Flash-1017-bf16
convert=(python examples/models/deepseek-v4.1/tools/convert_to_bf16.py
  --model-dir "$SOURCE_1017" --save-dir "$BASE_1017" --keep-vision)
"${convert[@]}" --check-only
"${convert[@]}"
```

图文模式逐字节保留原始 `vision.*`、`aligner.*`、`image_start`、`image_end`、`image_newline` 和每个主干层的 `ffn.gate.bias_vl`，同时复制 composite 配置、tokenizer、官方 prompt 编码器、图像预处理器和视觉参考实现。转换时移除 draft MTP/DSpark。`conversion.json` 记录模式、源分片大小与修改时间、可取得的 HF 下载 revision 和 etag，以及资产摘要。中断后用同样的参数加 `--resume` 恢复。

四层诊断模型使用单独的输出目录，在执行上述两条命令前向 `convert` 数组追加 `--decoder-layers 4`。它保留原生第 0 到 3 层、每层全部 384 个专家、整个视觉塔和第 1 层完整的 Engram 表与 scale，KV 和 index 源仍为第 2 层。候选过滤的源层在裁剪区间之外，因此关闭候选过滤。可对实际源 revision 执行 `--check-only` 估算空间。

## 准备默认纯文本基座

在与仓库匹配的 Relax CU13 环境中，从仓库根目录执行。转换器使用 PyTorch 和 Relax 日志模块，默认在 CPU 上独立运行。

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

## 将训练结果导出为 HF

在仓库根目录调用 CPU 导出器，`DCP_ITER` 选择要评测的训练迭代，`BASE_1017` 指向这次训练实际使用的只读 BF16 目录。训练在迭代目录记录 `v41-frozen-base.json`，恢复和导出时会核对资产摘要与分片名、大小和修改时间。迁移基座时需保留文件修改时间。请为完整 HF 模型准备足够的空间，并使用新的输出目录。

```bash
export DCP_ITER=/path/to/checkpoints/iter_0000100
export HF_EXPORT=/path/to/new-hf-export
convert=(python scripts/tools/convert_checkpoint_to_hf_cpu.py
  --input-dir "$DCP_ITER" --origin-hf-dir "$BASE_1017" --output-dir "$HF_EXPORT"
  --allow-base-fallback vision --allow-base-fallback engram)
"${convert[@]}" --dry-run
"${convert[@]}"
```

导出器从 DCP 全局分片重建语言权重，包括 routed/shared experts、mHC 与可训练的 Engram 投影。显式指定的两个 fallback 组仅允许从基座逐字节补回冻结视觉资产、图像路由 bias 和冻结 Engram 表。产物包含独立索引、处理器、tokenizer、原 composite 配置以及保持训练保存 dtype 的浮点权重。`conversion-result.json` 列出从基座补回的张量及 DCP 与基座之间的 dtype 差异。旧版 flat 纯文本模型只需 `--allow-base-fallback engram`。

导出精度以同一步训练 checkpoint 为参照。在 Megatron 和独立冷加载 HF 的 SGLang 上，用同一图文输入和相同的 assistant 监督位置比较平均负对数似然。若要衡量原始量化格式改用 BF16 对整模效果的影响，应在训练前分别评测原始 40 层 1017 和 40 层 BF16 基座。四层实验的测量结果见[八卡验证记录](../../../../docs/draft/deepseek-v41-1017-frozen-vision-thd-plan.md)。
