# CPT 持续预训练

Relax 在现有 SFT 管线中提供纯文本 CPT 模式，优先面向 Qwen3.5 Dense / MoE 的 Megatron 后端。使用现有模型配置与权重转换，不增加新的训练服务。

```bash
--loss-type sft \
--sft-training-mode cpt \
--sft-cpt-template qwen3_5 \
--prompt-data /path/to/train.jsonl \
--input-key text \
--use-dynamic-batch-size \
--max-tokens-per-gpu 8192 \
--sft-oversize-strategy truncate_right
```

`--sft-training-mode` 默认为 `sft`，已有 SFT / RL 脚本无需修改。CPT 的训练目标仍是 `causal_lm`；不要将 `--task-type` 或 `--loss-type` 改成 `cpt`。

Qwen3.5-35B-A3B 可直接使用交付脚本：

```bash
CPT_ROOT=/path/to/Relax-CPT \
HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B \
PROMPT_DATA=/path/to/cpt.jsonl \
bash scripts/training/cpt/run-qwen3.5-35B-A3B-cpt-8xgpu.sh
```

双机 16 卡使用同目录的 `run-qwen3.5-35B-A3B-cpt-16xgpu.sh`，并提供已有 Ray 集群的 `RAY_ADDRESS` 与 rank-0 GPU 节点的 `MASTER_ADDR`。交付镜像默认从 `/root/Megatron-LM` 加载 Megatron，并在启动前检查 expert grad norm 修复；不需要额外挂载运行时补丁。

## 数据和监督语义

一行是一篇文档，支持现有 JSONL / Parquet 读取方式。例如：

```json
{"text": "这里是一篇完整的领域文档。"}
```

也支持 ms-swift 的预训练数据格式，此时设置 `--input-key messages`：

```json
{"messages": [{"role": "assistant", "content": "这里是一篇完整的领域文档。"}]}
```

当未指定 `--input-key` 且行中没有默认 `input` 字段时，依次尝试 `text`、`messages`。显式指定的其他列名不会回退。每条 messages 必须只有一条纯文本 assistant 消息。

- 直接调用 tokenizer 编码，不应用 chat template，不添加 system/user/assistant 包装或思考前缀。
- `--sft-cpt-template raw`（默认）保留原文空白和思考内容，全部 token 参与监督。
- `--sft-cpt-template qwen3_5` 对齐 ms-swift Qwen3.5 的生成式预处理：去掉首尾空白、规范化 `<think>` 换行，并按 `all+ignore_empty_think` 屏蔽空思考前缀及紧随其后的空白。非空推理正文继续参与监督；`<thinking>` 是普通文本，不等同于 `<think>`。
- `raw` 保留 tokenizer 默认 special-token 行为，缺少 EOS 时补齐。`qwen3_5` 按 loss 权重分段、合并相邻同权重文本后分别编码，末尾补 EOS；已有 EOS 或 `<|endoftext|>` 时不重复添加。Qwen3.5 文本编码不添加 BOS。
- 数据层生成与 token 等长的 mask，Megatron 在每个样本内左移一次并屏蔽最后一个位置，避免预测下一篇文档的首 token。Qwen3.5 模板首位 mask 为 0，与 ms-swift 的 `labels[0] = -100` 对应。
- 训练、独立验证集、自动划分验证集均使用 CPT 编码；PPL 复用 SFT 的 token 加权评估。

纯文本 CPT 不接受图像、视频、音频、工具调用、多轮对话、分类标签、自定义数据集类或选择性 loss 设置。发现空文本、无 EOS 的 tokenizer、错误列名等会报错。`--sft-predict-interval` 属于对话生成评估，本模式暂不支持；使用 loss/PPL 评估。

## 与 ms-swift 基线的关系

参考相邻 ms-swift 仓库的 `PretrainArguments` 和生成模板：CPT 关闭 chat template。Qwen3.5 的参数初始化还会自动把 `all` 调整为 `all+ignore_empty_think`。选择 `qwen3_5` 模板可匹配此默认行为；Relax 训练运行时无需依赖或安装 ms-swift。

当前实现的边界：

| 能力 | Relax CPT |
| --- | --- |
| 全参训练、已有 Qwen3.5 模型适配 | 复用 SFT/Megatron |
| 流式读取、预取、确定性打乱 | 复用 SFT 数据集 |
| Packing、CP 等并行配置 | 复用 Megatron SFT 数据路径及其限制 |
| 超长文档 | 支持现有 `keep`、`skip`、`truncate_left`、`truncate_right` |
| ms-swift 的 `truncation_strategy=split` | 暂未实现；请提前离线切分长文档，每行一个片段 |
| ms-swift 的 `cached_dataset` 导出格式 | 不直接读取；使用 Relax 支持的 JSONL/Parquet |
| 多模态 CPT | 暂未实现 |

`capacity = max_tokens_per_gpu × context_parallel_size`。示例使用右侧截断；因此超长文档会丢弃尾部，尾部 EOS 也可能被截掉。需要保留全部语料时，应离线切分并留出 EOS 的位置。`keep` 可能超出动态批处理容量；`skip` 会改变实际消耗的行数，断点恢复时可能重读或跳过部分数据。稳定续训建议使用预先切分的数据，保持一行对应一个训练样本。

显式单消息 `loss_scale=1` 会按 ms-swift 规则覆盖空 think 屏蔽；其他非单位权重仍不支持。Qwen3.5 截断保留原生 image/video pad token，并重置截断后的首位标签；这只是编码边界兼容，不代表支持多模态 CPT。

Qwen3.5 模板只应显式用于相应模型。不要将 Gemma 的 thinking scaffold 或 SFT 的 assistant-only mask 直接套到 CPT 上。当前兼容实现也保留 ms-swift 的边界行为：`<think>` 之前的正文可能在规范化时被丢弃；连续空 think 块可能使后续答案也被屏蔽。这些行为由基线的首个结束标签切分、正则分段后二次匹配造成，应在数据准备时检查。
