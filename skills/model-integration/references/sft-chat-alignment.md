# 对话 SFT 对齐验收

用目标 checkpoint 的原生 tokenizer、模型实际使用的 processor 及其真实 chat template 或 Python 编码器。读取模型配置和 `generation_config.json`，记录 checkpoint 或 revision、模板哈希、训练与服务的模板覆盖、loss 选项和评测实际发送的参数。若使用模板覆盖，保留原生渲染作对照，并验证训练和服务实际生效的版本。简化模板和 fake tokenizer 可用于轻量回归，真实模板验收单独执行和记录。

## 样例覆盖

先从实际训练集中按下列场景分层抽样，再用最小的结构化消息补齐低频边界。两类样本都走真实模板和训练数据路径。工具调用通过 `tool_calls` 与 `tool_call_id` 表达，保留真实回包结构，避免用拼好的标签文本代替结构化输入。

- **文本与多轮**：直接回答、含历史 assistant 回复的多轮对话、最终文本回复。先按实际 recipe 验证，再覆盖本次改动影响的全轮监督、仅最后一轮监督和显式 `learn`。支持 thinking 的模型覆盖有内容和空 thinking，按对应配置校验开闭边界和 mask。
- **工具调用**：单次调用和同一 assistant 消息里的多次调用。覆盖有回包且继续到最终答复的轨迹、样本直接结束在工具组的轨迹，以及模板已经生成组结束符的情况。支持乱序回包时，用不同的 `tool_call_id` 和回包正文重放按序与乱序输入。
- **多模态**：用实际 processor 加载真图。在 recipe 支持的用户消息和工具回包位置验证媒体顺序，尤其用两张不同图片检查回包重排后是否仍与调用对应。
- **长度和前缀边界**：改动影响序列长度时，覆盖停止符正好装入样本和追加后超限两种情况，核对实际截断或跳过行为。recipe 使用 `add_generation_prompt` 或 `continue_final_message` 时，检查开放的生成前缀保持开放。

按场景记录样本来源或可复现的样本 ID、样本数、有效监督边界数、失败数和缺样项。整个批次通过时仍需核对各类边界是否实际出现。

## 比较三条路径

1. 用完整结构化消息走 checkpoint 的原生 `apply_chat_template`，保留工具 ID、名称、回包、thinking 和图片占位。记录本模型用于结束正文和交给工具的 token。EOS 可在模型或 generation config 里，多值配置中需要识别每个停止符的用途。
2. 走 Relax 的真实训练路径，从 `SFTStreamingDataset.get_batch_by_indices` 拿 `tokens` 和 `loss_mask`。按训练规则检查移位后的 predictor、label 和 mask。多模态样本需走实际 HF processor，核对展开后的图像位置和正文、停止符监督。
3. 用服务的实际模板参数生成每次模型调用的 prefill，包括第一次发工具和工具回包后。拿服务前缀与完整训练序列逐 token 对比，记录第一个分歧。用实际请求和原始生成记录验证 parser、有效停止符和 `ignore_eos`。如果原始流已生成停止符但解码继续，检查服务的停止配置。

## 在测试里断言

- 从真实模板、模型停止配置和消息结构独立确定停止符的 ID、语义和位置。正文 EOS 和工具交棒都要做 next-token 断言。若应监督的停止符在 `tokens[k]`，则 `k > 0`，并验证 predictor 位置 `k - 1` 的真实 label 等于该停止 token、移位 mask 等于 1。
- 验证 system、user 和工具的上下文正文及图片 token 的默认 mask 为 0。若角色标记同时承担前一条 assistant 消息的停止符，按该 assistant 的 `learn` 和轮次选择校验标记上的 loss。覆盖模板已渲染该标记和样本直接结束导致模板省略标记两种情况，验证应监督的停止符保留或补齐一次。
- 并行调用先按关系校验调用与回包，整组完成处才出现模型要求的交棒监督。回包重排时，回包正文、learn 标记和媒体引用须一起对应到原调用。
- 比较 tokenizer 的原生渲染与 Relax 训练 token。核对有意追加的 EOS，并报出意外的差分。前缀一致时，按 recipe 的 thinking 策略核对第一个生成位置及其 loss mask。
- 把稳定的断言放进 CPU 行为回归。修 bug 时，条件允许则用旧实现或针对性故障注入确认用例会因目标问题失败。分别报告真实模板回放、轻量回归和线上请求的结果。

## 归因

| 首次出现差异的路径 | 检查位置 |
| --- | --- |
| 原生 token 序列缺少结束符 | 先确认原生模板的约定。模板按约定省略停止符时，检查 SFT 是否按 checkpoint 的停止配置补齐 |
| 原生渲染未体现请求参数 | 检查 Jinja 或 Python 编码器对该参数的支持及条件分支 |
| 原生有正确 token，训练 token 缺失、错序或 loss 为 0 | canonical 消息适配、loss span 或 assistant mask 投影 |
| 文本短序列对齐，processor 展开后出现偏差 | 媒体顺序、占位扩展、长序列 mask 投影 |
| 训练和离线 prefill 相符，服务输出不同 | 网关实参、服务模板覆盖、reasoning 和 tool parser、解码停止配置 |

## 仓库样例

[真实模板的 SFT 数据集回归](../../../tests/cpu/engine/sft/dataset/test_qwen_chat_template_patch.py#L829) 从本地 checkpoint 加载 tokenizer，用一条带空 thinking 的结构化 JSONL 样本走 `SFTStreamingDataset`。它检查完整序列保留 thinking 边界，并用固定期望断言 loss 选中的文本包含回复及其结束符。新增用例可沿用这条路径，将输入扩展到改动涉及的场景，并补上对应的 next-token 断言。

从仓库根目录运行，`TOKENIZER_DIR` 指向该测试所需的本地 tokenizer 目录。

```bash
QWEN38_TEST_TOKENIZER_DIR="${TOKENIZER_DIR:?set to the local tokenizer directory}" \
    pytest -q tests/cpu/engine/sft/dataset/test_qwen_chat_template_patch.py::test_qwen38_real_tokenizer_dataset_text_item_loss
```
