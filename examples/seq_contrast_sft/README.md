# Sequence contrastive SFT / 序列对比训练

This example follows `examples/seq_cls_sft`: select a task with
`--loss-type sft --task-type seq_contrast` and reuse the existing SFT producer,
TransferQueue, Actor, Megatron forward/backward, optimizer, scheduler and native
checkpoint path. It adds a pair dataset and InfoNCE loss, without a separate
trainer or Actor implementation.

本例沿用 `seq_cls_sft` 的任务扩展方式，复用上述框架组件，不另写训练循环。
Qwen2-VL 通过 `relax/models/qwen2_vl` 注册独立 Bridge：语言模型用 MCore，
视觉塔和多模态位置编码沿用 Qwen2-VL 自身实现，不以 Qwen2.5-VL 替代。

## Inputs / 输入

- `${MODEL_DIR}/Qwen2-VL-2B-Instruct-emb`: a prepared HF checkpoint containing
  `<emb>` in its tokenizer and a corresponding model embedding row. Preparation
  must initialize and save the row consistently; training never adds tokens or
  resizes the checkpoint. / 预先准备并保存 `<emb>` 及其 embedding 行，启动时只读检查。
- `${DATA_DIR}/nli/pairs.jsonl`: one complete query/positive pair per row. Use the
  native four-message form below, or `query`/`positive` dictionaries each with
  pretokenized `input_ids`. / 每行一个完整正例 pair，可使用下述四条消息，或两个预编码分支。

```json
{"messages":[{"role":"user","content":"A query"},{"role":"assistant","content":"<emb>"},{"role":"user","content":"Its positive"},{"role":"assistant","content":"<emb>"}]}
```

Raw conversations use the existing SFT processor pool. This NLI adapter is
text-only; it does not load raw images or hard negatives. Each branch must fit
`seq_length`, and overlength or malformed rows raise an error. Rows are never
silently filtered or truncated. The final batch repeats the current epoch's
head; native checkpoint resume reconstructs that padded-epoch cursor.

原始消息复用 SFT processor pool；当前 NLI adapter 仅支持文本，不读取图片或 hard negative。
超长或无效分支直接报错，尾批复用本 epoch 开头，恢复时按补齐后的 epoch 游标继续。

## Launch / 启动

On an already provisioned, exclusively allocated Ray cluster, provide
`MODEL_DIR`, `DATA_DIR`, a **persistent** `SAVE_DIR` and the Ray Jobs HTTP
`RAY_ADDRESS`, then run from the repository root:

在已分配给本任务的 Ray 集群上设置上述目录和 Jobs HTTP 地址，然后从仓库根目录运行：

```bash
bash scripts/entrypoint/ray-job.sh \
  examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-nli-pt-8xgpu.sh
```

The script requests eight GPUs, TP=PP=CP=1 and a 512-**pair** global batch. The
default 1,078 updates correspond to two padded epochs only for 275,601 rows.
For other inputs, explicitly override `--num-rollout`; the custom dataset uses
this update count rather than `--num-epoch`.

The example selects Bridge's native **Kaiming-uniform LoRA A initialization**
(`RELAX_LORA_A_INIT_METHOD=kaiming`); LoRA B retains Bridge's zero initialization.
The initializer is included in the Ray runtime environment. Set
`RELAX_LORA_A_INIT_METHOD=xavier` explicitly to use Bridge's Xavier option, with a
new `EXP_NAME`. Other Relax recipes keep their existing initializer defaults.

本例默认使用 Bridge 原生 **Kaiming-uniform 初始化 LoRA A**，B 保持零初始化，
配置会传入 Ray 训练进程；如需 Xavier，可设置 `RELAX_LORA_A_INIT_METHOD=xavier`，
并更换 `EXP_NAME`。其他 Relax recipe 的默认初始化不变。

The native checkpoint path is `${SAVE_DIR}/${EXP_NAME}` for both save and load,
with an interval of 200 updates and retention of two checkpoints. Reuse the same
path to resume the same experiment; use a new `EXP_NAME` for a different run.
The default experiment is `qwen2-vl-2b-nli-bridge-lora-kaiming-dp8`. Start this
recipe in a new directory: historical precision-profile checkpoints used a
different norm layout and optimizer/RNG settings and are not supported resume
inputs for this recipe.
ClearML project defaults to `Relax/sft/seq-contrast`; each launch has a new task
name and local log, while the resume path stays fixed. Submission blocks unless
`RAY_NO_WAIT` is explicitly nonempty.

默认申请八卡，global batch 的单位是 **pair**，保存完整原生 checkpoint，每 200 步保存，
保留两份。相同实验沿用稳定的 load/save 路径；新实验改 `EXP_NAME`。
默认 1,078 步只对应 275,601 行数据的两个补齐 epoch，其他规模需显式改步数。
ClearML 每次启动新任务，训练恢复由持久化 checkpoint 决定。

## Training contract and validation limits / 训练契约与验证边界

- Pair rows stay atomic through DP sampling, then expand to adjacent
  `[query, positive]` sequences. Pool the hidden state preceding the first
  `<emb>`, using a frozen vocabulary head/embedding and vision tower.
- One optimizer update contains the complete contrastive negative set in a
  **single packed microbatch**. Exceeding `max_tokens_per_gpu` fails rather than
  splitting InfoNCE into independent objectives. The configured token budget is
  a packing limit, **not a proven memory fit** on a particular GPU.
- The loss is one-way mean cosine InfoNCE at temperature 0.05. DP gathers remote
  detached features, retains local gradients, and preserves the native averaged
  `1/DP` parameter-gradient scale through the shared Megatron loss wrapper.
- Initial support is synchronous, DP-only, THD training. Model parallelism,
  dynamic CP, independent microbatch accumulation, periodic SFT eval/predict,
  CPU prepack, and portable HF export are rejected. Native checkpoints remain
  the supported resume artifacts.
- The example uses Bridge's existing **CanonicalLoRA** with seven independent
  Q/K/V, output, gate/up and down projection adapters, with the standard Bridge
  forward and Transformer Engine layer specification. HF norm weights map to
  the corresponding fused TE projection norms.
- Data ordering uses the existing `NLIPairDataset`/`IndexManager` path. The script
  retains `--data-parallel-random-init` for per-rank RNG state. There are no
  reference-order replay, padded/eager projection overrides, HF norm/attention
  replacements, or custom optimizer/gradient-communication hooks.

Pair 分卡后才展开；完整负例集合不拆成独立 microbatch。token 预算不能证明实际显存足够。
当前只支持同步 DP、THD，明确拒绝尚未接入的并行模式和评测/导出组合。
复用 Bridge CanonicalLoRA 保留七种独立投影，实际参数及保存恢复仍须验证；不能仅据此宣称 PT 已对齐。

CPU tests exercise the production iterator/loss dispatch, pair/resume contracts,
real two-process Gloo loss/gradients, HF vision embeddings/positions and mapping
key coverage. A submission test checks that the example passes the initializer
through Ray's runtime environment while preserving the existing runtime options.
The language boundary in the CPU vision tests is a recording module, not MCore
attention.

This cleaned recipe has not been rerun through full GPU training/evaluation.
Previous alignment scores used different data-order/RNG settings and must not
be attributed to this configuration. Multi-node training and exact GPU
checkpoint-resume behavior remain unverified for this revision.

CPU 测试覆盖配对、共享训练/损失分派、Gloo 梯度、HF 视觉/位置编码与权重映射，
以及 example 的初始化配置是否正确进入 Ray runtime env。语言模型的 CPU 测试使用
记录边界，不能替代 MCore GPU 验证。本次清理版未重新跑完整 GPU 训练/评测；
历史对齐分数使用不同的数据顺序/RNG 设置，不代表本配置的成绩。多机训练与 GPU
精确断点续训仍需验证。历史精度实验的 checkpoint 请保留在原目录，不用于本例续训。
