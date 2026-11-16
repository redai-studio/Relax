# Sequence contrastive SFT / 序列对比训练

This example follows `examples/seq_cls_sft`: select a task with
`--loss-type sft --task-type seq_contrast` and reuse the existing SFT producer,
TransferQueue, Actor, Megatron forward/backward, optimizer, scheduler and native
checkpoint path. It adds a pair dataset and InfoNCE loss, without a separate
trainer or Actor implementation.

本例沿用 `seq_cls_sft` 的任务扩展方式，复用上述框架组件，不另写训练循环。
Qwen2-VL 通过 `relax/models/qwen2_vl` 注册独立 Bridge：语言模型用 MCore，
视觉塔和多模态位置编码沿用 Qwen2-VL 自身实现，不以 Qwen2.5-VL 替代。

The workflow supports Qwen2-VL-2B-Instruct and dense Qwen3.5-0.8B, 2B, 4B, and 9B.
The launchers keep their original `run-qwen2-vl-*` filenames and select the
architecture using the `config.json` in the `MODEL_DIR` you pass.

Qwen2-VL-2B-Instruct 和 Qwen3.5 稠密版 0.8B、2B、4B、9B 共用下文的命令。
启动脚本沿用原文件名，通过 `MODEL_DIR` 下的 `config.json` 选择模型架构。

## Inputs / 输入

- `MODEL_DIR`: the HF checkpoint directory for the current stage. Ret PT uses
  the prepared checkpoint containing `<emb>` in the tokenizer and its corresponding
  embedding row. Ret SFT uses the merged Ret PT export. / `MODEL_DIR` 指向当前阶段的
  HF checkpoint 目录。Ret PT 使用已准备好 `<emb>` 的底座，Ret SFT 使用合并后的 Ret PT 权重。
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
`RAY_ADDRESS`, then run from the repository root. The SFT and Rank stages also
need their prepared data and image paths accessible from the workers.

在已分配给本任务的 Ray 集群上设置上述目录和 Jobs HTTP 地址，然后从仓库根目录运行：

```bash
bash examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-nli-pt-8xgpu.sh
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
The default Ret PT experiment name is `<model>-nli-bridge-lora-kaiming-dp8`. Start this
recipe in a new directory: historical precision-profile checkpoints used a
different norm layout and optimizer/RNG settings and are not supported resume
inputs for this recipe.
Each launch writes a local log while the resume path stays fixed. Submission blocks unless
`RAY_NO_WAIT` is explicitly nonempty.

默认申请八卡，global batch 的单位是 **pair**，保存完整原生 checkpoint，每 200 步保存，
保留两份。相同实验沿用稳定的 load/save 路径；新实验改 `EXP_NAME`。
默认 1,078 步只对应 275,601 行数据的两个补齐 epoch，其他规模需显式改步数。
每次启动单独记录本地日志，训练恢复由持久化 checkpoint 决定。

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
  CPU prepack, and in-training `--save-hf` are rejected. Resume from native
  checkpoints and use the offline exporter below to hand off weights.
- The Ret stages use Bridge's **CanonicalLoRA**. Qwen2-VL targets independent
  Q/K/V, output, gate/up and down projections. Qwen3.5 targets language attention
  output and MLP down projections. Its model dimensions, mRoPE and embedding
  tying come from the HF checkpoint.
- Data ordering uses the existing `NLIPairDataset`/`IndexManager` path. The script
  retains `--data-parallel-random-init` for per-rank RNG state. There are no
  reference-order replay, padded/eager projection overrides, HF norm/attention
  replacements, or custom optimizer/gradient-communication hooks.

Pair 分卡后才展开；完整负例集合不拆成独立 microbatch。token 预算不能证明实际显存足够。
当前只支持同步 DP、THD，明确拒绝尚未接入的并行模式和评测/导出组合。
Ret 使用 Bridge CanonicalLoRA。Qwen2-VL 使用七种独立投影，Qwen3.5 使用语言侧 attention 输出和 MLP down 投影。
Qwen3.5 的模型尺寸、mRoPE 和输入输出 embedding 共享设置来自 HF checkpoint。

CPU tests exercise the production iterator/loss dispatch, pair/resume contracts,
real two-process Gloo loss/gradients, HF vision embeddings/positions and mapping
key coverage. A submission test checks that the example passes the initializer
through Ray's runtime environment while preserving the existing runtime options.
The language boundary in the CPU vision tests is a recording module, not MCore
attention.

The three-stage workflow has small-data single-node coverage. Full benchmark
scores, multi-node training and exact GPU checkpoint-resume behavior require
separate validation for the selected configuration.

CPU 测试覆盖配对、共享训练/损失分派、Gloo 梯度、HF 视觉/位置编码与权重映射，
以及 example 的初始化配置是否正确进入 Ray runtime env。语言模型的 CPU 测试使用
记录边界，不能替代 MCore GPU 验证。三段流程已覆盖单机小数据，完整 benchmark、多机训练与 GPU
精确断点续训需按实际配置验证。

## Three-stage workflow / 三段交付流程

### 1. Prepare Ret PT inputs / 准备 Ret PT 输入

Set `BASE` to the original HF checkpoint for either supported model family and
`NLI_CSV` to a CSV with `sent0` and `sent1` columns. Use an `EXP_PREFIX` unique
to the model and run, plus a separate `DATA_DIR` for each run's prepared files.
Keep the original data and images at their source paths. Keep the derived
directory variables in the same shell for the later stages.

`BASE` 指向所选模型的原始 HF checkpoint，`NLI_CSV` 需含 `sent0` 和 `sent1` 两列。
各实验用独立的 `EXP_PREFIX` 和 `DATA_DIR`，原始数据及图片可复用。后续步骤沿用同一 shell 中的目录变量。

```bash
export DATA_DIR SAVE_DIR RAY_ADDRESS

EMB_MODEL="${SAVE_DIR}/${EXP_PREFIX}-emb"
PT_MODEL="${SAVE_DIR}/${EXP_PREFIX}-ret-pt/merged"
RET_MODEL="${SAVE_DIR}/${EXP_PREFIX}-ret-sft/merged"
TRAIN_RETRIEVAL="${SAVE_DIR}/${EXP_PREFIX}-ret-sft/retrieval/train"
TEST_RETRIEVAL="${SAVE_DIR}/${EXP_PREFIX}-ret-sft/retrieval/test"

python examples/seq_contrast_sft/tools/prepare_nli.py \
  --base "${BASE}" --input "${NLI_CSV}" \
  --output-model "${EMB_MODEL}" \
  --pairs "${DATA_DIR}/nli/pairs.jsonl"
MODEL_DIR="${EMB_MODEL}" EXP_NAME="${EXP_PREFIX}-ret-pt" \
  bash examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-nli-pt-8xgpu.sh
```

### 2. Train Ret SFT and mine candidates / 训练 Ret SFT 并挖候选

Use the [offline exporter](#offline-export) to merge Ret PT against `EMB_MODEL`
and write `PT_MODEL`. Set `TRAIN_QUERIES` and `TRAIN_CANDIDATES` to the M-BEIR
training query and candidate JSONLs, `INSTRUCTIONS` to the query instruction TSV,
and `IMAGE_ROOT` to the directory from which M-BEIR image paths resolve. The
preparer writes both `pairs.jsonl` and `pairs.meta.json` for the launcher.

先把 Ret PT 合并到 `EMB_MODEL`，导出到 `PT_MODEL`。准备 M-BEIR 的训练 query、候选池、
指令 TSV 和图片根目录。数据准备会同时生成启动器使用的 `pairs.jsonl` 和 `pairs.meta.json`。

```bash
python examples/seq_contrast_sft/tools/prepare_mbeir.py \
  --queries "${TRAIN_QUERIES}" --candidates "${TRAIN_CANDIDATES}" \
  --instructions "${INSTRUCTIONS}" --image-root "${IMAGE_ROOT}" \
  --tokenizer "${PT_MODEL}" --output "${DATA_DIR}/mbeir/pairs.jsonl"
MODEL_DIR="${PT_MODEL}" EXP_NAME="${EXP_PREFIX}-ret-sft" \
  bash examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-mbeir-sft-8xgpu.sh
```

The data loader shuffles queries and samples a positive and instruction at each
visit. The launcher computes updates from the prepared query count and batch
size. The default batch size is 512; 1,332,790 queries take 2,604 updates.
Images are checked when read by the processor.

Export Ret SFT against `PT_MODEL` to `RET_MODEL`. Use `RET_MODEL` to mine the
training top100 and produce a separate fixed top50 for held-out evaluation. Set
`TEST_QUERIES`, `TEST_CANDIDATES`, and `TEST_QRELS` to the files for the same test
split. The training and test candidate pools need at least 100 and 50 entries,
respectively.

以 `PT_MODEL` 为底座把 Ret SFT 导出到 `RET_MODEL`。用它分别挖训练集 top100、生成测试集固定
top50。测试 query、候选池和 qrels 应属于同一测试集，训练和测试候选池至少分别有 100 和 50 条。

```bash
python examples/seq_contrast_sft/tools/retrieve.py \
  --model "${RET_MODEL}" --queries "${TRAIN_QUERIES}" \
  --candidates "${TRAIN_CANDIDATES}" --instructions "${INSTRUCTIONS}" \
  --image-root "${IMAGE_ROOT}" --top-k 100 --output "${TRAIN_RETRIEVAL}"

python examples/seq_contrast_sft/tools/retrieve.py \
  --model "${RET_MODEL}" --queries "${TEST_QUERIES}" \
  --candidates "${TEST_CANDIDATES}" --instructions "${INSTRUCTIONS}" \
  --qrels "${TEST_QRELS}" --image-root "${IMAGE_ROOT}" \
  --top-k 50 --output "${TEST_RETRIEVAL}"
```

Each retrieval command writes to a new output directory. The held-out Ret Recall
is in `${TEST_RETRIEVAL}/metrics.json`.

每次检索写入新目录。测试集 Ret Recall 在 `${TEST_RETRIEVAL}/metrics.json`。

### 3. Train and evaluate Rank SFT / 训练并评测 Rank SFT

Use `${TRAIN_RETRIEVAL}/rankings.json` as `--rerank-data` in the
[Rank SFT recipe](../lamra_rerank/README.md#prepare-training-data). This prepares
pointwise and listwise training JSONL before submission. Set `RANK_BASE` to
`RET_MODEL` for the three-stage workflow and use a fresh Rank experiment name.
The [Rank evaluation](../lamra_rerank/README.md#evaluate-fixed-retrieval-candidates)
uses the files in `TEST_RETRIEVAL` and the exported Rank model.

Rank 以 `RET_MODEL` 初始化，训练样本来自 `${TRAIN_RETRIEVAL}/rankings.json`。
评测使用 `TEST_RETRIEVAL` 里的固定候选和导出的 Rank 模型。

### Offline export

Use the same command for all three stages. `BASE_FOR_STAGE` is the exact HF
checkpoint the stage loaded. Set `FINAL_CHECKPOINT` to the final `iter_*`
directory under `${SAVE_DIR}/${STAGE_EXP_NAME}` and `UPDATES` to the number of
optimizer updates. Use fresh `ADAPTER_OUT` and `MERGED_OUT` paths.

离线导出时，`BASE_FOR_STAGE` 指向训练该阶段实际加载的 HF checkpoint，`FINAL_CHECKPOINT` 指向
实验目录内最终的 `iter_*`，`UPDATES` 填实际完成的优化器步数。adapter 和合并权重分别写入新目录。

| Stage / 阶段 | `STAGE_EXP_NAME`         | `BASE_FOR_STAGE` | `MERGED_OUT`    |
| ------------ | ------------------------ | ---------------- | --------------- |
| Ret PT       | `${EXP_PREFIX}-ret-pt`   | `${EMB_MODEL}`   | `${PT_MODEL}`   |
| Ret SFT      | `${EXP_PREFIX}-ret-sft`  | `${PT_MODEL}`    | `${RET_MODEL}`  |
| Rank SFT     | `${EXP_PREFIX}-rank-sft` | `${RANK_BASE}`   | `${RANK_MODEL}` |

```bash
ADAPTER_OUT="${SAVE_DIR}/${STAGE_EXP_NAME}/final_adapter"
CUDA_VISIBLE_DEVICES="" python examples/lamra_rerank/tools/export_checkpoint.py \
  --base "${BASE_FOR_STAGE}" --checkpoint "${FINAL_CHECKPOINT}" \
  --expected-updates "${UPDATES}" --adapter-output "${ADAPTER_OUT}" --output "${MERGED_OUT}"
```

The exporter reads the saved LoRA tensors and verifies the FP32 merge and HF
reload. Resume training from the native checkpoint. Use the merged HF directory
for the next stage or evaluation.

导出工具校验 FP32 合并和 HF 回读。续训用原生 checkpoint，下一阶段和评测用合并后的 HF 权重。
