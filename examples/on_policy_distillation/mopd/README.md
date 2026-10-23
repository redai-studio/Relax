# Multi-Teacher On-Policy Distillation (MOPD)

## 概述

MOPD (Multi-Teacher On-Policy Distillation) 实现多教师在线蒸馏，允许一个学生模型同时从多个异构教师模型中学习。每个教师专注于不同的数据领域（如文本数学、视觉几何），框架按样本的 `data_source` 把它路由到对应教师，请求 token 级 log-probability，实现跨模态、跨领域的知识传递。

教师由 **Relax 托管**：只需通过 `--opd-teacher-routes` 提供 `{data_source: HF 权重路径}`，Relax 会为每个教师自动拉起一个专用 SGLang 服务，运行时注入真实 URL。**无需手动 `sglang.launch_server`、无需管理端口。**

本示例场景：

- **文本教师** (Qwen3-4B-Instruct-2507 / Qwen3.6-27B)：处理 GSM8K / dapo-math-17k 文本数学题
- **视觉教师** (Qwen3-VL-4B-Instruct / Qwen3.5-27B)：处理 Geometry3K / multimodal-open-r1 视觉几何题
- **学生** (Qwen3-VL-2B-Instruct / Qwen3.5-35B-A3B)：在两个领域上同时训练

______________________________________________________________________

## MOPD 支持 colocate 模式

MOPD teacher 与 student 的 actor/rollout 共享同一个 GPU 卡池（placement group），二者按 bundle 分区、随训练 lock-step offload/onload：

```
actor 卡池（一整块 placement group，大小 = ACTOR_GPUS）
├── rollout 区   bundle [0, ROLLOUT_GPUS)                     ← student rollout
└── teacher 区   bundle [ROLLOUT_GPUS, ACTOR_GPUS)             ← 各 teacher 依次错位排布
```

- **训练阶段**：actor 用满整块卡池训练 student。
- **rollout 阶段**：student rollout 占前段 bundle，teacher 占后段 bundle，两边都从 offload 状态 onload 回显存。
- **硬约束（框架强制校验）**：`ROLLOUT_GPUS + TEACHER_GPUS == ACTOR_GPUS`，不满足会在启动时直接抛 `ValueError`。
- `--colocate --offload` 必须同时开启；`--rollout-num-gpus` 必须等于 `ROLLOUT_GPUS`。

______________________________________________________________________

## 提供的脚本

| 脚本                                                    | 规模                           | GPU 分配（actor / rollout / teacher）       | 特性说明                                      |
| ------------------------------------------------------- | ------------------------------ | ------------------------------------------- | --------------------------------------------- |
| `run-mopd-qwen3-vl-2b-8xgpu-colocate.sh`                | 学生 2B，单机 8 卡，教师多副本 | 8 / 4 / 4（2 教师 × 2 副本，各 TP=1）       | 基础多教师蒸馏，单一混合流式数据源            |
| `run-mopd-qwen3-vl-2b-source-mixture-8xgpu-colocate.sh` | 学生 2B，单机 8 卡，配比采样   | 8 / 4 / 4（2 教师 × 2 副本，各 TP=1）       | **固定配额配比采样**（GSM8K : Geo3K = 3 : 1） |
| `run-mopd-qwen35-9b-8xgpu-colocate.sh`                  | 学生 9B，单机 8 卡             | 8 / 4 / 4（2 教师各 TP=2，1 副本）          | 单机中等规模学生蒸馏                          |
| `run-mopd-qwen35-35ba3b-16xgpu-colocate.sh`             | 学生 35B-A3B，2 节点 16 卡     | 16 / 8 / 8（TP=4 PP=2 EP=8；2 教师各 TP=4） | 多机大模型 MoE 蒸馏                           |

### 8 卡布局示例（教师多副本）

```
┌───────────────────────────────────────────────────────────────┐
│                Single Node, shared 8-GPU pool                 │
├───────────────┬───────────────┬───────────────┬───────────────┤
│  bundle 0-3   │  bundle 4-5   │  bundle 6-7                   │
│ student rollout│ GSM8K teacher │ Geo3K teacher                │
│   (4 GPU)      │ 2 replicas    │ 2 replicas                   │
│                │ (TP=1 each)   │ (TP=1 each)                  │
├────────────────┴───────────────┴──────────────────────────────┤
│        training: actor uses ALL 8 bundles                     │
└───────────────────────────────────────────────────────────────┘
```

每个 teacher 内部按 `--teacher-num-gpus-per-engine` 拆成多副本，round-robin 分担同一 `data_source` 的请求量。

______________________________________________________________________

## 数据准备

用 `prepare_data.py`（本目录下）把原始数据转成统一 schema：`prompt`（str）、`label`（str）、`data_source`（str，路由键）、`images`（VL 内嵌 base64，可选）、`extra_info`（固定 `{"rm_type": "mopd", "source": ...}`）。

### Step 1: 下载数据

```bash
pip install datasets pandas hf

# GSM8K + Geometry3K
python -c "
from datasets import load_dataset
load_dataset('openai/gsm8k', 'main', split='train').to_parquet('/root/gsm8k/train.parquet')
load_dataset('hiyouga/geometry3k', split='train').to_parquet('/root/geo3k/train.parquet')
"

# dapo-math-17k + multimodal-open-r1-8k-verified
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified --local-dir /root/multimodal-open-r1-8k-verified
```

### Step 2: 转换数据

根据训练模式选择数据准备方式：

#### 方案 A：单一混合数据集（常规 MOPD 脚本使用）

```bash
python examples/on_policy_distillation/mopd/prepare_data.py \
    --gsm8k-dir /root/gsm8k \
    --geo3k-dir /root/geo3k \
    --dapo-math-path /root/dapo-math-17k/dapo-math-17k.jsonl \
    --openr1mm-path /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
    --output-dir /root/data/MOPD \
    --test-ratio 0.05
```

输出 `train.parquet` / `test.parquet` / `test_small.parquet`（按 data_source 均衡采样，训练中快速 eval 用）。

#### 方案 B：分源独立数据集（Source Mixture 配比采样脚本使用）

每个数据源单独切分并保留完整的 `train.parquet`，供配比采样引擎独立读取：

```bash
python examples/on_policy_distillation/mopd/prepare_data.py \
    --gsm8k-dir /root/gsm8k --output-dir /root/data/MOPD/gsm8k
python examples/on_policy_distillation/mopd/prepare_data.py \
    --geo3k-dir /root/geo3k --output-dir /root/data/MOPD/geometry3k
```

### Step 3: 下载模型

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /root/Qwen3-VL-2B-Instruct
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

______________________________________________________________________

## 启动常规训练

```bash
# 单机 8 卡 2B 学生基线
MODEL_DIR=/root/models DATA_DIR=/root/data bash examples/on_policy_distillation/mopd/run-mopd-qwen3-vl-2b-8xgpu-colocate.sh

# 双机 16 卡 35B-A3B 学生
MODEL_DIR=/root/models DATA_DIR=/root/data bash examples/on_policy_distillation/mopd/run-mopd-qwen35-35ba3b-16xgpu-colocate.sh
```

______________________________________________________________________

## 数据源批次配比采样 (Source Mixture)

在多领域多教师场景中，不同领域的数据集规模往往极不均衡（例如几何数据仅数千条，而纯文本数学达数十万条）。如果直接混合洗牌，小数据集会被瞬间稀释甚至抽样耗尽。

**Source Mixture (Batch-level 配比采样)** 提供了严格的 Batch 内配额保证：无论各数据源总容量差异多大，每个 Global Batch 内各来源的样本数量均精确锁定为预设比例，同时每个数据源独立维护其 Epoch 循环与 Shuffle 游标。

### 1. 核心机制与配额计算

- **严格的 Batch 级配额**：
  训练引擎在进入推理前校验输入的 Source 构成，并在送入训练队列前再次校验完成的 Batch。杜绝比例漂移与异步抢占。
- **最大余数法（Hare-Niemeyer Method）配额分配**：
  权重配置无需整除 Batch Size。算法先将各源理论份额向下取整，剩余名额按小数余数从大到小依次补齐（余数相同时按配置文件中声明的顺序确定）：
  - **Batch Size = 32, 权重 3 : 1** $\\rightarrow$ 严格分配 24 : 8；
  - **Batch Size = 32, 权重 3 : 2** $\\rightarrow$ 理论份额 19.2 : 12.8，整数分配 19 : 13；
  - **Batch Size = 10, 权重 1 : 1 : 1** $\\rightarrow$ 整数分配 4 : 3 : 3。
- **独立 Epoch 游标**：
  每个数据源拥有独立的 `IndexManager`，互不影响；即使某个小数据源提前刷完一整轮，也会自动完成该源的独立 Shuffle 并进入下一 Epoch，不影响其它数据源进度。

> \[!NOTE\]
> 若某个来源由于权重极小在当前 Batch Size 下计算出的配额为 0，框架在启动时会打印 Warning 提醒，该来源游标将不推进，对应教师资源仍按 `--opd-teacher-routes` 分配。如需该来源参与训练，需增大 Batch Size 或调高其权重。

### 2. 配置文件规范 (`source-mixture.yaml`)

配比配置采用 YAML 格式，路径支持环境变量（如 `${DATA_DIR}`）动态展开：

```yaml
sources:
  openai/gsm8k:
    path: ${DATA_DIR}/MOPD/gsm8k/train.parquet
    weight: 3
  hiyouga/geometry3k:
    path: ${DATA_DIR}/MOPD/geometry3k/train.parquet
    weight: 1
```

> \[!IMPORTANT\]
>
> - `sources` 下的键名（如 `openai/gsm8k`）必须与 `--opd-teacher-routes` 的 key 以及样本数据内的 `data_source` 字段**完全一致**。
> - 支持通过环境变量 `SOURCE_MIXTURE_CONFIG=/path/to/custom.yaml` 覆盖默认配置文件路径。

### 3. 运行配比采样训练

在已安装 Relax / Megatron 环境中，连接现有 Ray 集群启动验证脚本：

```bash
MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints \
    bash examples/on_policy_distillation/mopd/run-mopd-qwen3-vl-2b-source-mixture-8xgpu-colocate.sh
```

- 该脚本与基线 `run-mopd-qwen3-vl-2b-8xgpu-colocate.sh` 保持一致的 GPU 拓扑（8 卡，学生 4 卡，两教师各 2 卡）；
- 默认 `rollout-batch-size = global-batch-size = 32`，每条 prompt 生成 1 个回答，纯蒸馏验证（`--opd-only-reward`）；
- `PROJECT_NAME` 默认 `Relax/recipes/mopd`，`EXP_NAME` 默认 `mopd-qwen3-vl-2b-source-mixture-8xgpu-colocate`；
- Checkpoint 保存于 `${SAVE_DIR}/${EXP_NAME}`，每 5 步保存一份并滚动保留 2 份。

### 4. 断点续训与指纹校验 (Checkpoint Fingerprint)

为了在大规模分布式训练断点恢复时既保证**数据确定性**，又避免因云盘重新挂载导致的时间戳变化误报失败，Source Mixture 使用内容指纹与游标快照：

- **XXH3-128 高性能内容摘要**：
  初始化时复用 `datasets` 自带的 `xxhash` 库，以 1 MiB 分块读取数据文件并计算 128 位哈希。初始化完成后不再重复扫描文件。
- **解耦物理修改时间（mtime）**：
  只改变文件修改时间戳（如重新下载或 `touch`）、搬迁至新目录但内容字节一致的数据文件，只要相对配置和内容不变即可顺利恢复。
- **游标快照机制**：
  Rollout 预取下一个 batch 不会覆盖当前已完成 batch 的游标状态，状态字典仅保存对应已提交 batch 的游标，保证精确恢复。

### 5. 关键设计约束

1. **严格配额保护**：由于每个 Batch 必须满足硬性配额，当前模式下**禁止开启动态样本过滤（Dynamic Filter）**与**请求丢弃（Abort）**。
2. **沿用 StreamingDataset 容错**：样本处理或预取失败会记录日志并返回 `None`；长度检查自身发生异常时记录警告并放行样本。数据读取和来源校验错误在直接读取时抛出，在预取时按预取组件的原有规则处理。
3. **同源补齐**：处理失败或超过 `rollout_max_prompt_len` 的样本返回 `None` 后，从**同一数据源**继续取样补齐配额；来源始终没有可用样本时有界报错。

______________________________________________________________________

## 路由机制

1. 每条样本的 metadata 中带 `data_source` 字段（由数据管线自动注入）。
2. `--opd-teacher-key data_source` 指定路由键。
3. `--opd-teacher-routes` 定义 `data_source → HF 权重路径` 映射：

```json
{
  "openai/gsm8k": "/root/Qwen3-4B-Instruct-2507",
  "hiyouga/geometry3k": "/root/Qwen3-VL-4B-Instruct"
}
```

rollout 阶段按每条样本的 `data_source` 找到对应教师，请求 token 级 log-probability；某个教师请求失败或超时会逐样本回退到 rollout log-probs，仅当一整步内所有非空样本都失败才报错中断。

> `--resource` 里 `teacher` 的 GPU 数是教师总预算，先在教师间平均分配（要求整除），每个教师再按 `--teacher-num-gpus-per-engine` 拆成若干副本（副本 TP=该值），rollout 时先路由到教师再在其副本间 round-robin。

______________________________________________________________________

## 奖励路由：`rm_type=mopd`

所有数据源的 `extra_info.rm_type` 都固定为 `"mopd"`，统一分发到 `get_mopd_reward`，它再按 `data_source` 内部路由到具体打分函数，并**始终返回 float**。不要把某个数据源改成底层的 `"dapo"`/`"openr1mm"` 直接路由——它们返回值类型不一致（`dapo` 是 dict，其余是 float），混在同一 batch 会在 eval 聚合时报 `TypeError`。

______________________________________________________________________

## MOPD 专属参数

| 参数                            | 说明                                                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--opd-teacher-routes`          | `{data_source: HF 权重路径}` JSON，托管多教师路由映射                                                                     |
| `--opd-teacher-key`             | 路由键 metadata 字段（默认 `data_source`）                                                                                |
| `--teacher-num-gpus-per-engine` | 每个教师 SGLang 引擎的 GPU 数（=教师 TP）。教师副本数 = 该教师 GPU 数 / 此值                                              |
| `--rollout-source-mixture`      | 数据源配比 YAML 配置文件路径（开启 Source Mixture 模式）。若设置，`data_source_path` 自动绑定为 `SourceMixtureDataSource` |
| `--source-mixture-spec`         | （内部解析生成）各数据源的规范列表（包含名称、路径及原始权重）                                                            |
| `--source-mixture-quotas`       | （内部计算生成）基于最大余数法计算出的单批次各数据源整数配额字典，如 `{"openai/gsm8k": 24, "hiyouga/geometry3k": 8}`      |

### 稳定性注意

- `--colocate --offload` 必须同时开启，否则教师/rollout 训练时不换出显存，和 actor 抢卡导致 OOM。
- `ROLLOUT_GPUS + TEACHER_GPUS` 必须精确等于 `ACTOR_GPUS`，启动时报错而非运行时才 OOM。
- 35B 级别 student 建议把 `ACTOR_GPUS` 开够（本例 16 卡）：显存不够时 `optimizer.step()` 的 grad-norm all-reduce 会报 `ncclUnhandledCudaError` 而不是常见的 `torch.OutOfMemoryError`，容易被误判成别的问题。

______________________________________________________________________

## 扩展到 N 个教师

1. 用 `prepare_data.py` 为新领域样本生成带对应 `data_source` 的统一 schema 数据（或用 `--manifest` 接入已有 verl 格式数据）。
2. 在 `--opd-teacher-routes` 里加一条 `data_source → 权重路径`。
3. 若使用 Source Mixture，在 `source-mixture.yaml` 中添加该数据源及其采样权重。
4. 增大 `--resource` 里 `teacher` 的 GPU 总预算，并**同步缩小 `--rollout-num-gpus`**（或整体扩大 `actor_gpus`），保证 `rollout + teacher == actor` 始终成立——新增教师不能凭空多占卡。

______________________________________________________________________

## 参考

- [On-Policy Distillation - Relax Docs](../README.md)
- [OPD 参数配置示例](../README.md#common-combinations)
