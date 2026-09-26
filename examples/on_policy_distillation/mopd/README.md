# Multi-Teacher On-Policy Distillation (MOPD)

## 概述

MOPD (Multi-Teacher On-Policy Distillation) 实现多教师在线蒸馏，允许一个学生模型同时从多个异构教师模型中学习。每个教师专注于不同的数据领域（如文本数学、视觉几何），框架按样本的 `data_source` 把它路由到对应教师，请求 token 级 log-probability，实现跨模态、跨领域的知识传递。

教师由 **Relax 托管**：只需通过 `--opd-teacher-routes` 提供 `{data_source: HF 权重路径}`，Relax 会为每个教师自动拉起一个专用 SGLang 服务，运行时注入真实 URL。**无需手动 `sglang.launch_server`、无需管理端口。**

本示例场景：

- **文本教师** (Qwen3-4B-Instruct-2507 / Qwen3.6-27B)：处理 GSM8K / dapo-math-17k 文本数学题
- **视觉教师** (Qwen3-VL-4B-Instruct / Qwen3.5-27B)：处理 Geometry3K / multimodal-open-r1 视觉几何题
- **学生** (Qwen3-VL-2B-Instruct / Qwen3.5-35B-A3B)：在两个领域上同时训练

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

## 提供的脚本

| 脚本                                        | 规模                           | GPU 分配（actor / rollout / teacher）       |
| ------------------------------------------- | ------------------------------ | ------------------------------------------- |
| `run-mopd-qwen3-vl-2b-8xgpu-colocate.sh`    | 学生 2B，单机 8 卡，教师多副本 | 8 / 4 / 4（2 教师 × 2 副本，各 TP=1）       |
| `run-mopd-qwen35-9b-8xgpu-colocate.sh`      | 学生 9B，单机 8 卡             | 8 / 4 / 4（2 教师各 TP=2，1 副本）          |
| `run-mopd-qwen35-35ba3b-16xgpu-colocate.sh` | 学生 35B-A3B，2 节点 16 卡     | 16 / 8 / 8（TP=4 PP=2 EP=8；2 教师各 TP=4） |

### 8 卡布局（`run-mopd-qwen3-vl-2b-8xgpu-colocate.sh`，教师多副本）

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

每个 teacher 内部按 `--teacher-num-gpus-per-engine` 拆成多副本，round-robin 分担同一 data_source 的请求量。

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

### Step 2: 合并数据

```bash
python examples/on_policy_distillation/mopd/prepare_data.py \
    --gsm8k-dir /root/gsm8k \
    --geo3k-dir /root/geo3k \
    --dapo-math-path /root/dapo-math-17k/dapo-math-17k.jsonl \
    --openr1mm-path /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
    --output-dir /root/data/MOPD \
    --test-ratio 0.05
```

输出 `train.parquet` / `test.parquet` / `test_small.parquet`（按 data_source 均衡采样，训练中快速 eval 用）。`--output-dir` 需要和训练脚本的 `PROMPT_SET`/`DATA_DIR` 对上，对不上直接用 `PROMPT_SET`/`EVAL_SET` 环境变量覆盖即可。

### Step 3: 下载模型

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /root/Qwen3-VL-2B-Instruct
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

## 启动训练

```bash
MODEL_DIR=/root/models DATA_DIR=/root/data bash run-mopd-qwen3-vl-2b-8xgpu-colocate.sh
MODEL_DIR=/root/models DATA_DIR=/root/data bash run-mopd-qwen35-35ba3b-16xgpu-colocate.sh
```

框架会自动为每个教师拉起托管 SGLang 服务，健康检查通过后开始训练。

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

## 奖励路由：`rm_type=mopd`

所有数据源的 `extra_info.rm_type` 都固定为 `"mopd"`，统一分发到 `get_mopd_reward`，它再按 `data_source` 内部路由到具体打分函数，并**始终返回 float**。不要把某个数据源改成底层的 `"dapo"`/`"openr1mm"` 直接路由——它们返回值类型不一致（`dapo` 是 dict，其余是 float），混在同一 batch 会在 eval 聚合时报 `TypeError`。

## MOPD 专属参数

| 参数                            | 说明                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------- |
| `--opd-teacher-routes`          | `{data_source: HF 权重路径}` JSON，托管多教师                                |
| `--opd-teacher-key`             | 路由键 metadata 字段（默认 `data_source`）                                   |
| `--teacher-num-gpus-per-engine` | 每个教师 SGLang 引擎的 GPU 数（=教师 TP）。教师副本数 = 该教师 GPU 数 / 此值 |

### 稳定性注意

- `--colocate --offload` 必须同时开启，否则教师/rollout 训练时不换出显存，和 actor 抢卡导致 OOM。
- `ROLLOUT_GPUS + TEACHER_GPUS` 必须精确等于 `ACTOR_GPUS`，启动时报错而非运行时才 OOM。
- 35B 级别 student 建议把 `ACTOR_GPUS` 开够（本例 16 卡）：显存不够时 `optimizer.step()` 的 grad-norm all-reduce 会报 `ncclUnhandledCudaError` 而不是常见的 `torch.OutOfMemoryError`，容易被误判成别的问题。

## 扩展到 N 个教师

1. 用 `prepare_data.py` 为新领域样本生成带对应 `data_source` 的统一 schema 数据（或用 `--manifest` 接入已有 verl 格式数据）。
2. 在 `--opd-teacher-routes` 里加一条 `data_source → 权重路径`。
3. 增大 `--resource` 里 `teacher` 的 GPU 总预算，并**同步缩小 `--rollout-num-gpus`**（或整体扩大 `actor_gpus`），保证 `rollout + teacher == actor` 始终成立——新增教师不能凭空多占卡。

## 参考

- [On-Policy Distillation - Relax Docs](../README.md)
- [OPD 参数配置示例](../README.md#common-combinations)
