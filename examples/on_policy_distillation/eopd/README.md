# EOPD / OPD - Qwen3-0.6B (MATH)

本目录提供 Qwen3-0.6B-Base student、Qwen3-8B teacher 的 EOPD 复现实验，以及使用相同训练配置的 OPD baseline。两个入口共享模型、数据、优化器、rollout、保存和评测参数，便于直接对比 EOPD 门控带来的收益。

## 1. 环境

脚本按单机 8 卡编写，建议直接使用 Relax 标准训练镜像。进入仓库根目录后确认基础依赖可用：

```bash
cd /path/to/Relax
python3 -c "import torch, ray, sglang, relax"
```

如果当前环境尚未安装本仓库，可以先执行：

```bash
python3 -m pip install -e .
```

准备 student 和 teacher checkpoint。默认目录约定如下：

```bash
export EXP_DIR=/root/exps
export MODEL_DIR=${EXP_DIR}
export DATA_DIR=${EXP_DIR}

hf download Qwen/Qwen3-0.6B-Base --local-dir ${MODEL_DIR}/Qwen3-0.6B-Base
hf download Qwen/Qwen3-8B --local-dir ${MODEL_DIR}/Qwen3-8B
```

也可以通过 `STUDENT_MODEL_NAME`、`TEACHER_MODEL_NAME` 或 `TEACHER_MODEL_PATH` 指向已有 checkpoint。

## 2. 准备数据

下载 MATH 训练集和 MATH500 测试集：

```bash
export RAW_DATA_DIR=/root/datasets
mkdir -p ${RAW_DATA_DIR}

hf download DigitalLearningGmbH/MATH-lighteval \
  --repo-type dataset \
  --local-dir ${RAW_DATA_DIR}/MATH-lighteval

hf download HuggingFaceH4/MATH-500 test.jsonl \
  --repo-type dataset \
  --local-dir ${RAW_DATA_DIR}/MATH-500
```

转换为 Relax 使用的 jsonl：

```bash
python3 examples/on_policy_distillation/eopd/prepare_data.py \
  --math-dir ${RAW_DATA_DIR}/MATH-lighteval \
  --math500-jsonl ${RAW_DATA_DIR}/MATH-500/test.jsonl \
  --out-dir ${DATA_DIR}/math-eopd
```

转换后会生成：

- `${DATA_DIR}/math-eopd/math_train.jsonl`
- `${DATA_DIR}/math-eopd/math500_test.jsonl`

## 3. 启动 EOPD

```bash
EXP_DIR=/root/exps MODEL_DIR=/root/exps DATA_DIR=/root/exps \
EOPD=1 bash examples/on_policy_distillation/eopd/run-eopd-qwen3-0.6B-math-8xgpu.sh
```

默认使用 `tau=0.8`、teacher top-k `k=16`、每 50 step 保存、每 100 step 评测。可通过 `EOPD_TAU`、`OPD_TOPK`、`SAVE_INTERVAL`、`EVAL_INTERVAL` 覆盖。

## 4. 启动 OPD baseline

```bash
EXP_DIR=/root/exps MODEL_DIR=/root/exps DATA_DIR=/root/exps \
EOPD=0 bash examples/on_policy_distillation/eopd/run-eopd-qwen3-0.6B-math-8xgpu.sh
```

两种模式共用同一个脚本，只由 `EOPD` 切换：`EOPD=1` 走熵门控 EOPD，`EOPD=0` 走 plain OPD baseline。`EOPD` 没有默认值——不设或写错会直接报错退出，不会静默跑成 baseline；脚本启动时会回显解析出的 MODE 供核对。两组实验默认写入同一个 `PROJECT_NAME=Relax/dev/eopd`，实验名分别以 `eopd-tau...` 和 `opd-baseline...` 开头。

## 5. Smoke test

正式训练前可以先运行 3 个 rollout 的快速检查：

```bash
SMOKE=1 EOPD=1 bash examples/on_policy_distillation/eopd/run-eopd-qwen3-0.6B-math-8xgpu.sh
SMOKE=1 EOPD=0 bash examples/on_policy_distillation/eopd/run-eopd-qwen3-0.6B-math-8xgpu.sh
```

训练日志写入仓库根目录的 `log/`，checkpoint 写入 `${EXP_DIR}/save/`。
