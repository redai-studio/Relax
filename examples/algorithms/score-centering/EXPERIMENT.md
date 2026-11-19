# Qwen3-0.6B Countdown 实验记录

本记录说明 Countdown 四组对照的复现步骤和实验设置。SC 的通用介绍见[算法参考](../README.md)。四组实验结果见文末。

## 文件

| 文件                      | 用途                                                |
| ------------------------- | --------------------------------------------------- |
| `run-qwen3-0.6B-8xgpu.sh` | 用环境变量切换 RLOO、RLOO+TIS、SC、SC+TIS           |
| `recipe.py`               | 生成数据集、将正确率映射为 ±1 奖励、限制总 token 数 |
| `countdown.py`            | Countdown 题目生成、prompt 和算术判分               |
| `__init__.py`             | Python 包入口                                       |
| `EXPERIMENT.md`           | 复现步骤、配置差异与实验结果                        |

代码基于 [Relax PR #440](https://github.com/redai-studio/Relax/pull/440) 的 `30ac5f4ca0d4b202675d8c102f81df7b67c1ea12`。SGD 兼容、BF16 HF 导出及 checkpoint 预取恢复修复来自独立的 [PR #449](https://github.com/redai-studio/Relax/pull/449)，复现时需包含这些修复。实验设计参考 [verl PR #8010](https://github.com/verl-project/verl/pull/8010)。

## 准备模型与数据

以下命令均从仓库根目录执行。先设置自己的路径：

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/countdown-data
export SAVE_DIR=/path/to/checkpoints
export RAY_ADDRESS=http://ray-head:8266
```

将原始 Qwen3-0.6B BF16 HF 模型放在 `$MODEL_DIR/Qwen3-0.6B`。从同一模型转换 FP8 权重；已有对应模型时可跳过。转换需要 CUDA：

```bash
python scripts/tools/convert_hf_to_fp8.py \
  --model-dir "$MODEL_DIR/Qwen3-0.6B" \
  --save-dir "$MODEL_DIR/Qwen3-0.6B-FP8" \
  --strategy block --block-size 128 128
```

生成固定 Countdown 数据：

```bash
python -m examples.algorithms.score-centering.recipe \
  --model "$MODEL_DIR/Qwen3-0.6B" --output "$DATA_DIR"
```

生成器以 seed=0 构造 20,000 道可解的四数运算题，每个数字在 1–100，整数目标在 1–1000；再用 `Dataset.shuffle(seed=0)` 打乱，前 64 道作为 dev，其余 19,936 道作为 train。训练时不额外 shuffle。

每条 JSONL 包含聊天消息 `prompt`、题目标签 `label` 和 token 预算。构造解只保存在标签中，不放进模型输入。判分检查 `<answer>` 中算式的数字使用和计算结果，正确奖励 +1，否则 −1。数据生成和判分参考作者实现 `martin-marek/score-centering` 的 `envs/countdown.py`，版本为 `1de793964e9cba9487176775fb3e9c7bc8ad7702`。

判分保留固定上游的兼容行为：允许小数点，并按数字片段检查数字使用。因此，这里的 reward 不等价于严格整数操作数语法校验；本轮没有修改奖励口径。

生成器同时输出 `manifest.json`。四组使用完全相同的数据，预期文件指纹如下：

| 文件                    |   条数 | SHA256                                                             |
| ----------------------- | -----: | ------------------------------------------------------------------ |
| `countdown-train.jsonl` | 19,936 | `9749d99af0a552953cc2bc3dcb59edc62d283e9edfed8ab2b47bfbe74a2e63ba` |
| `countdown-dev.jsonl`   |     64 | `e641394a7a98ad67d2b55fb26c283737271bc99b22152eda884cbcab8b2fbf51` |

## 公共配置

| 项目       | 配置                                                                          |
| ---------- | ----------------------------------------------------------------------------- |
| 模型与硬件 | Qwen3-0.6B，单机 8×H20，colocate                                              |
| 并行       | trainer TP/PP/CP=1、DP=8；8 个单卡 rollout engine                             |
| 精度       | BF16 trainer；FP8 W/A sampler、FP8 E4M3 KV cache                              |
| batch      | 64 prompts×8 responses，global batch=512，每批一次参数更新                    |
| 优化器     | SGD，lr=0.01、momentum=0、weight decay=0、不裁剪梯度                          |
| 训练       | seed=0，400 次更新，同步训练、max-staleness=0                                 |
| 采样       | temperature=1、top-p=1、top-k=-1，完整采样 support                            |
| 长度       | chat template 的 enable_thinking=false；prompt+completion 总计最多 512 tokens |
| loss       | KL coefficient=0、entropy coefficient=0，按全局有效 token 归一化              |
| 在线验证   | 训练前及每 25 步，64 道 dev×8 responses                                       |
| 保存       | 每 100 步保存训练 checkpoint 和 BF16 HF 权重；训练 checkpoint 最多保留 1 份   |

`NUM_ROLLOUT`、`SAVE_INTERVAL`、`EVAL_INTERVAL` 可覆盖，对照实验应保持一致。HF 导出是独立产物，训练 checkpoint 的保留数不限制 HF 导出数量。

## 四组配置差异与启动

| USE_SC | USE_TIS | 组别     | 算法参数差异                                                                                  |
| ------ | ------- | -------- | --------------------------------------------------------------------------------------------- |
| 0      | 0       | RLOO     | `--advantage-estimator rloo`                                                                  |
| 0      | 1       | RLOO+TIS | RLOO 加 `--use-tis --tis-clip 2 --tis-clip-low 0`                                             |
| 1      | 0       | SC       | `--advantage-estimator score_centering --score-centering-is none`                             |
| 1      | 1       | SC+TIS   | `--advantage-estimator score_centering --score-centering-is tis --score-centering-tis-clip 2` |

仅 SC 两组额外使用 `--rollout-top-logprobs-num 128 --use-rollout-logprobs --get-mismatch-metrics`。SC+TIS 使用 SC 内部加权，不再叠加原生 `--use-tis`。

依次运行以下命令，上一组结束后再启动下一组：

```bash
# RLOO
USE_SC=0 USE_TIS=0 bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/score-centering/run-qwen3-0.6B-8xgpu.sh

# RLOO + TIS
USE_SC=0 USE_TIS=1 bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/score-centering/run-qwen3-0.6B-8xgpu.sh

# SC
USE_SC=1 USE_TIS=0 bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/score-centering/run-qwen3-0.6B-8xgpu.sh

# SC + TIS
USE_SC=1 USE_TIS=1 bash scripts/entrypoint/ray-job.sh \
  examples/algorithms/score-centering/run-qwen3-0.6B-8xgpu.sh
```

RLOO 使用原生 leave-one-out advantage，SC 使用 PR 自带的 group-mean advantage；组大小为 8 时前者是后者的 8/7。因此这是原生算法配方对照，不是仅开关 SC 的严格消融。没有自定义或替换训练 loss。SC 使用 top-128 候选及尾部近似，不是全词表精确修正。

ClearML 默认 project 为 `Relax/validation/pr440-native-countdown`。默认实验名为 `qwen3-06b-countdown-8gpu-<arm>-s0`，`PROJECT_NAME` 和 `EXP_NAME` 均可覆盖。load/save 固定在 `$SAVE_DIR/$EXP_NAME/checkpoint`；独立重跑需显式更换实验名，不能只更换时间戳。本轮 Megatron 依赖的原生 SGD 完整优化器恢复存在 `init_state_fn=None` 加载错误，PR #449 未覆盖这一问题，因此当前配方不能承诺完整恢复。重新从 HF 起跑应更换 `EXP_NAME` 或 `SAVE_DIR`；`--no-load-optim` 不等价于完整恢复。

Ray 默认阻塞跟随日志，显式设置 `RAY_NO_WAIT=1` 才非阻塞。日志位于 `log/`，由仓库的 `.gitignore` 排除。

## 评估口径

在线验证使用当前 FP8 rollout 引擎，dev 准确率为 `(signed_mean_reward+1)/2`。同时观察生成长度、截断比例、梯度范数和可用的 TIS/SC 指标。

独立 BF16 评估需另行执行并与在线 FP8 结果分开记录；使用相同 dev、64×8 responses、temperature=1、完整采样 support、总长度 512。具体评估入口与随机种子待补齐。

各组按实际完成的更新数和 dev 评估点报告；未完成的训练区间不作外推。

## 实验结果

实验曲线见 [PR #440 中的实验报告](https://github.com/redai-studio/Relax/pull/440#issuecomment-6248536659)。

Qwen3-0.6B，8×H20，BF16 trainer / FP8 sampler，seed=0。下表为在线 FP8 dev 准确率：64 道题各采样 8 次，共 512 个回答。

| 方法     | 实际训练步数 | 第 25 步 | 第 50 步 | 最高准确率（步数） | 最后准确率（步数） | 训练表现                                                  |
| -------- | -----------: | -------: | -------: | ------------------ | ------------------ | --------------------------------------------------------- |
| RLOO     |           51 |    0.00% |    0.20% | 3.71%（初始）      | 0.20%（50）        | 早期质量崩塌，第 50 步截断比例 93.36%                     |
| RLOO+TIS |           61 |   28.32% |   34.38% | 34.38%（50）       | 34.38%（50）       | 训练 reward 崩塌，提前结束；表中准确率为最后一次 dev 评估 |
| SC       |          400 |   30.66% |   43.95% | 60.55%（225）      | 54.30%（400）      | 完成 400 步，最后三次评估保持在 54%–56%                   |
| SC+TIS   |          400 |   26.17% |   36.52% | 58.98%（325）      | 58.40%（400）      | 完成 400 步，最后三次评估保持在 56%–59%                   |

同为第 50 步，SC 比 TIS 高 9.57 个百分点。SC 和 SC+TIS 均完成 400 步，最终准确率分别为 54.30% 和 58.40%；SC+TIS 最终高 4.10 个百分点，但全程最高准确率为 SC 的 60.55%。两组基线未跑满 400 步，因此不能比较四组的最终收敛效果。本轮仅一个 seed，RLOO 与 SC 的 advantage 还存在前述 8/7 尺度差异；独立 BF16 评估尚未执行。
