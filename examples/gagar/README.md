# GAGAR：等价 group reward 转换

`core/gar.py` 实现 [GAGAR 论文](https://arxiv.org/abs/2609.32577v1) Appendix A.2 / A.3 的二值奖励优势重分配及 reward-space 等价形式。它只接收 outcome 和已经获得的质量权重，不耦合评分模型、sandbox 或训练启动方式。

## 接口

```python
from examples.gagar.core.gar import compute_group_rewards

scores = compute_group_rewards(
    rewards=[1, 1, 1, 0],
    weights=[0.3, 0.3, 0.4, 0],
    cap=1.5,
)
# [0.8625, 0.8625, 0.9, 0]
```

同一组的候选顺序必须对齐。输入是完整有效组的二值 outcome；通过候选的质量权重在 `[0, 1]`，失败候选的权重忽略。权重不要求和为 1，允许零和并列；模型输出解析、非法轨迹过滤、作弊 outcome 修正和评分失败回退由调用方负责。

## 计算过程

令 P 为通过候选集合，mean(r) 为原始组均值：

```text
a[i] = r[i] - mean(r)
scale = min(sum(a[i] for i in P) / sum(w[i] * a[i] for i in P), cap)
B[i] = scale * w[i] * a[i]   # passing
B[i] = a[i]                 # failing
score[i] = B[i] + mean(r)
```

随后框架减去新的组均值，得到 `score[i] - mean(score) = B[i] - mean(B)`。

上例的逐步结果：

| 阶段                                 |  候选 0 |  候选 1 |  候选 2 |   候选 3 |
| ------------------------------------ | ------: | ------: | ------: | -------: |
| 原始 outcome                         |       1 |       1 |       1 |        0 |
| 原始优势 `a`                         |    0.25 |    0.25 |    0.25 |    -0.75 |
| 质量折扣后                           |   0.075 |   0.075 |     0.1 |    -0.75 |
| 公共系数 `scale=min(3,1.5)` 后的 `B` |  0.1125 |  0.1125 |    0.15 |    -0.75 |
| 返回 group reward                    |  0.8625 |  0.8625 |     0.9 |        0 |
| Relax 减去新均值后的优势             | 0.20625 | 0.20625 | 0.24375 | -0.65625 |

cap 生效时不再补回全部正优势；最终减均值也会改变失败候选的优势。默认 `1.5` 对应论文 Flash 配置，并非禁止使用其他 cap。通过候选权重全零时直接令其 `B=0`，避免除零；最终中心化后这些候选仍可能有正优势。全通过或全失败组返回原 outcomes，中心化后为零。

## 接入 Relax

在自定义 reward 函数里调用该函数，并使用现有 `--group-rm` 传入整组。采用 `--disable-grpo-std-normalization`，保留 reward mean centering；不启用额外 advantage whitening，不裁剪返回分数（可能超过 1）。在同一完整有效组上满足这些条件，才得到上述最终优势。

原始 0/1 outcome 应另存用于通过率、过滤和 baseline 比较，不能用转换后的训练分数代替。该模块没有包含完整 agentic grader，也没有声称训练收益已被验证。

## Agentic grader demo（Draft）

两种执行环境、各自的 prompts 和安装步骤见 [DEMO.md](DEMO.md)。该 demo 待审阅，训练收益尚未验证。

目录：`core/gar.py` 包含 GAR 公式和 reward 记录；`core/grs.py` 包含共享 agentic grader 与环境接口。`app/code_generate/` 包含代码生成 baseline 的数据/判题适配和 GAGAR 环境；`app/minisweagent/` 包含 SWE GAGAR 实现。训练脚本和 `run_grs.sh` 位于本目录。历史运行回放保留在 [PR #446](https://github.com/redai-studio/Relax/pull/446) 正文。

两种任务的 system / instance prompt、agent 预算、环境参数与 GenRM 推理配置统一在 [grader.yaml](grader.yaml)。设置 `GAGAR_CONFIG` 可指定另一份相同结构的 YAML；模型路径、镜像位置等节点配置继续使用原有环境变量。
