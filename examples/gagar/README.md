# GAGAR：等价 group reward 转换

`gagar.py` 实现 [GAGAR 论文](https://arxiv.org/abs/2609.32577v1) Appendix A.2 / A.3 的二值奖励优势重分配及 reward-space 等价形式。它只接收 outcome 和已经获得的质量权重，不耦合评分模型、sandbox 或训练启动方式。

## 接口

```python
from examples.gagar.gagar import compute_group_rewards

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

## 验证

```bash
pytest -q tests/examples/test_gagar.py
```

测试直接经过 Relax 的生产组归一化函数，覆盖 cap 前后、零权重、全通过/全失败、候选顺序和不修改输入。
