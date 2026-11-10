# SimpleVQA 与 VDR 公开基准参考

[评测运行方式](evaluation.md)

以下保留 OpenSearch-VL 的带工具对照和纯 SFT 消融，并补充无工具参考。分数均为百分数；不同模型、工具、裁判和题集的结果分别报告。来源核对日期：2026-11-10。

## 无工具：当前实验优先参考

[OpenSearch-VL v1 Table 2](https://arxiv.org/html/2605.05185v1#S5.T2) 的 Direct Reasoning 只根据图片、问题与模型自身知识作答，由 GPT-4o 判断最终答案：

| 模型             | SimpleVQA | VDR |
| ---------------- | --------: | --: |
| Qwen3-VL-8B      |      47.1 | 2.8 |
| Qwen3-VL-30B-A3B |      53.2 | 3.8 |
| Qwen3-VL-32B     |      58.0 | 4.1 |
| GPT-4o           |      51.7 | 1.7 |
| GPT-5            |      61.6 | 9.8 |
| Gemini-2.5-Pro   |      63.0 | 8.0 |

此外，[Qwen3.5-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B#vision-language) 给出的 SimpleVQA 是 **43.4%**；同表 Qwen3.5-9B 为 51.2%。该表没有 VDR 分数，也没有足以对齐本例题集、裁判和生成预算的完整说明，因此不能用 43.4% 替代本机基座实测。

## 带工具：Agentic Workflow

以下是此前选定的 [OpenSearch-VL v1 Table 2](https://arxiv.org/html/2605.05185v1#S5.T2) 对照。模型自主搜索、读取和操作图片；OpenSearch-VL 为 **SFT+RL 最终模型**。这些数字保留为工具模式目标，不能直接作为当前关闭工具实验的排名。

| 模型                  | SimpleVQA |  VDR | MMSearch | LiveVQA | BrowseComp-VL | FVQA | InfoSeek | 平均 |
| --------------------- | --------: | ---: | -------: | ------: | ------------: | ---: | -------: | ---: |
| Qwen3-VL-8B           |      52.0 | 17.0 |     37.4 |    50.6 |          27.9 | 58.7 |     50.3 | 42.0 |
| OpenSearch-VL-8B      |      71.6 | 20.8 |     64.5 |    59.6 |          37.6 | 71.5 |     70.2 | 56.6 |
| Qwen3-VL-30B-A3B      |      55.1 | 20.2 |     44.2 |    62.0 |          34.1 | 63.0 |     56.2 | 47.8 |
| OpenSearch-VL-30B-A3B |      74.9 | 33.5 |     68.7 |    67.4 |          41.1 | 73.2 |     72.4 | 61.6 |
| Qwen3-VL-32B          |      58.7 | 23.1 |     53.9 |    45.5 |          35.1 | 61.2 |     58.5 | 48.0 |
| OpenSearch-VL-32B     |      76.2 | 33.8 |     72.3 |    70.5 |          43.8 | 74.7 |     74.8 | 63.7 |

## 纯 SFT：与训练阶段相关的消融

[OpenSearch-VL v1 Table 3](https://arxiv.org/html/2605.05185v1#S5.T3) 仍在带工具的 agent 评测下比较训练阶段：

| Qwen3-VL-8B 阶段 | SimpleVQA | InfoSeek | FVQA | 三项平均 |
| ---------------- | --------: | -------: | ---: | -------: |
| 训练前，带工具   |      52.0 |     50.3 | 58.7 |     53.7 |
| 仅 SFT，带工具   |      66.1 |     62.4 | 65.3 |     64.6 |

这里的“纯 SFT”指没有接 RL，不能理解成关闭工具。该表未报告纯 SFT 的 VDR，也没有 Qwen3.5-4B 结果。

## VDR-testmini 的题集口径

[VDR-Bench 论文](https://arxiv.org/html/2602.02185v1) 定义的是 2,000 题完整基准。[官方评测说明](https://github.com/Osilly/Vision-DeepResearch/blob/main/rllm/eval/README.md#datasets) 另提供 `testmini500_v1.parquet`，共 500 题；同目录的 `simplevqa_300.parquet` 只有 300 题。本例实际使用 SimpleVQA 全部英文 **1,013 题**和 VDR-testmini **500 题**，固定下载 revision 见[评测文档](evaluation.md#评测数据与去重)。

官方仓库的 [Performance 表](https://github.com/Osilly/Vision-DeepResearch#performance) 也提供 VDR Direct Answer：Qwen3-VL-8B-Instruct 2.8%、Qwen3-VL-8B-Thinking 5.6%、Qwen3-VL-30B-A3B-Instruct 3.8%。它有助于理解 thinking/agent 配置的区别；主表未逐行绑定题目 ID 或 parquet hash，不能仅凭列名“VDR”认定与本例 500 题完全一致。

## 如何比较本项目结果

当前协议为无工具、温度 0、thinking 开启、32K 输入输出总预算、每题一次生成，使用配置指定的裁判模型并保存其实际型号。参考答案只给裁判；超限、无答案和异常题均计入分母，另报排除 SFT train 重叠题后的分数。

公开分数用于判断大致量级。严格判断 SFT 的收益，需要同一权重来源、题目 ID、图片处理、提示、生成设置、裁判及去重口径下的基座和训练后两组实测；不能把公开基座分数与本项目 SFT 分数相减，或将无工具结果与上方 agent 分数混排。当前比较脚本还会检查 workers、QPS 和 evaluator source hash，历史协议不同的结果不能直接自动合并。

本项目成绩与来源证据保存在[Qwen3.5-4B 实验记录](results/qwen3.5-4b.md)，公开参考与本项目实测分开保存。

## 官方8B的SFT训练量

官方8B（以及32B、30B-A3B）SFT配置为8 epochs、36,592条、全局batch256、学习率2e-5，视觉编码器与多模态projector均参与全参训练；这是训练预算，不应与本项目短程实验或2epoch计划混淆。[论文附录C.1/Table 4](https://arxiv.org/html/2605.05185v1)
