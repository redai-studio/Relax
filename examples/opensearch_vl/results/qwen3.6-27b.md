# Qwen3.6-27B Benchmark

[汇总](../README.md) · [共用评测协议](../evaluation.md) · [模型脚本](../scripts/run-qwen3.6-27B-8xgpu.sh)

新实验插在前方，缺失字段留空。每次实验使用[记录模板](../templates/model-results.md)，运行生成的结构化记录与权重保存在仓库外。

## SFT 2 epochs：训练未完成

### 实验信息

| 字段                         | 内容                                                                                   |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| Run ID                       |                                                                                        |
| 状态                         | 训练已停止，未完成完整基准评测                                                         |
| 实验启动时间 UTC             |                                                                                        |
| 训练开始 / 结束 UTC          |                                                                                        |
| 评测开始 / 结束 UTC          |                                                                                        |
| 代码分支                     | `github/feat/wuhuan/opensearch_v1`                                                     |
| 训练 commit                  | `1c621a84ae894455f77dc3f3d105ff637ad845b5`                                             |
| 评测 commit                  |                                                                                        |
| 初始模型                     | Qwen3.6-27B                                                                            |
| 初始模型公开 revision        |                                                                                        |
| 训练数据                     | `OpenSearch-VL/Search-VL-SFT-36K`，revision `2c1c460af4fa15bd63210cbf426a96664b959944` |
| train / val / test           | 34,607 / 32 / 32                                                                       |
| 计划训练量                   | 2 epochs                                                                               |
| 实际训练量                   |                                                                                        |
| 训练并行度                   | TP4 / PP1 / DP2，full recompute，75% optimizer CPU offload                             |
| GPU 型号与数量               |                                                                                        |
| Global batch / 学习率 / seed | 32 / 2e-5 / 42                                                                         |
| 保存策略                     | 无中间 checkpoint；计划仅在最后导出                                                    |
| SwanLab 训练链接             |                                                                                        |
| SwanLab 评测链接             |                                                                                        |
| 最终模型引用 / SHA256        |                                                                                        |

### 评测协议

计划采用共用 Direct Answer 协议，具体裁判、推理并行度和生成配置待实验确认。该模型的单卡推理容量尚未验证，不直接复用4B的每卡一个副本配置。

### 最终结果

| 数据集       | 题数 | 正确数 | Pass@1 | 去重题数 | 去重 Pass@1 | 失败数 |
| ------------ | ---: | -----: | -----: | -------: | ----------: | -----: |
| SimpleVQA EN |      |        |        |          |             |        |
| VDR-testmini |      |        |        |          |             |        |

### 分析与产物

尚无完整基准分数，不能用其他模型或基础模型的分数代填。

| 产物                   | 可分享的引用 |
| ---------------------- | ------------ |
| 最终 HF 权重及校验信息 |              |
| 逐题结果与 summary     |              |
| 运行日志               |              |
