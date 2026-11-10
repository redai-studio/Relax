# Qwen3-VL-8B-Instruct Benchmark

[汇总](../README.md) · [共用评测协议](../evaluation.md)

每次实验复制下面的记录段，插在既有实验之前。开始时填写来源和时间，结束后更新同一段；未确认的字段留空。只提交 Markdown，运行生成的 JSON、轨迹、日志和权重保存在仓库外。

## 待提交：SFT 2 epochs + SimpleVQA

### 实验信息

| 字段                                   | 内容                                                                          |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| Run ID                                 |                                                                               |
| 状态                                   | 准备中，尚未提交正式任务                                                      |
| 实验启动时间 UTC                       |                                                                               |
| 训练开始 / 结束 UTC                    |                                                                               |
| 评测开始 / 结束 UTC                    |                                                                               |
| 代码分支                               | `github/feat/wuhuan/opensearch_v1`                                            |
| 训练 commit（完整 hash）               |                                                                               |
| 评测 commit（完整 hash，按数据集区分） |                                                                               |
| 初始模型与 revision                    | Qwen/Qwen3-VL-8B-Instruct；revision待确认                                     |
| 训练数据与 revision                    | `OpenSearch-VL/Search-VL-SFT-36K`，`2c1c460af4fa15bd63210cbf426a96664b959944` |
| train / val / test 条数                | 34,607 / 32 / 32，远端重新准备并核对，不按长度过滤                            |
| 计划 / 实际 steps、epochs              | 计划2 epochs；实际完成量待运行后记录                                          |
| GPU 型号与数量、TP / PP                | 计划8 GPU；TP/CP由实测候选选择，PP1                                           |
| Global batch / 学习率 / seed           | 32 / 2e-5 / 42                                                                |
| 重算 / optimizer offload               | full / 0                                                                      |
| 保存策略                               | 无中间checkpoint；最终native/HF仅用于评测，成功后清理，不保存optimizer/RNG    |
| SwanLab 训练链接                       |                                                                               |
| SwanLab 评测链接                       |                                                                               |
| 最终模型引用 / SHA256                  |                                                                               |

### 评测协议

| 字段                                   | 内容                                                                                                                          |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Direct Answer / Agentic                | 仅SimpleVQA Direct Answer，无工具；1,013道英文题                                                                              |
| 数据集 revision、文件 SHA256、题目范围 | `037cf89fb6f1212691756b66d1ecde6c2ce89e54` / `9385aeef23b8a44894b4b8818dbcb87a4ea0a6cc327aa13da96207cfc014ac45` / 英文1,013题 |
| 温度 / thinking / 总上下文 / 最大回合  | 0 / 原生Instruct无thinking开关 / 32K / 1                                                                                      |
| 模型和图片处理配置                     |                                                                                                                               |
| 裁判模型与精确版本                     |                                                                                                                               |
| 裁判 prompt 版本                       |                                                                                                                               |
| workers / 裁判 QPS                     | 32 / 0                                                                                                                        |
| 训练集去重方式                         |                                                                                                                               |

### 最终结果

当前流程只评测SimpleVQA。分数单位为 %。失败题计入全部题数；未完成的测试不填正式分数。

| 数据集       | 题数 | 正确数 | Pass@1 | 去重题数 | 去重正确数 | 去重 Pass@1 | 失败数 | 上下文耗尽 |
| ------------ | ---: | -----: | -----: | -------: | ---------: | ----------: | -----: | ---------: |
| SimpleVQA EN |      |        |        |          |            |             |        |            |

### 分析与异常

记录同口径基座对比、失败原因和补判情况。基座未完成时不计算提升；补判复用原模型答案，保留原始错误记录。

### 复现与产物

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/prepared-data
export SAVE_DIR=/path/to/artifacts
export NUM_EPOCHS=2
export GLOBAL_BATCH_SIZE=32
export TP_SIZE=2 PP_SIZE=1 CP_SIZE=1 RECOMPUTE=full
bash scripts/entrypoint/ray-job.sh examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh relax
```

| 产物                   | 可分享的引用 |
| ---------------------- | ------------ |
| 最终 HF 权重及校验信息 |              |
| 数据处理与训练配置     |              |
| 模型来源与评测协议     |              |
| 逐题结果与 summary     |              |
| 运行日志               |              |

## 官方基准参考

这些是官方Qwen3-VL-8B结果，不能替代当前实验的实测基座。当前只评测SimpleVQA，不再跑VDR。

| 阶段                       | SimpleVQA Pass@1（%） | 评测方式               |
| -------------------------- | --------------------: | ---------------------- |
| 初始Qwen3-VL-8B            |                  47.1 | 无工具Direct Reasoning |
| 初始Qwen3-VL-8B            |                  52.0 | 带工具Agentic          |
| 仅SFT                      |                  66.1 | 带工具Agentic          |
| SFT+RL最终OpenSearch-VL-8B |                  71.6 | 带工具Agentic          |

来源：[论文Table 2](https://arxiv.org/html/2605.05185v1#S5.T2)、[Table 3](https://arxiv.org/html/2605.05185v1#S5.T3)。论文采用GPT-4o裁判；本项目按实际配置裁判判分，题目范围与生成预算也须单独对齐。

官方SFT为8 epochs、36,592条、GBS256、学习率2e-5，全参训练含视觉编码器和多模态projector；当前计划2 epochs、七域34,607条训练/32验证/32测试、GBS32，不能称为完整复现该训练量。[官方训练配置Table 4](https://arxiv.org/html/2605.05185v1)

## 性能设置

保留Flash Attention、动态batch、负载均衡、按rank取数与预取、sequence parallel、分块logits loss。32K长序列保留full recompute，不启用CPU optimizer offload，PP为1。候选比较TP2/CP1、TP4/CP1、TP2/CP2；各候选8步、不保存权重，剔除前3步后按稳定吞吐与峰值显存选正式配置。性能数字待实际测量，不使用4B吞吐作为8B结论。正式训练后由最终HF起TP1/DP8推理，只评SimpleVQA。
