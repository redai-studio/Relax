# Qwen3-VL-8B-Instruct Benchmark

[汇总](../README.md) · [共用评测协议](../evaluation.md)

每次实验复制下面的记录段，插在既有实验之前。开始时填写来源和时间，结束后更新同一段；未确认的字段留空。只提交 Markdown，运行生成的 JSON、轨迹、日志和权重保存在仓库外。

## 2026-11-10：SFT 2 epochs + SimpleVQA，匹配运行环境重试

| 字段                         | 内容                                                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| Run ID                       | `qwen3-vl-8b-20261110-113956`                                                           |
| 状态                         | 调优已完成；canary训练与导出通过，看板SDK兼容性修复中，正式训练尚未开始                 |
| 实验启动时间 UTC             | 2026-11-10 11:39:56                                                                     |
| 训练开始 / 结束 UTC          |                                                                                         |
| 评测开始 / 结束 UTC          |                                                                                         |
| 代码分支                     | `github/feat/wuhuan/opensearch_v1`                                                      |
| 训练 commit（完整 hash）     | `a045c20e25f34302bc61675e978ee7c19ce2dda2`                                              |
| 初始模型与 revision          | Qwen/Qwen3-VL-8B-Instruct；revision待确认                                               |
| 训练数据                     | 七域全量34,607 / 32 / 32；不按长度预过滤                                                |
| 计划 / 实际训练预算          | 2 epochs，预期2,162步；实际待完成                                                       |
| GPU与并行                    | 8 × SM103，TP1 / PP1 / CP1 / DP8                                                        |
| Global batch / 学习率 / seed | 32 / 2e-5 / 42                                                                          |
| 重算 / offload               | none（比测选定） / offload0                                                             |
| 运行环境                     | 匹配SM103的CUDA13构建；真实推理兼容性待canary核验                                       |
| 保存与评测                   | 无中间checkpoint；最终native/HF导出后评测全部1,013道英文SimpleVQA；成功后仅清理本次权重 |
| SwanLab 训练 / 评测链接      |                                                                                         |

| 数据集       | 题数 | 正确数 | Pass@1（%） | 去重 Pass@1（%） | 失败数 |
| ------------ | ---: | -----: | ----------: | ---------------: | -----: |
| SimpleVQA EN |      |        |             |                  |        |

本次none/full各8步，剔除前3步后吞吐中位数分别为110,120.3 / 86,015.9 tokens/s，峰值allocated显存182.86 / 73.25 GiB；选择none，短测吞吐高约28.0%。尚未完成真实推理canary与训练看板验收。

与下方基座使用同一Direct Answer协议：温度0、32K、无工具、原生Instruct模板及同一裁判。先前运行的内核不匹配错误已保留在历史记录；本次未完成前不填正式分数或宣称2epoch完成。

## 2026-11-10：基座 SimpleVQA

| 字段             | 内容                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------- |
| Run ID           | `qwen3-vl-8b-base-20261110-112630`                                                      |
| 状态             | 已保留130题；修复本机图像编码超时后断点恢复，尚无正式分数                               |
| 实验启动时间 UTC | 2026-11-10 11:26:30                                                                     |
| 模型             | Qwen/Qwen3-VL-8B-Instruct，原始权重；revision待确认                                     |
| 代码分支         | `github/feat/wuhuan/opensearch_v1`                                                      |
| 评测 commit      | `a045c20e25f34302bc61675e978ee7c19ce2dda2`                                              |
| 恢复 commit      | `3952695bafa39b329752574e06d8c8a07f2d71a6`；原始来源与恢复来源分别保留                  |
| 硬件与并发       | 6 × H800，TP1 / DP6，32 workers；仅使用空闲且通过CUDA计算检查的卡                       |
| 评测协议         | SimpleVQA全部1,013道英文题，Direct Answer，无工具，温度0，32K总上下文，原生Instruct模板 |
| 裁判             | 与SFT评测共用裁判和rubric；客户端不限速                                                 |
| 去重             | 使用同一34,607条SFT训练数据的问题与图片hash索引                                         |
| SwanLab 评测链接 |                                                                                         |
| 评测结束 UTC     |                                                                                         |

| 数据集       | 题数 | 正确数 | Pass@1（%） | 去重 Pass@1（%） | 失败数 |
| ------------ | ---: | -----: | ----------: | ---------------: | -----: |
| SimpleVQA EN |      |        |             |                  |        |

生成任何答案前先后遇到本机PyTorch 2.9.1/cuDNN兼容检查、SGLang与Transformers 5的nullable `pad_token_id`兼容问题，均保留失败日志。重启沿用本机已验证的禁用cuDNN方案，并仅在独立推理进程补齐缺失的配置默认值，不改权重或系统依赖；完整评测尚待完成。推理控制进程随后异常退出，保留84题结果，恢复时按ID继续剩余929题，不重新生成已完成答案。运行时为SGLang 0.5.9+sy2、Torch 2.9.1/CUDA12.8、Transformers 5.12.1；与训练后推理的运行时差异需另记。恢复后定位到图像patch投影在非cuDNN的Conv3d路径超过300秒，触发scheduler看门狗；独立推理进程将单patch卷积改为等价的flatten加linear，CPU float64最大误差约1e-15，GPU BF16大批量计算通过。已完成130题保留，按ID继续剩余883题；数值实现变化与恢复记录单独留存。运行记录、模型校验清单与逐题输出均保存在仓库外。

## 2026-11-10：SFT 2 epochs + SimpleVQA

### 实验信息

| 字段                                   | 内容                                                                          |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| Run ID                                 | `qwen3-vl-8b-20261110-103025`                                                 |
| 状态                                   | 调优与canary训练/导出成功；canary推理服务启动失败，正式训练尚未开始           |
| 实验启动时间 UTC                       | 2026-11-10 10:30:25                                                           |
| 训练开始 / 结束 UTC                    |                                                                               |
| 评测开始 / 结束 UTC                    |                                                                               |
| 代码分支                               | `github/feat/wuhuan/opensearch_v1`                                            |
| 训练 commit（完整 hash）               | `a045c20e25f34302bc61675e978ee7c19ce2dda2`                                    |
| 评测 commit（完整 hash，按数据集区分） |                                                                               |
| 初始模型与 revision                    | Qwen/Qwen3-VL-8B-Instruct；revision待确认                                     |
| 训练数据与 revision                    | `OpenSearch-VL/Search-VL-SFT-36K`，`2c1c460af4fa15bd63210cbf426a96664b959944` |
| train / val / test 条数                | 34,607 / 32 / 32，远端重新准备并核对，不按长度过滤                            |
| 计划 / 实际 steps、epochs              | 计划2 epochs；实际完成量待运行后记录                                          |
| GPU 型号与数量、TP / PP                | 8 × SM103（每卡267.7 GiB可见显存）；TP1 / PP1 / CP1 / DP8                     |
| Global batch / 学习率 / seed           | 32 / 2e-5 / 42                                                                |
| 重算 / optimizer offload               | none（实测选定） / 0                                                          |
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

首次任务训练调优和canary训练/导出成功；推理启动因运行时只含SM90内核、缺少SM103对应的`common_ops`而失败，尚未开始正式训练。已改用匹配硬件的运行镜像重试。基座未完成时不计算提升；补判复用原模型答案，保留原始错误记录。

### 复现与产物

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/prepared-data
export SAVE_DIR=/path/to/artifacts
export NUM_EPOCHS=2
export GLOBAL_BATCH_SIZE=32
export TP_SIZE=1 PP_SIZE=1 CP_SIZE=1 RECOMPUTE=none
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

保留Flash Attention、动态batch、负载均衡、按rank取数与预取、分块logits loss。本次固定TP1/PP1/CP1/DP8，不开启sequence parallel或CPU optimizer offload；避免TP/CP激活通信和PP流水线开销，仅保留必要的DP梯度及distributed optimizer参数同步。

比较模型层重算none与full；none仍保留loss重算和chunked logits。各候选8步、不保存权重，剔除前3步后按稳定吞吐与峰值显存选正式配置。32K显存占用、硬件kernel兼容性和性能均须通过真实多模态前后向验证，当前不填写性能数字。训练后由最终HF起TP1/DP8推理，只评SimpleVQA。

本次8步候选剔除前3步后，TP1/PP1/CP1的吞吐中位数为77,394.6 tokens/s（none）与57,953.7 tokens/s（full）；各组5个测量点，峰值allocated显存182.85/73.25 GiB。选择none，候选吞吐约高33.5%；短测不替代正式长跑结果。硬件峰值映射未校准，MFU暂不作为有效比较指标。
