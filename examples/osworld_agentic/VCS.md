# Visual Context Segmentation

[English](VCS.en.md)

VCS 在生成请求时保留完整文字历史、批量裁剪旧截图，将连续多次回答导出成一个训练片段。默认关闭，baseline 仍使用上游的两张图像窗口和逐调用导出。不改任务执行、evaluator、reward shaping 或 GRPO 的逻辑轨迹分组。

## 开启

编辑本 recipe 的 `app/osworld_config.yaml`，通过原有训练入口使用配置：

```yaml
context_mode: vcs
context_policy: baseline
vcs_keep_images: 2
vcs_prune_images: 2
```

`context_mode: baseline` 恢复原路径。`vcs_keep_images: 4`、`vcs_prune_images: 4` 是可选对照配置，不是新的正式默认值。两个数均须大于零。VCS 不能与上游的截图跳过策略组合；`max_image_history_length` 在 VCS 下只影响上游临时 prompt 的构造，实际请求的图像窗口由 VCS 控制。

沿用现有脚本中的显式 export、自定义 advantage、dynamic batching 和 token 预算设置。多片段不能改用普通逐样本 reward 归一化。任务指令与工具定义较长时，保留文字仍可能触及上下文上限；VCS 只限制图片，不保证任意长轨迹都能装入上下文。

也可通过训练脚本的 worker/agent 环境统一开启（显式环境变量优先于 YAML）：

```bash
export RELAX_VCS=1
export RELAX_VCS_KEEP_IMAGES=2
export RELAX_VCS_PRUNE_IMAGES=2
```

`RELAX_VCS=0` 强制关闭；未设置时保留原 YAML 行为。Ray 启动脚本须把这些变量放入 runtime env 并传给 agent 子进程，单独在提交端 export 不保证 worker 可见。

公共实现位于 `relax.utils.visual_context.VisualContext`，OSWorld 的原导入路径保留为兼容适配器。其他 CUA agent 使用默认的 `full_history` 模式：每个顺序会话创建独立实例，在实际请求前调用 `prepare(payload)`，回复后用精确的请求与真实 assistant 消息组成 record 调用 `commit(record, usage)`，会话结束调用 `finish(reason)`，显式导出 `records` 而不是每次调用。每个环境轮开始调用 `begin_turn(turn)`；同轮解析重试不重置。全量历史可包含 user/tool 截图、工具调用、工具结果和初始任务文本。上游截图折叠须关闭，由 VCS 统一裁剪。`latest_observation` 模式只用于 OSWorld 当前重建 prompt 的约定。

## 精确规则与例子

依据 [SCALECUA §3.4、Figure 4 / Algorithm 1 与附录 C.1 / Algorithm 2](https://arxiv.org/pdf/2607.11185)，采用算法中的严格条件 `下一次请求图片数 > K + Δ`。正文另有“达到阈值”的描述，本实现以伪代码为准。计数包含初始截图、当前截图；不是唯一图片数，相同截图再次观察也计一次。超过阈值后移除最旧的 Δ 张，并非立即降到 K 张。每次只加入一张新截图，因此阈值刚超过时裁剪后为 K+1 张。

以 K=2、Δ=2、连续 8 轮无重试、工具不变为例。`I1` 是第一次推理前的初始截图；`Ot` 是该轮上游提供的文字观察；`At` 是真实回答。每段还带有固定 system 和一个独立的片段标记。

| 推理轮 | 请求中的截图 | 请求中的文字历史 | 推理后的训练归属             |
| ------ | ------------ | ---------------- | ---------------------------- |
| 1      | I1           | O1               | A1 属于段 0                  |
| 2      | I1–I2        | O1,A1,O2         | A2 属于段 0                  |
| 3      | I1–I3        | O1,A1,O2,A2,O3   | A3 属于段 0                  |
| 4      | I1–I4        | O1,A1,…,O4       | A4 属于段 0                  |
| 5      | I3–I5        | O1,A1,…,O5       | 先保存段 0，再生成段 1 的 A5 |
| 6      | I3–I6        | O1,A1,…,O6       | A6 属于段 1                  |
| 7      | I5–I7        | O1,A1,…,O7       | 先保存段 1，再生成段 2 的 A7 |
| 8      | I5–I8        | O1,A1,…,O8       | A8 属于段 2；结束时保存尾段  |

最终三行的监督目标分别为 A1–A4、A5–A6、A7–A8。第二行的 A1–A4、第三行的 A1–A6 仅作为输入，不参与损失。裁剪前保存的是已提交的回答末态，不包含尚未生成回答的新观察。片段内部严格追加；不会重写先前的 Instruction、Previous actions 或工具结果。上游当前观察中的这些文字作为新消息追加，可能包含重复的指令/摘要；没有增加摘要模型。

## 数据与损失路径

`RelaxHybridAgent.call_llm` 在 HTTP 请求前调用 `VisualContext.prepare`；真实回复继续写入 `turn_records`，VCS 另行保存片段末态。`agent.py` 导出片段的原始 messages、tools、chat_template_kwargs。`AgenticSessionShard` 按这些字段查找已提交的 Forest 状态，`SessionForest.build_sample` 直接连接该分支的真实生成 token 与 rollout log probability：生成节点 mask=1，后续观察 mask=0。

新段会在 system 后、所有历史前放置新的 `[Visual context segment N]` 用户消息。该标记防止最长前缀匹配意外复用任意旧分支，包括工具切换后切回、相同文本的解析重试。新根下重放的历史整体属于 prompt，即 mask=0；不会重新 tokenize 旧回答后再配上旧概率。每段仅导出最后一个 committed state，不逐 turn 导出同一条线性链。适配层没有处理或合成任何 token/probability 数组。

所有片段仍使用同一 Session / `Sample.index`，共享轨迹级 advantage。框架 sample-mean loss 已按逻辑 index 汇总各片段的有效 token 分母，无需按片段数再除。正式脚本的 `OSWORLD_PER_TRAJ_NORM=0` 不变；若用户使用旧的 `=1`，VCS 按原始模型调用数缩放，保持 baseline 的尺度，而不按片段数缩放。历史 `rollout/raw_reward` 也保留原始调用数加权口径。`agent_turns` 保持环境轮数，成功率仍由每 Session 的指标上报，reward 只写入第一条 export。

## 边界、诊断与限制

- 基于 README 固定的 [OSWorld fork 提交](https://github.com/fxsc03/OSWorld-main-nex/tree/50b4a85778cfc13ce00e3bf63a3cd08bd260fb96)。其新增观察/执行反馈位于最后一条 user 消息，工具定义主要在 system 文本中。每轮读取最新 system、tools 和模板参数；变化时先保存旧段，再用新定义开启新段，保留文字历史。
- 上游解析失败会用相同 prompt 重采样，最多五次。VCS 保持该策略：失败回复仍参与训练，但作为先前片段的末次回答；重试输入不加入该失败回复。每次已返回非空 content 的调用均保留，含 `finish_reason=length`；空回复与没有拿到结果的传输尝试不凭空生成训练目标。若外部版本开始改写解析重试的消息，则明确报错，不能静默丢弃纠错提示。
- 任务完成、正常步数耗尽，以及驱动已处理的 `predict_error` / context length 提前结束都会保存最后非空片段。取消、丢弃和无法完成结果写出的异常仍遵循原有 Session 生命周期，不保证导出。HTTP 响应丢失后的重试仍继承原有服务语义，不提供新的 exactly-once 传输协议。
- `metadata.vcs_segment` 包含分段原因、调用索引、环境轮号、解析尝试号、新回答数、图像数、裁剪数、触发时图像数、输入/回答 token 数。尾段原因采用任务终止原因。原有 `_session_metrics` 增加 `osworld/vcs_segments`、`osworld/vcs_pruned_images`、`osworld/input_tokens`、`osworld/answer_tokens`，经现有指标接口上报。调用数、解析重试数、环境步数仍走原指标。token 统计来自模型 API usage，精确训练 mask/token/probability 以 Forest 生成的 Sample/dump 为准。
- VCS 算法共用于 Relax 公共模块和 CUA 适配层，不增加依赖，也不改变 SessionForest 的 token 归属逻辑。归档仍记录每次实际模型请求，而不是用片段代替调用。外部 `HybridAgentLocal.trajectory` 的调试字段仍是其临时构造的 baseline prompt，实际 VCS 请求请查 Relax dump 或归档。

CPU 回归命令（mock 模型和环境，无 Nex/GPU）：

```bash
python3 -m pytest -q tests/examples/test_osworld_visual_context.py \
  tests/examples/test_osworld_agentic_advantage.py \
  tests/examples/test_osworld_agentic_monitoring.py \
  tests/examples/test_osworld_archive.py tests/test_agentic_rollout.py
```

测试使用真实 Forest 匹配、提交、显式导出与训练数据转换，编码器和生成器为 mock；确定性对照结果写入 pytest 临时目录的 `comparison-*.json`。这些数字不代表 Qwen 的实际 token 开销或速度。后续应固定初始模型、任务池、rollout 数、硬件与 token/时间预算，对比 baseline 和 VCS 的独立评测集效果、有效轨迹比例、rollout 耗时、训练更新耗时及完整周期耗时。真实多模态编码、GPU 训练稳定性和吞吐尚需独立验证，不承诺论文的 2.83× 加速。
