# 架构评审参考案例

## 推理恢复：让编排回到拥有者

对照 [原实现 #548](https://github.com/redai-studio/Relax/pull/548)（`0ab3d29a`）与 [职责重构 #705](https://github.com/redai-studio/Relax/pull/705)（`2ee4ffc`），范围为 Megatron 路径。

原实现让训练 rank 同时读取恢复状态、执行服务握手和 RL HTTP 评估；独立 Executor 借用 Controller 的服务集合与状态完成恢复。训练计算、跨服务编排和资源管理交织。

```text
原来：组件 -> 训练 rank [计算 + 恢复准入 + 服务握手 + 同步/评估]
现在：组件 [阶段编排 + RecoveryGate] -> 后端计算 -> 同步协调 -> 指标收尾
      Controller [恢复流程] -> Service / Rollout / updater / DCS [各自资源]
```

- 将服务握手、接收方选择和 RL HTTP 评估交给现有协调模块；组件统一调用训练、同步和评估阶段。恢复开关只改变准入与失败处理，删除内部兼容双路径。
- 将没有独立资源或生命周期的 Executor 收回 Controller；Rollout 封装生成窗口清理，updater/DCS 管理传输资源和完整接收确认。后端保留模型、CPU 快照、通信操作与本地离线评估。
- 这样划分使控制决策归编排者、资源操作归拥有者，减少跨层状态依赖和重复路径。保持用户可见的训练与恢复语义，旧私有调用形态不自动构成兼容要求。

参考其职责判断方式；具体目录与抽象仍需依据实际调用链和资源归属，不照搬案例布局。
