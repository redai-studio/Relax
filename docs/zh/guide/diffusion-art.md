# SenseNova ART 图像编辑 RL

ART 复用已合入的 SenseNova SFT 模型实现，通过 SGLang 采样 ODE 轨迹，
再由 Relax FSDP2 回放训练。当前支持单机 8 卡同步 colocate，每卡一个推理
副本；训练生成分支指定模块的全部参数，理解分支保持冻结，CFG=1、关闭思考。
SFT 和 GRPO 保持原有入口。

环境、数据格式和可复制启动命令见[英文指南](../../en/guide/diffusion-art.md)。
#495 已提供 SGLang v0.5.21 升级和通用补丁，本 PR 不重复升级。
使用本分支 `docker/Dockerfile.cu13` 构建镜像，它会在通用补丁之后应用独立的
`v0.5.21-sensenova-art.patch`；已有服务需重建镜像并重启以载入 ART 接口。
ART 补丁保留默认 RoPE 缓存与原生注意力路径，不包含 SenseNova 专属 LoRA、
TP/SP/分布式 CFG 或可选 FA3 适配。通用补丁中的已有能力保持不变。
TextPecker 示例需要独立 OpenAI 兼容 OCR 服务、CPU LPIPS 及离线 AlexNet 权重。
模型、图片、奖励地址和输出路径均由使用者显式提供。

每轮使用完整条件组的原始标量奖励计算终态协方差，将每组候选均分成两个不重叠
窗口，执行两次 AdamW 更新后只发布一次 live 权重，不使用 EMA、重新加噪或旧策略前向。
回放保留原始 FP32 时间坐标、速度和负 Euler 步长，FP16/BF16 状态保持原比特。
奖励失败整轮跳过；更新中途失败必须从最近完整轮次恢复。

`ART_GROUP_OVERLAP=1` 开启有界奖励与采样重叠；额外的
`ART_GENERATION_PREFETCH=1` 默认关闭，要求至少两个待处理组、每组 8 候选。
无损 mask 压缩、轨迹传输和减少重复复制由补丁自动启用。保存间隔建议从
`SAVE_INTERVAL=50` 起，结合恢复成本与磁盘预算调整。

提交包含 CPU 数学、更新、轨迹和 DCP/RNG 回归；迁移后的 GPU 完整链路、
业务质量、显存及性能仍需在部署使用的确切镜像、模型和奖励服务上验证。
