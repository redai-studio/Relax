# SenseNova ART 图像编辑 RL

ART 复用已合入的 SenseNova SFT 模型实现，通过 SGLang 采样 ODE 轨迹，
再由 Relax FSDP2 回放训练。当前支持单机 8 卡同步 colocate，每卡一个推理
副本；训练生成分支指定模块的全部参数，理解分支保持冻结，CFG=1、关闭思考。
SFT 和 GRPO 保持原有入口。

环境、数据格式和可复制启动命令见[英文指南](../../en/guide/diffusion-art.md)。
#495 已提供 SGLang v0.5.21 升级和通用补丁，本 PR 不重复升级。
使用本分支 `docker/Dockerfile.cu13` 构建镜像，统一的
`docker/patch/sglang/v0.5.21.patch` 已包含 ART 支持，构建时一次应用到固定的干净源码。
已有服务需重建镜像并重启以载入 ART 接口。
ART 补丁保留默认 RoPE 缓存与原生注意力路径，不包含 SenseNova 专属 LoRA、
TP/SP/分布式 CFG 或可选 FA3 适配。通用补丁中的已有能力保持不变。
TextPecker 示例需要独立 OpenAI 兼容 OCR 服务、CPU LPIPS 及离线 AlexNet 权重。
模型、图片、奖励地址和输出路径均由使用者显式提供。
推理部署默认读取启动脚本旁的 `scripts/training/diffusion/configs/sglang-sensenova-art-8xgpu.yaml`，
可通过 `SGLANG_CONFIG_PATH` 指定其他配置。
启动脚本直接提交 Ray Job，不再包含独立 Python recipe 或 `--mode` 预检入口。
提交前检查图片、离线 LPIPS 权重、奖励服务和磁盘余量；算法、轨迹及恢复校验仍在运行时执行。
使用独立 Ray 集群，训练退出会关闭其 Serve 应用；`RAY_NO_WAIT=1` 为非阻塞提交。
看板参数可直接追加 `--use-clearml --use-metrics-service --tb-project-name ... --tb-experiment-name ...`。

每轮使用完整条件组的原始标量奖励计算终态协方差，将每组候选均分成两个不重叠
窗口，执行两次 AdamW 更新后只发布一次 live 权重，不使用 EMA、重新加噪或旧策略前向。
回放保留原始 FP32 时间坐标、速度和负 Euler 步长，FP16/BF16 状态保持原比特。
奖励失败整轮跳过；更新中途失败必须从最近完整轮次恢复。
保存时由 actor 先落盘同轮数据游标，DCP 提交前再次校验游标及完整训练状态和各 rank RNG；
通过后才发布 `COMMITTED`、更新 latest 并轮转，失败时保留上一完整恢复点。
ART 启动时要求 eager 数据源、`max_staleness=0`；已提交轮次的游标不可覆盖，
相同保存重试复用原游标，冲突保存直接拒绝。
恢复训练前会校验同轮 eager 数据游标；缺失、损坏或候选计数不一致时终止恢复，
通用 `train.py --load` 入口也执行此校验。新数据游标绑定有序 manifest 内容、切片、解析及
shuffle/seed 配置；指纹仅在数据源启动时计算并复用于保存，旧游标会提示身份未经验证。
引用的媒体及 tokenizer 文件须保持不变。
评估使用只返回图像的终态 ODE 请求；SenseNova 和 Qwen-Image 均以固定评估数据集和候选身份
派生初始噪声，真实轮次用于结果归档。训练噪声仍随轮次变化。

无损 mask 压缩、轨迹传输和减少重复复制由补丁自动启用。保存间隔建议从
`SAVE_INTERVAL=50` 起，结合恢复成本与磁盘预算调整。

提交包含 CPU 数学、更新、轨迹和 DCP/RNG 回归；迁移后的 GPU 完整链路、
业务质量、显存及性能仍需在部署使用的确切镜像、模型和奖励服务上验证。
