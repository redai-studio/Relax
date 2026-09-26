# DeepEyes RL (Multi-turn VLM Tool Use)

本示例将 DeepEyes 奖励函数和多轮工具环境（zoom/rotate）集成到 `Relax` 框架中，用于训练视觉语言模型（VLM）的工具使用能力。

## 概述

DeepEyes 是一个多轮交互式视觉问答环境，模型可以通过调用工具（如缩放、旋转图像）来更好地理解和回答问题。

## 核心组件

### 1. 多轮 Rollout

- **模块**: `examples.deepeyes.rollout.generate`
- **功能**: 实现自定义的多轮交互式采样逻辑

### 2. 交互环境

- **文件**: `env_deepeyes.py`
- **功能**:
  - 解析模型输出的 `<tool_call>{...}</tool_call>` 格式
  - 返回 `<tool_response>...</tool_response>` 和更新后的图像
  - 支持工具：缩放（zoom）、旋转（rotate）等

### 3. 奖励函数

- **文件**: `reward_deepeyes.py`
- **功能**:
  - 基于 judge 模型的答案质量评分
  - 工具调用格式正确性检查
  - 综合奖励计算

## 快速开始

### 运行训练

```bash
cd /path_to/Relax/scripts
bash benchmark.sh run_deepeyes
```

或安装benchmark.sh 中的依赖后，直接运行：

```bash
bash examples/deepeyes/run_deepeyes.sh
```

在单机 8×H800（80GB）上运行 `run_deepeyes.sh` 时的 GPU 使用率监控（平均约 66%）：

![单机 8×H800（80GB）运行 run_deepeyes.sh 的 GPU 使用率](../../docs/public/deepeyes-h800.png)

## 文件结构

```
examples/deepeyes/
├── README.md              # 本文档
├── run_deepeyes.sh        # 训练启动脚本
├── base_env.py            # 环境实现基类
├── env_deepeyes.py        # 环境实现
├── reward_deepeyes.py     # 奖励函数实现
├── processor_patch_utils.py  # 预展开 token 处理逻辑
├── sglang_patch/             # SGLang 外部 processor 注册包
└── rollout.py             # 多轮采样逻辑

```

## SGLang processor 兼容性

`run_deepeyes_r3.sh` 通过 SGLang 外部 processor 注册机制加载
`DeepEyesQwenVLImageProcessor`。该子类只覆盖 `process_mm_data_async` 并委托上游
实现；差异仅处理 DeepEyes 传入的预展开
`input_ids`：调用上游 processor 前将连续的 `<|image_pad|>` 折叠为一个占位符，
处理完成后再恢复原始 token 序列；多轮工具调用产生多个图像 item 时，会先合并
各 item 的 grid 再重新计算 mRoPE。其余输出仍由上游实现产生。
脚本不会修改 SGLang 安装目录，重复 import/注册也会覆盖到同一个 processor 映射。

当前补丁对应官方镜像中的 SGLang `0.5.12.post1`。升级 SGLang 时，先对比上游
`QwenVLImageProcessor.process_mm_data_async` 的签名和返回类型，再更新
`SUPPORTED_SGLANG_VERSION`，运行：

```bash
pytest -v tests/examples/test_deepeyes_processor_patch.py
bash examples/deepeyes/run_deepeyes_r3.sh
```

版本或方法签名不兼容时，启动会直接报错并提示需要复核补丁，而不会静默回退。
