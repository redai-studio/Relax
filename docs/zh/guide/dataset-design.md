# 数据集设计

本文档描述了 Relax 中数据加载基础设施的技术设计和实现，涵盖 `Dataset` 类（急切加载）和 `StreamingDataset` 类（惰性加载）。

## 概述

Relax 框架提供两种数据集实现来处理不同规模和性能需求：

| 类                 | 加载策略                     | 最适合                     |
| ------------------ | ---------------------------- | -------------------------- |
| `Dataset`          | 急切加载（初始化时加载全部） | 中小型数据集，快速随机访问 |
| `StreamingDataset` | 惰性加载（按需加载）         | 大型数据集，内存受限环境   |

两个类都继承自 `BaseDataset` 并共享通用接口，可以通过配置互换使用。

## 架构

### 类层次结构

```
BaseDataset (ABC)
├── Dataset            # 急切加载实现
└── StreamingDataset   # 惰性加载实现
```

### 数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Data Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌────────────────────┐    ┌─────────────────────┐      │
│  │  JSONL/      │    │  Dataset /         │    │  RolloutDataSource  │      │
│  │  Parquet     │───>│  StreamingDataset  │───>│  WithBuffer         │      │
│  │  Files       │    │                    │    │                     │      │
│  └──────────────┘    └────────────────────┘    └──────────┬──────────┘      │
│                                                           │                 │
│                                                           ▼                 │
│                      ┌─────────────────────────────────────────────────┐    │
│                      │           RolloutWorker.generate()              │    │
│                      │   - get_samples(num_samples)                    │    │
│                      │   - Iterates until rollout_batch_size reached   │    │
│                      └─────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Dataset（急切加载）

`Dataset` 类在初始化期间将所有数据加载到内存中。这提供了：

- **O(1) 随机访问**（初始化后）
- **一致的内存占用**（整个训练过程）
- **简单实现**（直接的洗牌机制）

### 使用场景

- 中小型数据集（\< 100 万样本）
- 快速随机访问需求
- 预过滤数据，初始化时长度检查可接受

## StreamingDataset（惰性加载）

`StreamingDataset` 类按需加载数据，提供：

- **恒定内存使用**（无论数据集大小）
- **快速初始化**（延迟加载）
- **LRU 缓存**（频繁访问的样本）

### 使用场景

- 大型数据集（> 100 万样本或多模态数据）
- 内存受限环境
- 需要快速启动的场景

## 使用示例

### 使用 Dataset（急切加载）

默认使用当前模式

```python
from relax.utils.data.data import Dataset

dataset = Dataset(
    path="data/train.jsonl",
    tokenizer=tokenizer,
    processor=processor,
    max_length=2048,
    prompt_key="text",
    apply_chat_template=True,
)

# 为新 epoch 洗牌
dataset.shuffle(epoch_id=1)

# 访问样本
sample = dataset[0]
batch = dataset.samples[0:32]
```

### 使用 StreamingDataset（惰性加载）

启动该模式，需要脚本增加如下参数：

```bash
    --use-streaming-dataset \
    --streaming-buffer-size 10000 \
```

```python
from relax.utils.data.streaming_dataset import StreamingDataset

dataset = StreamingDataset(
    path="data/train.jsonl",
    tokenizer=tokenizer,
    processor=processor,
    max_length=2048,
    buffer_size=10000,
    prompt_key="text",
    apply_chat_template=True,
)

# 获取批次（流式推荐 API）
samples, crossed_epoch = dataset.get_batch(32)
```

## 推荐

| 场景                | 推荐类           | 原因             |
| ------------------- | ---------------- | ---------------- |
| \< 10 万样本        | Dataset          | 更简单，访问更快 |
| 10 万 - 100 万样本  | 两者皆可         | 取决于可用内存   |
| > 100 万样本        | StreamingDataset | 内存效率         |
| 多模态（图像/视频） | StreamingDataset | 样本大小大       |

## 下一步

- [配置指南](./configuration.md) - 配置数据集设置
- [架构设计](./architecture.md) - 理解系统设计
