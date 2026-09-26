# Dataset Design

This document describes the technical design and implementation of the data loading infrastructure in Relax, covering both the `Dataset` class (eager loading) and `StreamingDataset` class (lazy loading).

## Overview

The Relax framework provides two dataset implementations to handle different scale and performance requirements:

| Class              | Loading Strategy                | Best For                                        |
| ------------------ | ------------------------------- | ----------------------------------------------- |
| `Dataset`          | Eager (all data loaded at init) | Small to medium datasets, fast random access    |
| `StreamingDataset` | Lazy (on-demand loading)        | Large datasets, memory-constrained environments |

Both classes inherit from `BaseDataset` and share a common interface, making them interchangeable through configuration.

## Architecture

### Class Hierarchy

```
BaseDataset (ABC)
├── Dataset            # Eager loading implementation
└── StreamingDataset   # Lazy loading implementation
```

### Data Flow

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

## Dataset (Eager Loading)

The `Dataset` class loads all data into memory during initialization. This provides:

- **O(1) random access** after initialization
- **Consistent memory footprint** throughout training
- **Simple implementation** with straightforward shuffling

### Use Cases

- Small to medium datasets (\< 1M samples)
- Fast random access requirements
- Pre-filtered data where length checking is acceptable at init

## StreamingDataset (Lazy Loading)

The `StreamingDataset` class loads data on-demand, providing:

- **Constant memory usage** regardless of dataset size
- **Fast initialization** (deferred loading)
- **LRU caching** for frequently accessed samples

### Use Cases

- Large datasets (> 1M samples or multimodal data)
- Memory-constrained environments
- Scenarios where fast startup is important

## Usage Examples

### Using Dataset (Eager Loading)

This is the default mode.

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

# Shuffle for new epoch
dataset.shuffle(epoch_id=1)

# Access samples
sample = dataset[0]
batch = dataset.samples[0:32]
```

### Using StreamingDataset (Lazy Loading)

To enable this mode, add the following arguments to your script:

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

# Get batch (preferred API for streaming)
samples, crossed_epoch = dataset.get_batch(32)
```

## Recommendations

| Scenario                  | Recommended Class | Reason                      |
| ------------------------- | ----------------- | --------------------------- |
| \< 100K samples           | Dataset           | Simpler, faster access      |
| 100K - 1M samples         | Either            | Depends on available memory |
| > 1M samples              | StreamingDataset  | Memory efficiency           |
| Multimodal (images/video) | StreamingDataset  | Large sample sizes          |

## Next Steps

- [Configuration Guide](./configuration.md) - Configure dataset settings
- [Architecture](./architecture.md) - Understand the system design
