# Configuration Reference

The parameters are divided into three categories:

1. Megatron parameters: Relax will read all parameters defined in Megatron from the PYTHONPATH. You can configure them by passing arguments such as --tensor-model-parallel-size 2.
2. SGLang parameters: All parameters supported by the installed SGLang environment are available. These parameters must be prefixed with --sglang. For example, --mem-fraction-static should be passed as --sglang-mem-fraction-static.
3. Relax-specific parameters: Please refer to `relax/utils/arguments.py`.

For common configuration usage and examples, see the [Quick Start Guide](./quick-start.md).

---

## Cluster and Resource Configuration

### Ray Launch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--rollout-num-gpus-per-engine` | int | 1 | GPUs per SGLang inference engine, equivalent to SGLang's `tp_size` |
| `--num-gpus-per-node` | int | 8 | GPUs per node. Set this if using fewer than 8 GPUs per node in colocate mode |
| `--resource` | json | - | Ray resource configuration in JSON format. Example: `'{"actor":[replicas, gpus], "rollout":[replicas, gpus]}'` |
| `--colocate` | flag | False | Whether to colocate inference engines and training Actors on the same GPUs. Automatically sets `--offload` to True |
| `--offload` | flag | False | Equivalent to setting both `--offload-train` and `--offload-rollout` |
| `--offload-train` | flag | None | Whether to offload training Actor to CPU during training. Always True when `--colocate` is enabled |
| `--offload-rollout` | flag | None | Whether to offload Rollout generator to CPU during training. Always True when `--colocate` is enabled |
| `--distributed-backend` | str | nccl | Distributed backend |
| `--distributed-timeout-minutes` | int | 10 | Distributed timeout in minutes |

### TransferQueue Data Queue

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-data-storage-units` | int | 1 | Number of TransferQueue SimpleStorageUnit actors |
| `--per-rank-fetch` | flag | False | Let every TP/PP rank pull its own copy from TransferQueue in parallel instead of paying one rank-0 pickle + one TP/PP broadcast. Cross-rank consistency relies on the TQ sampler's `(partition_id, task_name, dp_rank, batch_index)` cache, which is PP/TP-invariant. Auto-disabled when `rollout_routed_experts` is in `data_fields` (jagged NestedTensor broadcast path is incompatible). Recommended for multi-GPU training together with `--num-data-storage-units >= TP world size`. Wins when pickle dominates `tgd_bcast_tp_time`. |
| `--max-staleness` | int | 0 | Maximum staleness for TransferQueue data system (0=on-policy) |
| `--polling-mode` | bool | True | Whether to use polling mode when fetching metadata |
| `--num-iters-per-train-update` | int | 1 | Number of iterations per global batch in fully async pipeline |

---

## Training Backend and Mode

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--train-backend` | str | megatron | `megatron` | Training backend selection |
| `--qkv-format` | str | thd | `thd`, `bshd` | QKV layout for Megatron backend. Dynamic batching not supported in `bshd` mode; must specify `--micro-batch-size` |
| `--megatron-to-hf-mode` | str | raw | `raw`, `bridge` | Megatron to HF weight conversion method. `bridge` uses megatron bridge for automatic conversion |
| `--true-on-policy-mode` | flag | False | - | Whether to enable true on-policy mode |
| `--fully-async` | flag | False | - | Whether to use fully asynchronous training pipeline |

---

## Checkpoint Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--hf-checkpoint` | str | None | HuggingFace model checkpoint path. Used to initialize SGLang and provide tokenizer. Need not contain latest parameters, only consistent with training model architecture |
| `--ref-load` | str | None | Reference model checkpoint path. Used as initial checkpoint for training when `--load` is not set |
| `--ref-ckpt-step` | int | None | Reference model checkpoint step |
| `--load` | str | None | Actor model checkpoint load path. Specify for resuming training |
| `--save` | str | None | Path to save model during training |
| `--save-interval` | int | None | Model save interval in steps |
| `--save-hf` | str | None | Path to save HuggingFace format model for Megatron backend. Path can include `{rollout_id}` placeholder |
| `--async-save` | flag | False | Asynchronous checkpoint saving |
| `--no-save-optim` | flag | False | Do not save optimizer state in checkpoint. Reduces checkpoint size but prevents resuming training from that checkpoint |
| `--rotate-ckpt` | flag | False | Whether to rotate checkpoints. Requires setting `--save`, `--save-interval`, and `--async-save` |
| `--max-actor-ckpt-to-keep` | int | None | Maximum number of Actor checkpoints to keep |
| `--checkpoint-engine-backend` | str | nccl | Checkpoint engine backend |
| `--critic-load` | str | None | Critic model checkpoint path. When None, equals `--load` |
| `--critic-save` | str | None | Critic model save path |

---

## Data Configuration

### Dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--prompt-data` | str | None | Training prompt dataset path |
| `--input-key` | str | input | Key for input field in dataset |
| `--label-key` | str | None | Key for label field in dataset |
| `--conversation-key-map` | json | None | JSON map that rewrites non-OpenAI conversation messages into the `{role, content}` shape SFT expects. A single flat map covers two distinct renames: **field-name rename** (applied to every message-dict key, e.g. `from`→`role`, `value`→`content`) and **role-value rename** (applied only to the value of the resulting `role` field, e.g. `human`→`user`, `gpt`→`assistant` — so message bodies that happen to contain the same words are left untouched). Only consulted when `--label-key` is unset (i.e., `--input-key` holds a full messages list). Sharegpt example: `'{"from":"role","value":"content","human":"user","gpt":"assistant"}'` |
| `--metadata-key` | str | metadata | Key for metadata field in dataset |
| `--tool-key` | str | tools | Key for tools field when applying Chat Template |
| `--apply-chat-template` | flag | False | Apply Chat Template to input as OpenAI message format |
| `--apply-chat-template-kwargs` | json | {} | Additional parameters for Chat Template |
| `--system-prompt` | str | None | Optional system prompt added before user input. Final message is `<system_prompt> + <dataset_prompt>` |
| `--rollout-shuffle` | flag | False | Whether to shuffle prompt order during Rollout |
| `--rollout-seed` | int | 42 | Random seed for Rollout, used for shuffling prompts and random sampling |
| `--use-streaming-dataset` | flag | False | Use streaming dataset to save memory |
| `--streaming-buffer-size` | int | 10000 | Buffer size for streaming dataset |
| `--prefetch-chunk-size` | int | 32 | Number of samples to dispatch to the thread-pool in each prefetch round. Larger values increase throughput but also memory pressure. Only effective when `--use-streaming-dataset` is set and the dataset contains multimodal data |
| `--prefetch-max-cached` | int | 256 | Maximum number of pre-loaded samples kept in the prefetch cache. When the cache is full the background prefetch thread pauses until consumers free space. Set to 0 to disable prefetching. Only effective when `--use-streaming-dataset` is set and the dataset contains multimodal data |
| `--prefetch-num-workers` | int | 1 | Number of parallel worker threads inside the prefetch buffer for I/O-bound media decoding (video/image). Set to 1 to serialise all decoding (safest for FFmpeg which is not fully thread-safe). Higher values increase parallelism but may trigger EAGAIN errors on some platforms. Only effective when prefetching is enabled |
| `--custom-prompt-path` | str | None | Dotted import path to a custom function that transforms the prompt before conversation/multimodal processing. Function signature: `def custom_fn(prompt, data: dict) -> prompt`. Example: `my_package.prompt_utils.add_prefix` |
| `--data-source-path` | str | `relax.engine.rollout.data_source.RolloutDataSourceWithBuffer` | Rollout data source class path |
| `--start-rollout-id` | int | None | Starting Rollout step. If not set, attempts to read from checkpoint specified by `--load` |

### Multimodal Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--multimodal-keys` | json | None | Multimodal data field mapping. Example: `'{"image": "image_key"}'` |
| `--use-audio-in-video` | flag | False | Whether to process audio in video |
| `--image-max-token-num` | int | None | Maximum token count for image processing. Default is 16384 if not set |
| `--image-min-token-num` | int | None | Minimum token count for image processing. Default is 4 if not set |
| `--video-min-token-num` | int | None | Minimum token count for video frame processing. Default is 128 if not set |
| `--video-max-token-num` | int | None | Maximum token count for video frame processing. Default is 768 if not set |
| `--video-fps` | float | None | Target FPS for video processing. Default is 2.0 if not set |
| `--video-fps-min-frames` | int | None | Minimum frames for video processing. Default is 4 if not set |
| `--video-fps-max-frames` | int | None | Maximum frames for video processing. Default is 768 if not set |
| `--image-resize-scale-factor` | int | None | Scale factor for image resize dimension alignment. Default uses `patch_size * spatial_merge_size` (typically 28). Set to 0 to disable alignment |
| `--audio-sample-rate` | int | None | Sample rate for audio processing. Default is 16000 if not set |
| `--frame-factor` | int | None | Frame alignment factor. Default is 2 if not set |
| `--mm-processor-pool-size` | int | 0 | Size of the multimodal processor pool. 0 (default) disables the pool and uses ThreadPoolExecutor. When set to a positive integer, creates a ProcessPoolExecutor with the specified number of workers for true parallelism without GIL contention |

---

## Rollout Configuration

### Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-rollout` | int | None | Total number of Rollout rounds. Choose either this or `--num-epoch` |
| `--num-epoch` | int | None | Number of training epochs. Automatically calculates `num_rollout` based on dataset size. Ignored if `--num-rollout` is also set |
| `--rollout-batch-size` | int | **Required** | Number of prompts per rollout round. Total data = `rollout-batch-size * n-samples-per-prompt` |
| `--n-samples-per-prompt` | int | 1 | Number of responses generated per prompt |
| `--rollout-temperature` | float | 1.0 | Sampling temperature for inference engine |
| `--rollout-top-p` | float | 1.0 | Top-p sampling parameter for inference engine |
| `--rollout-top-k` | int | -1 | Top-k sampling parameter for inference engine. -1 means not used |
| `--rollout-max-response-len` | int | None | Maximum response length, equivalent to SGLang's `max_tokens` |
| `--rollout-max-prompt-len` | int | None | Maximum prompt length. Filters long prompts during dataset initialization if set |
| `--rollout-max-context-len` | int | None | Maximum context length for inference engine. Should not exceed `max_position_embeddings` in HuggingFace model `config.json` |
| `--rollout-stop` | str (list) | None | Stop words for Rollout. Can be one or multiple strings |
| `--rollout-stop-token-ids` | int (list) | None | Stop token IDs for Rollout |
| `--rollout-skip-special-tokens` | flag | False | Whether to skip special tokens in responses |

### Oversampling and Dynamic Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--over-sampling-batch-size` | int | None | Sampling batch granularity. When None, uses `rollout-batch-size`. Must be >= `rollout-batch-size` |
| `--dynamic-sampling-filter-path` | str | None | Dynamic sampling filter function path. Implements filters like DAPO (e.g., excluding all-correct or all-wrong samples). Example: `relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std` |
| `--buffer-filter-path` | str | None | Buffer filter function path. Function signature: `list[list[Sample]]` -> `list[list[Sample]]` |

### Partial Rollout

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--partial-rollout` | flag | False | Enable partial Rollout. Incomplete samples are recycled to data buffer, suitable for long response scenarios |
| `--partial-rollout-max-aborted-count` | int | None | Maximum number of times a sample can be aborted. After reaching threshold, sample is guaranteed to complete |
| `--mask-offpolicy-in-partial-rollout` | flag | False | Whether to mask previous generation in partial Rollout. When set, only on-policy generated tokens participate in training |

### Weight Update

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--update-weight-buffer-size` | int | 512MB | Buffer size for weight updates in bytes. Updates weights in chunks, useful for MoE models |
| `--update-weights-interval` | int | 1 | Weight update interval |
| `--keep-old-actor` | flag | False | Whether to keep Rollout model during training |

### External Inference Engine

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--rollout-external` | flag | False | Use external SGLang instance instead of framework-launched instance |
| `--rollout-external-engine-addrs` | str (list) | None | List of external engine addresses and ports |

### SGLang Engine Parameters

For more parameters, refer to SGLang official documentation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sglang-mem-fraction-static` | float | - | SGLang static memory allocation ratio |
| `--sglang-profile` | flag | False | Enable torch profiling on SGLang engines during rollout. Profile traces will be saved per rollout step |
| `--sglang-profile-steps` | int (list) | None | List of absolute rollout step IDs (0-indexed) at which to enable SGLang profiling. Takes precedence over `--sglang-profile-step-start/end`. Example: `--sglang-profile-steps 3 10 50` |
| `--sglang-profile-step-start` | int | None | Start of the rollout step range for SGLang profiling (**inclusive**, 0-indexed). Used with `--sglang-profile-step-end` to specify a contiguous range. Ignored if `--sglang-profile-steps` is set |
| `--sglang-profile-step-end` | int | None | End of the rollout step range for SGLang profiling (**inclusive**, 0-indexed). Used with `--sglang-profile-step-start` to specify a contiguous range. Ignored if `--sglang-profile-steps` is set. E.g. start=2, end=4 profiles steps 2, 3, 4 |
| `--sglang-profile-output-dir` | str | None | Output directory for SGLang profile traces. Defaults to `traces/<tb_experiment_name>/sglang_trace` |
| `--sglang-profile-num-steps` | int | 3 | Number of SGLang forward steps to profile per rollout. -1 profiles the entire rollout step until `stop_profile` is called |
| `--sglang-profile-activities` | str (list) | ["CPU", "GPU"] | Activities to profile (e.g., `CPU GPU`) |
| `--sglang-profile-by-stage` | flag | False | Profile by stage (prefill/decode) separately |
| `--sglang-profile-with-stack` | flag | False | Record call stack in profile traces |
| `--sglang-profile-record-shapes` | flag | False | Record tensor shapes in profile traces |

### Custom Rollout Functions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--rollout-function-path` | str | `relax.engine.rollout.sglang_rollout.generate_rollout` | Rollout generation function path |
| `--custom-generate-function-path` | str | None | Custom generate function to replace default rollout generate function. Suitable for multi-turn dialogue, function calling, etc. |
| `--rollout-data-postprocess-path` | str | None | Rollout data postprocessing function, called after all data (including log_probs) is fetched. Can be used to update loss mask |
| `--custom-rollout-log-function-path` | str | None | Custom Rollout logging function |
| `--custom-eval-rollout-log-function-path` | str | None | Custom evaluation Rollout logging function |

---

## Batch Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--global-batch-size` | int | None | Global batch size. Defines sample count needed for one parameter update (optimizer.step) |
| `--micro-batch-size` | int | 1 | Micro batch size. Ignored when `--use-dynamic-batch-size` is enabled |
| `--num-steps-per-rollout` | int | None | Training steps per Rollout. Equivalent to setting GBS = `rollout_batch_size * n_samples_per_prompt / num_steps_per_rollout` |
| `--use-dynamic-batch-size` | flag | False | Enable dynamic batching. Dynamically packs samples by length so each micro-batch's total tokens approach `--max-tokens-per-gpu` limit |
| `--max-tokens-per-gpu` | int | None | Maximum tokens per GPU. Must be set when dynamic batching is enabled. Should be set to approximately `max_response_len / cp_size` when using CP |
| `--log-probs-max-tokens-per-gpu` | int | None | Maximum tokens per GPU when computing log probs. When None, equals `max-tokens-per-gpu` |
| `--balance-data` | flag | False | Use `karmarkar_karp` to balance token count across data parallel ranks on the static/seqlen-balanced path. In fully async with `--use-dynamic-batch-size`, DP token balancing is automatic through `StreamingTokenBudgetSampler`; the flag is accepted but has no additional effect. Note: different responses for the same prompt may be assigned to different training steps |

---

## Parallelism Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--tensor-model-parallel-size` | int | 1 | Tensor parallelism size |
| `--pipeline-model-parallel-size` | int | 1 | Pipeline parallelism size |
| `--sequence-parallel` | flag | False | Enable sequence parallelism |
| `--context-parallel-size` | int | 1 | Context parallelism size |
| `--expert-model-parallel-size` | int | 1 | Expert parallelism size (for MoE models) |
| `--expert-tensor-parallel-size` | int | 1 | Expert tensor parallelism size |

### Recomputation

Recomputation parameters use native Megatron parameters. For details, refer to Megatron documentation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--recompute-granularity` | str | - | Recomputation granularity: `full`, `selective` |
| `--recompute-method` | str | - | Recomputation method: `uniform`, `block` |
| `--recompute-num-layers` | int | - | Number of layers to recompute |

---

## Optimizer Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lr` | float | 1e-6 | Learning rate |
| `--optimizer` | str | - | Optimizer type (native Megatron parameter) |
| `--lr-decay-style` | str | - | Learning rate decay style (native Megatron parameter) |
| `--weight-decay` | float | - | Weight decay (native Megatron parameter) |
| `--adam-beta1` | float | - | Adam beta1 (native Megatron parameter) |
| `--adam-beta2` | float | - | Adam beta2 (native Megatron parameter) |
| `--clip-grad` | float | 1.0 | Gradient clipping |
| `--seed` | int | 1234 | Random seed |
| `--optimizer-cpu-offload` | flag | - | Enable CPU offload for optimizer state (native Megatron parameter) |
| `--overlap-cpu-optimizer-d2h-h2d` | flag | - | Overlap CPU optimizer D2H/H2D communication (native Megatron parameter) |
| `--use-precision-aware-optimizer` | flag | - | Use precision-aware optimizer (native Megatron parameter) |
| `--use-distributed-optimizer` | flag | - | Shard optimizer state, ZeRO-1 style (native Megatron parameter) |
| `--overlap-grad-reduce` | flag | - | Overlap backward compute with grad reduce-scatter (native Megatron parameter) |
| `--overlap-param-gather` | flag | - | Overlap reduce-scatter with next-step param all-gather; requires `--overlap-grad-reduce` (native Megatron parameter) |
| `--calculate-per-token-loss` | flag | False | Calculate loss per token (native Megatron parameter) |

### FP16 Optimizer Compatibility Defaults

| Parameter | FP16 compatibility fallback | Native non-FP16 default | Description |
|-----------|-------------------------------|--------------------------|-------------|
| `--initial-loss-scale` | `32768` | `2**32` | Initial scale used by dynamic loss scaling |
| `--min-loss-scale` | `1` | `1` | Minimum scale used by dynamic loss scaling |
| `--use-precision-aware-optimizer` / `--no-use-precision-aware-optimizer` | enabled | disabled | Enable or disable TransformerEngine's precision-aware optimizer |
| `--store-param-remainders` / `--no-store-param-remainders` | disabled | enabled | Control parameter-remainder storage in the distributed optimizer |

With dynamic FP16 loss scaling (`--loss-scale` omitted), Relax preserves its historical values for omitted options and
emits one warning listing the applied fallbacks. With a static `--loss-scale`, `--initial-loss-scale` and
`--min-loss-scale` are inactive, so Relax does not fill, validate, or warn about them; the two boolean optimizer options
still use their FP16 compatibility fallbacks when omitted. Pass the active options explicitly to silence the warning.
In dynamic mode, both scale values must be finite and greater than zero, and `--min-loss-scale` must not exceed
`--initial-loss-scale`. Non-FP16 defaults come directly from Megatron's `OptimizerConfig`.

The Qwen3-4B FP16 recipe configures all four values explicitly. Extra arguments passed to the shell script are appended
to the training command, so a later value can override the recipe without editing it:

```bash
bash scripts/training/text/run-qwen3-4B-fp16-8xgpu.sh \
  --initial-loss-scale 65536 \
  --min-loss-scale 2 \
  --no-use-precision-aware-optimizer \
  --store-param-remainders
```

### Optimizer Flag Compatibility

| Scenario | `--use-distributed-optimizer` | `--overlap-grad-reduce` / `--overlap-param-gather` |
|---|---|---|
| Text-only dense | ✅ | ✅ |
| Dense VL, CP = 1 | ✅ | ✅ |
| Dense VL, CP > 1 | ✅ | ❌ |
| MoE | ✅ | ❌ |

---

## Algorithm Configuration

### Advantage Estimation

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--advantage-estimator` | str | grpo | generated from `ALGORITHM_SPECS` in `relax/algorithms/spec.py`; currently `grpo`, `gspo`, `sapo`, `cispo`, `rloo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline` | Advantage estimator. `--help` is authoritative: the choices are read from the registry, so a new algorithm appears there without this table being edited. OPD is independent of this choice; enable it with `--use-opd` and its KL/loss coefficient |
| `--normalize-advantages` | flag | False | - | Whether to normalize advantages |
| `--disable-grpo-std-normalization` | flag | - | - | Disable GRPO standard deviation normalization (from [Dr.GRPO](https://arxiv.org/pdf/2503.20783)) |
| `--disable-rewards-normalization` | flag | - | - | Disable reward normalization |
| `--disable-compute-advantages-and-returns` | flag | - | - | Disable advantage and return computation. Used for SFT or custom loss functions |

### Loss Function

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--loss-type` | str | policy_loss | `policy_loss`, `sft`, `custom_loss` | Loss type. `policy_loss` runs PPO/GRPO/etc. RL training; `sft` runs supervised fine-tuning (see [SFT Configuration](#sft-configuration)); `custom_loss` requires `--custom-loss-function-path`. `sft_loss` is a deprecated alias for `sft`. |
| `--custom-loss-function-path` | str | None | - | Custom loss function path |
| `--eps-clip` | float | 0.2 | - | PPO clipping range (lower bound) |
| `--eps-clip-high` | float | None | - | PPO clipping upper bound. When None, equals `--eps-clip` |
| `--eps-clip-c` | float | None | - | Dual-clip PPO value lower bound ([paper](https://arxiv.org/pdf/1912.09729)) |
| `--value-clip` | float | 0.2 | - | Value function clipping range |
| `--entropy-coef` | float | 0.0 | - | Entropy loss coefficient |

### PPO Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gamma` | float | 1.0 | Discount factor for PPO GAE |
| `--lambd` | float | 1.0 | Lambda for PPO GAE |
| `--value-clip` | float | 0.2 | Clipping range for PPO Critic value loss |
| `--use-rollout-logprobs` | flag | False | Use Rollout logprobs as PPO old-policy logprobs; required by the provided colocate topology |

PPO currently supports synchronous colocate mode and requires `critic` and `advantages` entries in `--resource`. See [PPO Training](./ppo-training.md) for resource topology, checkpoint consistency, and KL constraints.

### KL Divergence Related

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--kl-coef` | float | 0.0 | KL penalty coefficient for reward shaping. Synchronous PPO resets non-zero values to `0.0`. Cannot be non-zero simultaneously with `--kl-loss-coef` |
| `--use-kl-loss` | flag | False | Enable loss-level KL for GRPO-like algorithms. Synchronous PPO automatically disables this option |
| `--kl-loss-coef` | float | 0.0 | KL penalty coefficient added to the final policy loss. Cannot be non-zero simultaneously with `--kl-coef` |
| `--kl-loss-type` | str | k1 | `k1`, `k2`, `k3`, `low_var_kl` | KL loss type |
| `--use-unbiased-kl` | flag | False | Enable unbiased KL estimation |
| `--ref-update-interval` | int | None | Reference model update interval in Rollout steps. None means no update |

### SAPO Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sapo-tau-pos` | float | 1.0 | SAPO positive advantage temperature |
| `--sapo-tau-neg` | float | 1.05 | SAPO negative advantage temperature |

### Critic Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-critic-only-steps` | int | 0 | Number of steps to train Critic only |
| `--critic-train-only` | flag | False | Train Critic model only |
| `--critic-load` | str | None | Critic checkpoint to load. When None, equals `--load` |
| `--critic-save` | str | None | Critic checkpoint output directory |
| `--critic-lr` | float | None | Critic learning rate. When None, equals `--lr` |
| `--critic-lr-warmup-iters` | int | 0 | Number of iterations for linear warmup of Critic model |

### Off-Policy Correction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-rollout-logprobs` | flag | False | Use Rollout's logprobs when computing importance sampling ratio. When not set, uses Actor model's logprobs |
| `--use-tis` | flag | False | Enable TIS (Truncated Importance Sampling) off-policy importance sampling |
| `--tis-clip` | float | 2.0 | Upper clipping threshold for importance sampling ratio |
| `--tis-clip-low` | float | 0 | Lower clipping threshold for importance sampling ratio |
| `--custom-tis-function-path` | str | None | Custom TIS/RS function path |
| `--custom-pg-loss-reducer-function-path` | str | None | Custom pg_loss reducer function path. When set, pg_loss uses custom reducer while other metrics use default sum_of_sample_mean |

### Routing Replay and OPSM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-routing-replay` | flag | False | Enable Routing Replay ([paper](https://arxiv.org/abs/2507.18071)) |
| `--use-rollout-routing-replay` | flag | False | Enable Rollout Routing Replay ([paper](https://arxiv.org/abs/2510.11370)). Automatically enables `--use-routing-replay` |
| `--use-opsm` | flag | False | Enable Off-Policy Sequence Masking (OPSM) |
| `--opsm-delta` | float | 1e-4 | OPSM threshold |

### Other Training Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--reset-optimizer-states` | flag | False | Whether to reset optimizer state after each Rollout |
| `--use-rollout-entropy` | flag | False | Whether to compute entropy when calculating logprobs. Used for special loss mask |
| `--get-mismatch-metrics` | flag | False | Whether to compute mismatch metrics. Requires setting `--custom-tis-function-path` |

---

## Parameter Freezing and Selective Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--only-train-params-name-list` | str (list) | None | List of regex patterns for parameters to train. Other parameters are frozen. Cannot be used simultaneously with `--freeze-params-name-list`. Example: `--only-train-params-name-list experts` |
| `--freeze-params-name-list` | str (list) | None | List of regex patterns for parameters to freeze. Other parameters remain trainable. Example: `--freeze-params-name-list embedding output_layer` |

---

## SFT Configuration

These flags only apply under `--loss-type sft`. The SFT pipeline runs an `SFTStreamingDataset` producer that pushes packed samples into the TransferQueue under partition `sft_<rollout_id>`, and an `MegatronTrainRayActor` consumer that trains, periodically evaluates (PPL), and optionally runs generative prediction.

### Training Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval-size` | float | None | Carve a held-out eval split from `--prompt-data` instead of supplying a separate `--eval-prompt-data`. A value <1 is treated as a fraction of the train dataset (e.g. `0.05` → last 5%); a value ≥1 is treated as an absolute sample count. The reserved tail is removed from the train pool so train and eval samples never overlap. Mutually exclusive with `--eval-prompt-data`. |
| `--sft-predict-interval` | int | None | Every N rollout steps run a generative predict pass over the eval set and write completions to `<save>/predict/predictions_step_<rollout_id>.jsonl`. Setting this flag implicitly spins up the Rollout role under SFT (SGLang must be online). Controls the generative complement to the always-on PPL eval (`--eval-interval`). **Requires** `--save` (writes under `<save>/predict/`) and at least one eval source (`--eval-prompt-data` / `--eval-config` / `--eval-size`). |

### Streaming Dataset Prefetch

The SFT producer uses its own `PrefetchBuffer` independent from the rollout data source's `--prefetch-*` knobs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sft-prefetch-buffer-size` | int | 256 | Max pre-loaded samples held by the SFT streaming dataset's PrefetchBuffer. Set to 0 to disable prefetching; the producer then falls back to an `asyncio.gather` path over the ProcessorPool for batch-level parallelism. |
| `--sft-prefetch-chunk-size` | int | 32 | Chunk size dispatched to the SFT prefetch thread-pool per round. |
| `--sft-prefetch-num-workers` | int | 4 | Worker threads inside the SFT PrefetchBuffer for I/O-bound media decoding (video/image). |

### Oversize Sample Handling

How the SFT producer handles samples whose tokenized + media-expanded length exceeds the per-GPU capacity (`--max-tokens-per-gpu × --context-parallel-size`). All branches log a WARNING per oversized sample.

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--sft-oversize-strategy` | str | keep | `skip`, `keep`, `truncate_left`, `truncate_right`, `custom` | `skip` drops the sample; `keep` returns it unchanged (may OOM downstream); `truncate_left` keeps the last `capacity` tokens; `truncate_right` keeps the first `capacity` tokens; `custom` delegates to `--sft-oversize-custom-function-path`. ⚠ Truncating multimodal samples in-place may misalign `multimodal_train_inputs` — use `custom` if you also need to trim media inputs. |
| `--sft-oversize-custom-function-path` | str | None | - | Required when `--sft-oversize-strategy custom`. Importable path to a function with signature `def truncate(tokens, loss_mask, capacity, idx) -> (tokens, loss_mask) \| None`. Returning `None` is treated as `skip`. |

::: tip Related dataset flags
SFT also uses the general dataset flags from [Data Configuration](#data-configuration), in particular `--input-key`, `--label-key`, `--conversation-key-map` (for sharegpt-style datasets), `--multimodal-keys`, `--system-prompt`, and `--tool-key`.
:::

---

## Evaluation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval-interval` | int | None | Evaluation interval in Rollout rounds |
| `--eval-prompt-data` | str (list) | None | Evaluation datasets in format: `dataset_name /path/to/data.jsonl`. Can specify multiple pairs |
| `--eval-config` | str | None | OmegaConf YAML/JSON evaluation config file path. When set, overrides `--eval-prompt-data` |
| `--eval-function-path` | str | None | Evaluation generation function path. When None, uses `--rollout-function-path` |
| `--skip-eval-before-train` | flag | False | Whether to skip evaluation before training |
| `--eval-input-key` | str | None | Key for input field in evaluation data. When None, uses `--input-key` |
| `--eval-label-key` | str | None | Key for label field in evaluation data |
| `--eval-tool-key` | str | None | Key for tool field in evaluation data |
| `--n-samples-per-eval-prompt` | int | 1 | Number of samples per evaluation prompt |
| `--eval-temperature` | float | None | Sampling temperature for evaluation |
| `--eval-top-p` | float | None | Top-p parameter for evaluation |
| `--eval-top-k` | int | None | Top-k parameter for evaluation |
| `--eval-max-response-len` | int | None | Maximum response length for evaluation |
| `--eval-max-prompt-len` | int | None | Maximum prompt length for evaluation |
| `--eval-min-new-tokens` | int | None | Minimum new tokens generated for evaluation |
| `--eval-max-context-len` | int | None | Maximum context length for evaluation. When None, equals `--rollout-max-context-len` |

---

## Reward Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--rm-type` | str | None | Built-in reward model type |
| `--rm-type-fallback` | str | None | Fallback for unknown/missing reward types: `zero` scores 0.0 with a warning, a registered type name routes there. None keeps the error behavior |
| `--rm-type-infer` | flag | False | Infer the reward type from the sample label via registered matchers when no explicit type is set; conflicts warn and the explicit type wins |
| `--custom-rm-path` | str | None | Custom reward function path. A single-sample function receives one sample; a batched/group function receives the complete sample list and returns one result per sample. Bypasses format-aware routing |
| `--reward-max-concurrency` | int | 64 | Maximum concurrent reward calls in each caller process. One custom batch/group invocation counts as one call |
| `--reward-num-workers` | int | 16 | Number of Ray actors used to run synchronous rewards. Async custom rewards do not use these workers |
| `--reward-key` | str | None | Key to extract reward value when reward function returns dict |
| `--eval-reward-key` | str | None | Reward key for evaluation. When None, equals `--reward-key` |
| `--group-rm` | flag | False | Whether to compute reward for entire group |
| `--rm-url` | str | None | Remote reward model service URL (for `--rm-type remote_rm`) |
| `--custom-reward-post-process-path` | str | None | Custom reward postprocessing function path. Default is GRPO normalization |
| `--custom-convert-samples-to-train-data-path` | str | None | Custom function to convert samples to training data. Signature: `def convert_samples_to_train_data(args, samples) -> dict` |

---

## GenRM (Generative Reward Model)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--genrm-model-path` | str | None | GenRM model path. Enables GenRM when set |
| `--genrm-num-gpus` | int | 1 | Total GPUs for GenRM |
| `--genrm-num-gpus-per-engine` | int | 1 | GPUs per GenRM engine instance |
| `--genrm-engine-config` | json | None | GenRM engine initialization parameters. Example: `'{"dp_size": 1, "pp_size": 1, "max_total_tokens": 8192}'` |
| `--genrm-sampling-config` | json | None | GenRM sampling parameters. Available keys: `temperature` (default 0.1), `top_p` (default 1.0), `top_k` (default -1), `max_response_len` (default 4096) |

---

## On-Policy Distillation (OPD)

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--use-opd` | flag | False | - | Enable On-Policy Distillation. Must also specify `--opd-type` |
| `--opd-type` | str | None | `sglang`, `megatron` | OPD type. `sglang`: fetch teacher model logprobs from external SGLang server; `megatron`: load teacher model via `--opd-teacher-load` |
| `--opd-kl-coef` | float | 1.0 | - | OPD KL penalty coefficient |
| `--opd-teacher-load` | str | None | - | OPD teacher model checkpoint path. Required when `--opd-type=megatron` |
| `--opd-teacher-ckpt-step` | int | None | - | OPD teacher model checkpoint step |

---

## Fault Tolerance Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-fault-tolerance` | flag | False | Whether to enable fault tolerance during Rollout |
| `--use-health-check` | flag | False | Whether to enable global health check system. Controller's HealthManager monitors all services and triggers automatic restart on failure |
| `--max-global-restart` | int | 3 | Maximum number of global restarts allowed. Training terminates after exceeding. Only effective when `--use-health-check` is enabled |
| `--rollout-health-check-interval` | float | 30.0 | Rollout engine health check interval in seconds |
| `--rollout-health-check-timeout` | float | 30.0 | Rollout engine health check timeout in seconds |
| `--rollout-health-check-first-wait` | float | 0 | Initial wait time before starting health check in seconds. Increase this value when using deepgemm |

---

## Elastic Scaling Configuration

### Autoscaler

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--autoscaler-config` | str | None | Path to autoscaler YAML configuration file. Enables autoscaling when set, disabled when not set. Example: `--autoscaler-config relax/utils/autoscaler/autoscaler.yaml` |

For autoscaler YAML configuration details, see [`relax/utils/autoscaler/autoscaler.yaml`](https://github.com/redai-studio/Relax/blob/main/relax/utils/autoscaler/autoscaler.yaml).

### Scale-Out Operation Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--scale-out-timeout` | float | 300.0 | - | Timeout for all scale-out operations (engine startup, connect, health check, weight sync) in seconds |
| `--scale-out-partial-success-policy` | str | rollback_all | `rollback_all`, `keep_partial` | Policy for partial success during scale-out. `rollback_all` reverts all engines on any failure; `keep_partial` keeps successfully scaled engines |
| `--scale-weight-sync-precheck` | bool | True | - | Run an independent NCCL precheck before scale-out weight sync; fail-closed on incompatible transport. On by default; disable with `--no-scale-weight-sync-precheck` (disabling drops the fail-closed protection). See [Elastic Rollout · Weight Sync Precheck](./elastic-rollout.md#weight-sync-precheck) |

### Precheck Env Vars

These environment variables are for tuning and troubleshooting; all are optional (defaults shown). Read at runtime by `relax/utils/env.py`.

| Env var | Type | Default | Description |
|----------|------|--------|------|
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES` | int | 536870912 (512 MiB) | Minimum free GPU memory (bytes) for the Stage 2 probe; below it the probe returns `INSUFFICIENT_GPU_MEMORY` and fails closed. SGLang reserves 85–90% of VRAM; lowering this risks probe OOM. |
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE` | int | 18000 | Lower bound of the Stage 2 probe subprocess rendezvous port range |
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX` | int | 20000 | Upper bound of the Stage 2 probe subprocess rendezvous port range. **Cluster network policy must allow `[PORT_BASE, PORT_MAX]`**, otherwise the probe cannot connect → `PROBE_FAILED` |
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS` | int | 2 | Max attempts for the Stage 2 probe |
| `RELAX_SCALE_OUT_MAX_REASON_ITEMS` | int | 3 | Max number of scale-out failure reasons surfaced in the TUI / stable log |
| `RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN` | int | 120 | Truncation length per failure reason |
| `RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN` | int | 512 | Total truncation length for failure reasons (prevents excessively long raw-error output) |

### Scale-In Operation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scale-in-drain-timeout` | float | 30.0 | Timeout to wait for in-flight requests to drain before force-aborting (seconds) |
| `--scale-in-shutdown-timeout` | float | 30.0 | Timeout for graceful SGLang engine shutdown; `ray.kill` is used if exceeded (seconds) |

---

## Logging and Monitoring Configuration

### TensorBoard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-tensorboard` | flag | False | Enable TensorBoard logging |
| `--tb-project-name` | str | None | TensorBoard log directory. Defaults to environment variable `TENSORBOARD_DIR` |
| `--tb-experiment-name` | str | None | TensorBoard experiment name |

### ClearML

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-clearml` | flag | False | Enable ClearML logging |

### Metrics Service

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-metrics-service` | flag | False | Enable centralized metrics collection and reporting service |
| `--timeline-dump-dir` | str | None | Timeline trace event export directory (Chrome Trace format). Timeline tracing disabled if not set |

### Logging Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--log-passrate` | flag | False | Enable pass@n pass rate logging |
| `--log-multi-turn` | flag | False | Enable multi-turn Rollout information logging |
| `--log-reward-category` | str | None | Log reward category statistics. Specify key in reward dict |

### Notifications

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--notify-urls` | str | None | Apprise notification URL list (comma-separated) |

---

## Router Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-slime-router` | flag | False | Use SlimeRouter for text-based routing instead of SGLang's token-based routing |
| `--slime-router-middleware-paths` | str (list) | "" | List of middleware paths |
| `--slime-router-timeout` | float | None | SlimeRouter HTTP request timeout in seconds |
| `--slime-router-max-connections` | int | None | SlimeRouter HTTP client maximum connections |
| `--slime-router-health-check-failure-threshold` | int | 3 | Mark worker as unhealthy after this many consecutive failures |
| `--slime-router-sticky` | flag | False | Enable sticky-session routing: pin a routing key (read from the `X-SMG-Routing-Key` header) to a worker so repeated requests for the same key reuse that worker's prefix/KV cache. A live pin is never redistributed (only remapped when its worker leaves the healthy set). Requires `--use-slime-router` |
| `--slime-router-sticky-idle-secs` | float | 600.0 | Evict a sticky key→worker assignment after it has been idle (not routed to) for this many seconds, bounding the map against unbounded routing-key cardinality. Requires `--slime-router-sticky` |

---

## Other Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train-env-vars` | json | {} | Additional environment variables for training process |
| `--train-memory-margin-bytes` | int | 1GB (1024³) | Reserved space for training memory allocation |
| `--disable-weights-backuper` | flag | - | Disable weight backup to save host memory |
| `--custom-model-provider-path` | str | None | Custom model provider function path |
| `--recompute-loss-function` | flag | False | Whether to recompute loss function to save VRAM |
| `--log-probs-chunk-size` | int | -1 | Chunk size for computing log probs. Used to save VRAM |
| `--allgather-cp` | flag | False | - |
| `--model-name` | str | None | Model name for Megatron to HF weight conversion. Inferred from `AutoConfig.from_pretrained(hf_checkpoint)` if not set |
| `--only-load-weight` | flag | False | Reference model and actor fwd only load weights |
| `--rlsp-server-port` | int | 8234 | RLSP Server HTTP port |
| `--custom-config-path` | str | None | Custom parameter YAML config file path. Key-value pairs in file override existing parameters |

---

## Debug & Profiling Parameters

### Debug

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--debug-rollout-only` | flag | False | Run Rollout only, no training |
| `--debug-train-only` | flag | False | Run training only, no Rollout |
| `--load-debug-rollout-data` | str | None | Load debug Rollout data. Automatically enables `--debug-train-only` when set |
| `--load-debug-rollout-data-subsample` | float | None | Subsample debug Rollout data to accelerate debugging |
| `--save-debug-rollout-data` | str | None | Save Rollout data. Path supports `{rollout_id}` placeholder |
| `--save-debug-train-data` | str | None | Save training data. Path supports `{rollout_id}` placeholder |
| `--dump-details` | str | None | Export all training details for post-hoc analysis |
| `--check-weight-update-equal` | flag | False | Check if weight updates are equal |
| `--enable-cuda-memory-check` | flag | False | Enable memory check around low-level NCCL communication calls. Logs available GPU memory before each collective and attaches memory info to exceptions on failure |

### Training Performance Profiling

These parameters control the PyTorch Profiler for training steps. Trace files are saved to `traces/<tb_experiment_name>/train_trace/` by default.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-pytorch-profiler` | flag | False | Enable PyTorch's built-in profiler to record CUDA kernels, CPU ops, and communication during training (from Megatron) |
| `--profile-step-start` | int | 10 | Step offset at which to start profiling (**inclusive**, from Megatron). Counts from 0 since the current training launch, not absolute rollout ID; resets on checkpoint resumption |
| `--profile-step-end` | int | 12 | Step offset at which to stop profiling (**inclusive**, from Megatron). Same counting semantics as above. E.g. start=10, end=12 profiles steps 10, 11, 12 (3 steps) |
| `--profile-target` | str (list) | train_overall | Profiling targets: `train_overall`, `train_actor`, `train_log_probs` |
| `--profile-with-stack` | flag | False | Record stack information in profiler traces |
| `--profile-with-memory` | flag | False | Record memory information in profiler traces |
| `--profile-with-flops` | flag | False | Estimate FLOPs in profiler traces |

### GPU Memory Profiling

These parameters control GPU memory snapshot collection for diagnosing memory leaks and OOM issues. Snapshot files can be viewed with PyTorch Memory Viz tools (`torch.cuda.memory._viz`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--record-memory-history` | flag | False | Enable CUDA memory allocation history recording (from Megatron). Records call stacks and tensor info for each allocation/deallocation, and auto-dumps a snapshot on OOM |
| `--memory-snapshot-path` | str | snapshot.pickle | Memory snapshot filename (from Megatron) |
| `--memory-snapshot-dir` | str | None | Memory snapshot output directory. Defaults to `traces/<tb_experiment_name>/memory_snapshot` |
| `--memory-snapshot-num-steps` | int | None | Proactively dump a memory snapshot after the specified number of steps (0-indexed, i.e., setting 3 means dump after step 2) |
| `--memory-recorder` | str | torch | Memory recorder backend: `torch` (PyTorch built-in), `memray` (requires `pip install memray`) |

### Network

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--http-proxy` | str | None | HTTP proxy address |
| `--use-distributed-post` | flag | False | Use distributed POST requests |

---

## Environment Configuration

Relax uses `configs/env.yaml` to configure runtime environment variables:

```yaml
env_vars:
  TOKENIZERS_PARALLELISM: 'true'
  NCCL_DEBUG: 'WARN'
  CUDA_DEVICE_MAX_CONNECTIONS: '1'
  GLOO_SOCKET_IFNAME: "eth0"
  TP_SOCKET_IFNAME: "eth0"
```
