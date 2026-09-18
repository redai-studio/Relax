# 扩散生成式 RL

Relax 不仅能对自回归 LLM 做 RL 后训练，也能对**扩散 / 流匹配（flow-matching）生成模型**做 RL 后训练。该路径运行 [FlowGRPO](https://arxiv.org/abs/2505.05470) 风格的策略优化：rollout 引擎使用随机（SDE）采样器生成图像并返回去噪轨迹，奖励模型对生成的媒体打分，actor 再回放（replay）其中一部分去噪步来计算裁剪后的 PPO 目标。

训练后端是 **FSDP2**（`--train-backend fsdp`）而非 Megatron，rollout 引擎是 **SGLang 原生 diffusion server** 而非文本 `srt` 引擎。其余部分——Ray Serve 控制器、TransferQueue 数据面、colocate GPU 分时复用、指标与 checkpoint——与[架构设计](./architecture.md)中描述的 Relax 机制完全一致。

## 支持范围

::: warning 能力边界
生成式路径是刻意收敛的。以下约束会在启动预检（`relax/backends/fsdp/arguments.py` 中的 `validate_generative_config`）中**强制校验**，并一次性列出全部违规项后中止任务；完整清单见下文「预检校验」一节。

- **只有文生图。** `GENERATION_TASKS` 是 `("t2i",)`。图像编辑、视频、音视频任务不在本分支上。
- **只有同步 colocate。** 必须传 `--colocate`；`--fully-async` 与 `--hybrid` 会被拒绝。
- **全参微调或 LoRA。** `--fsdp-trainable-mode` 取 `full` 或 `lora`，且必须与 `--lora-rank` 一致（见 [LoRA](#lora)）。
- **没有 KL-to-reference。** 这里的 FlowGRPO 没有 reference model，所以非零的 `--kl-coef` 或 `--use-kl-loss` 会被拒绝，而不是静默忽略。
- **不支持 CFG。** `guidance_scale` 必须为 `1.0`；actor 只回放一次正向条件前向，CFG rollout 会训练一个它从未采样过的策略。
- **`sde_type="sde"` 下不能训练 SDE step 0**（见下文「训练的 SDE step」一节）。
  :::

### 任务与适配器

`--generation-task` 只接受 `t2i`。跑哪个模型由 `--model-adapter-path` 指向的适配器决定：

| 适配器 | Dotpath | 声明的任务 | 状态 |
|---|---|---|---|
| Qwen-Image | `relax.models.qwen_image.adapter.QwenImageAdapter` | `t2i` | **已端到端验证**，附带两个开箱即用的启动脚本 |

::: tip 接入自己的模型
运行时只依赖 `relax/models/generative.py` 里的 `GenerativeModelAdapter` 协议（`load_train_model`、`build_rollout_request`、`validate_rollout_response`、`pack_trajectory`、`replay_transition`、`artifact_tracks`、`weight_name_map`）。接一个新模型族就是加一个 dotpath —— 外加把它的任务名加进 `GENERATION_TASKS`，这样任务词表里就不会出现没有可用 replay 路径的任务。
:::

## 架构

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    Ray cluster — 8 GPUs, synchronous colocate             │
│                                                                           │
│  ┌──────────────────────────┐            ┌───────────────────────────┐    │
│  │  FSDPTrainRayActor       │  weight    │  SGLangNativeGeneration   │    │
│  │  relax/backends/fsdp     │  sync      │  Engine (one per GPU)     │    │
│  │                          │  (bf16,    │                           │    │
│  │  transformer  (FSDP2)    │──buckets,─►│  DiT + VAE + text encoder │    │
│  │  AdamW (offloaded)       │  CUDA IPC) │  POST /rollout/generate   │    │
│  └──────────┬───────────────┘            └─────────────┬─────────────┘    │
│             │                                          │                  │
│             │                          trajectory +    │                  │
│             │                          decoded media   ▼                  │
│             │                          ┌──────────────────────────────┐   │
│             │                          │ artifact_root/<task>/        │   │
│             │                          │   rollout_*/group_*.safeten. │   │
│             │                          │   rollout_*/..._sample_*.png │   │
│             │                          └─────────────┬────────────────┘   │
│             │                                        │                    │
│             │                                        ▼                    │
│             │                          ┌──────────────────────────────┐   │
│             │                          │ Reward scorer, e.g. PickScore│   │
│             │                          │ group-normalized advantages  │   │
│             │                          └─────────────┬────────────────┘   │
│             │        TransferQueue train rows        │                    │
│             └────────────────────────────────────────┘                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### 一个训练步

1. **Rollout** —— `relax.engine.rollout.native_generation.generate_rollout` 先为本轮解析一次训练用的 SDE step（这样所有候选、所有 engine 都用同一组），回收过期的 artifact 目录，然后把一批 prompt group 按 wave 分发到所有 engine。组内每个候选拿到不同的 `sample_id`，它决定该候选的初始 latent 与逐步噪声，于是组内奖励方差是真实存在的。
2. **轨迹 sidecar** —— engine 返回 DiT 轨迹 latent、timesteps、逐步 rollout log-prob 与冻结条件（`denoising_env`），并回显请求里的 `height` / `width`，让训练侧能还原真实 latent grid。适配器打包后由 driver 在 `--artifact-root` 下为每组写一个 safetensors sidecar；解码出的图片以 PNG 写在旁边。大张量永远不进 TransferQueue。
3. **Reward** —— `relax.engine.rewards.generative.post_process`（通过 `--custom-reward-post-process-path` 接线）每轮 rollout 触发一次，批量给媒体打分，把每个奖励分量**在其 prompt 组内**归一化，再按权重合成单个 advantage。
4. **更新** —— actor 重新加载 sidecar，把本地切片划分成 `--num-updates-per-batch` 份互不相交的更新，通过 transformer 回放被训练的 SDE step，重算 Flow-SDE transition log-prob，并应用 `relax.models.flow_grpo` 的裁剪 PPO 目标。
5. **权重同步** —— transformer 被聚合并通过 CUDA IPC 流式发给每个 engine（rank *j* → engine *j*，同一张物理卡，按 device UUID 匹配），校验 manifest 后提交，然后 actor 再次休眠，把卡交还给下一轮 rollout 的 engine。

FlowGRPO 的数学部分（`flow_sde_transition_moments`、`flow_sde_log_prob`、`replay_transition_logp`、`grpo_clip_loss`、`normalize_grouped`）位于 `relax/models/flow_grpo.py`，是不依赖任何框架的纯 `torch` 代码，被 rollout 引擎与 actor replay 共用，是唯一事实来源。

## 前置条件

### Docker 镜像

生成式路径运行在标准 Relax 训练镜像里。该镜像以 `lmsysorg/sglang:v0.5.15.post1-cu129` 为基线，应用 `docker/patch/sglang/v0.5.15.post1.patch`，并从 `requirements.txt` 安装 diffusion 运行时依赖。

```bash
docker build -f docker/Dockerfile -t relax:latest .
```

镜像包含 `diffusers>=0.37.0`、`imageio[ffmpeg]`、`soundfile` 和 `peft>=0.20.0,<0.21.0`；不需要单独的 overlay 镜像。

::: tip 补丁提供了什么
Rollout 本身是上游能力：SGLang 的 diffusion server 已经提供 `POST /rollout/generate`，返回轨迹、逐步 log-prob 与冻结条件。Relax 合并版 SGLang 补丁额外提供**内存态 diffusion 权重更新路径**（`/update_weights_from_tensor`）、LoRA adapter 端点（`/set_lora_from_tensor`），以及一个在 SGLang 已把 DiT 转成 LoRA layer 之后拒绝全量张量同步的保护。补丁还把 driver 的 sigma、逐步 seed 与 `x_T` 配方接入 SGLang 请求链路，并包含 Qwen-Image 文本截断窗口和 `rollout_sde_type="dance"` 的 log-prob 实现。没有权重更新路径就只能每步从磁盘重载；缺一致性补丁则 replay log-prob 会和 rollout 轨迹不匹配。补丁布局、hash 与校验命令见 `examples/diffusion/README.md`。

`SGLangNativeGenerationEngine` 启动 server 前只做静态契约检查，确认当前 SGLang 源码已经应用 `docker/patch/sglang/v0.5.15.post1.patch`；不再在 worker 进程内安装 monkey patch。
:::

### 模型

```bash
# 策略模型
hf download Qwen/Qwen-Image --local-dir ${MODEL_DIR}/Qwen-Image

# 奖励模型（PickScore v1，一个微调过的 CLIP-H）
hf download yuvalkirstain/PickScore_v1 --local-dir ${MODEL_DIR}/PickScore_v1
```

`QwenImageAdapter.load_train_model` 只用 `diffusers.QwenImageTransformer2DModel` 加载 `transformer` 子目录 —— VAE 与 text encoder 由 rollout 引擎以冻结方式加载，FSDP actor 永远不碰。

## 数据准备

原生生成使用**统一 JSONL** schema，一行一个 JSON 对象：

```json
{"prompt": "a red panda in a teacup", "metadata": {"task": "t2i", "sample_id": "pickapic_0000001"}}
```

`examples/diffusion/` 下的三个脚本负责构建并把关这个文件。下面的命令产出的文件名，正是启动脚本默认要读的那些。

### 1. 转换公开数据集

```bash
python3 examples/diffusion/prepare_data.py \
  --source pickapic \
  --input /path/to/pick_a_pic_prompts \
  --task t2i \
  --output ${DATA_DIR}/raw/t2i/pickapic.jsonl
```

`--source` 从 `CONVERTERS` 中选择转换器：`pickapic`，或 `prompts`（通用的 parquet `prompt`/`caption` 列、带 `prompt` 字段的 JSONL，或每行一个 prompt 的纯文本）。

### 2. 去重、过滤与切分

```bash
python3 examples/diffusion/curate_data.py \
  --input ${DATA_DIR}/raw/t2i/pickapic.jsonl \
  --out-dir ${DATA_DIR}/processed/t2i \
  --prefix pickapic_ --min-prompt-words 6 \
  --eval-size 2048 --eval-subset-sizes 64 256 --no-check-media
```

这会写出 `pickapic_train.jsonl`、`pickapic_eval.jsonl` 以及 `pickapic_eval64.jsonl` / `pickapic_eval256.jsonl` 两个子集 —— 正是两个启动脚本里 `PROMPT_SET` 与 `EVAL_SET` 的默认值。

`--min-prompt-words 6` 加上 2048 条留出集，是两个启动脚本使用的 Pick-a-Pic 对齐预设：未过滤的一两个词的 caption 打分几乎没有组内方差，会让 GRPO 拿不到信号。所有变换都是确定性的 —— prompt 按 prompt + 排序后媒体路径的稳定内容 hash 去重，缺媒体的记录被丢弃（用 `--no-check-media` 跳过该检查），train/eval 划分来自对 `sample_id` 做 hash 而非 RNG —— 所以重跑得到完全相同的划分。不加 `--prefix` / `--eval-size` / `--eval-subset-sizes` 时，得到的是普通的 `train.jsonl` / `eval.jsonl` 和基于 `--eval-fraction` 的切分。

### 3. 上卡前先校验

```bash
python3 examples/diffusion/inspect_data.py \
  --train ${DATA_DIR}/processed/t2i/pickapic_train.jsonl \
  --eval  ${DATA_DIR}/processed/t2i/pickapic_eval256.jsonl \
  --decode-media
```

检查必填键、合法的 `metadata.task`、`<image>` / `<video>` 占位符与媒体的一致性，以及 train/eval 切分是否泄漏。`--decode-media` 还会打开每个被引用的文件确认能解码。

## 快速开始

已验证的配方是 8 卡上的 Qwen-Image T2I **LoRA**，使用 adapter 同步模式：

```bash
cd /path/to/Relax
export EXP_DIR=/your/shared/fs/native-generation   # models/、data/、runs/ 都在这里

bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
```

脚本期望 `${EXP_DIR}/models/Qwen-Image`、`${EXP_DIR}/models/PickScore_v1` 以及 `${EXP_DIR}/data/processed/t2i/` 下整理好的 prompt 文件；可用 `MODEL_DIR`、`DATA_DIR`、`ARTIFACT_ROOT`、`SAVE_DIR` 或 `EXP_DIR` 覆盖位置。它提交的 Ray 任务包含：

```bash
--resource '{"actor": [1, 8], "rollout": [1, 8]}'   # colocate：两个角色共用同一批 8 张卡
--colocate
--train-backend fsdp
--generation-task t2i
--model-adapter-path      relax.models.qwen_image.adapter.QwenImageAdapter
--rollout-engine-class-path relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine
--rollout-function-path   relax.engine.rollout.native_generation.generate_rollout
--custom-convert-samples-to-train-data-path relax.engine.rollout.native_generation.convert_samples_to_train_data
--custom-reward-post-process-path           relax.engine.rewards.generative.post_process
```

两个配方的完整参数列表（含每个取值的理由）见下文「参考配置」一节。最关键的几何参数：

| 设置 | LoRA（已验证） | 全参 | 说明 |
|---|---|---|---|
| `--rollout-batch-size` × `--n-samples-per-prompt` | 32 × 8 = 256 | 8 × 8 = 64 | GRPO 的组几何；`--global-batch-size` 是二者之积 |
| `--num-updates-per-batch` | 2 | 2 | 把 batch 切成 2 份**互不相交**的优化器更新 —— 见下面的告警 |
| `--sampling-config` | 384×384、12 步、`eta=0.7`，每轮从 `[1,2,3,4,5]` 重抽 3 个训练 SDE step | 同左 | 生成 12 步去噪，只训练其中 3 步 |
| `--generative-advantage-std-mode` | `group` | `group` | 默认使用 Relax/Text GRPO 的组内 std；需要复现实验性参考 diffusion 口径时可显式设为 `batch` |
| `--lr` | `3e-4` | `3e-5` | 从 `B = 0` 起步的 adapter 需要约 10 倍于全参的学习率 |
| `--eps-clip` | `1e-4` | `1e-4` | FlowGRPO 用极小的 PPO clip 区间，而不是文本 RL 的 `0.2` |
| `--reward-runtime` | `colocate` | `colocate` | actor 已 offload 时，PickScore 在 rollout GPU 上进程内打分 |

::: warning `--num-updates-per-batch` 是切分，不是复用
rollout batch 会被划分成 N 份**互不相交**、样本数相等的优化器更新；每个样本只参与一次优化器步。因此优化器步的边界是 `global_batch_size / num_updates_per_batch`，而不是 `global_batch_size`。

`--num-updates-per-batch 1` 时重要性比恒为 1，配合组中心化的 advantage，clipped loss 按构造为 0，clipping 永远不会触发。请用 `2` 或更大。
:::

### 启动脚本

使用 `scripts/training/diffusion/` 下的启动脚本。它们直接传入已验证的 `train.py` 参数，是这个分支上的运行准线：

```bash
export EXP_DIR=...

# 全参微调参考
bash scripts/training/diffusion/run-qwen-image-t2i-8xgpu.sh

# 已验证的 LoRA recipe
bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
```

::: warning
`examples/diffusion/` 下的 YAML profile 是参考配置片段，不是启动路径。请从 `scripts/training/diffusion/` 下的启动脚本开始，那里才是真正在 8×96 GB 上跑过的设置。
:::

## 配置参考

本节是选择 `--train-backend fsdp` 后生效的全部参数参考。这些参数由 `relax/backends/fsdp/arguments.py` 中的 `add_generative_arguments` / `add_fsdp_arguments` 注册，并在 `relax/utils/arguments.py` 中接入主解析器。通用（非生成式）参数见[配置说明](./configuration.md)。

::: tip 默认值是惰性的
下列每个参数的默认值都不会影响标准的 Megatron token-RL 路径，因此把生成式参数组加入解析器本身没有任何副作用，只有传入 `--train-backend fsdp` 后才会生效。
:::

### 后端选择

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--train-backend` | `megatron` \| `fsdp` | `megatron` | 选择训练 actor。`fsdp` 会启用 `FSDPTrainRayActor` 与生成式预检校验。 |

### 任务与模型

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--generation-task` | `t2i` | `None` | 生成任务。`fsdp` 下**必填**。同时决定 checkpoint 与 artifact 的子目录名。 |
| `--model-path` | str | `None` | 交给 adapter `load_train_model` 的 HF 格式模型目录。`fsdp` 下**必填**。 |
| `--model-revision` | str | `None` | 可选 revision。参与 base-model hash，并透传给 rollout engine。 |
| `--model-adapter-path` | str | `None` | 指向 `GenerativeModelAdapter` 实现的 dotpath。`fsdp` 下**必填**。 |

`GENERATION_TASKS` 现在是 `("t2i",)`。图像编辑、视频、音视频任务及其 Wan / LTX adapter 不在本分支上；通过 `--model-adapter-path` 自带 adapter 的调用方也必须把对应任务名加进 `GENERATION_TASKS`，这样任务词表里就不会出现「能配出来但没有可用 replay 路径」的任务。

随仓库发布的 adapter：`relax.models.qwen_image.adapter.QwenImageAdapter`（`family = "qwen_image"`，`supported_tasks = ("t2i",)`）。

### Rollout 接线

生成式 rollout 替换四个可插拔钩子。其中三个是框架级的 `--custom-*` / `--rollout-function-path`，第四个是生成式专属的。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--rollout-engine-class-path` | str | `None` | rollout engine 类的 dotpath。为空表示默认的文本 `SGLangEngine`。设为 `relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine`。 |
| `--rollout-function-path` | str | — | 设为 `relax.engine.rollout.native_generation.generate_rollout`。 |
| `--custom-convert-samples-to-train-data-path` | str | — | 设为 `relax.engine.rollout.native_generation.convert_samples_to_train_data`。 |
| `--custom-reward-post-process-path` | str | — | 设为 `relax.engine.rewards.generative.post_process`。 |
| `--rollout-num-gpus-per-engine` | int | `1` | 每个 diffusion server 占用的 GPU 数。必须 ≥ 1，且 actor world size 必须能被它整除 —— 每个 engine 的 GPU 必须能整齐铺满 FSDP rank，CUDA-IPC 权重同步才成立。 |

::: tip SGLang 静态补丁自检
`SGLangNativeGenerationEngine` 在启动 diffusion server 前检查当前 SGLang 源码是否满足静态 patch 契约：driver sampling 字段可穿过请求链路、driver sigmas 不被二次 shift、Qwen-Image 文本窗口为 512、逐步 seed / rollout variance noise / driver `x_T` 路径存在，并且 `rollout_sde_type="dance"` 可用。静默漏掉任何一项都会让训练用上与记录轨迹不匹配的 log-prob，所以检查失败会直接拒绝启动，并提示重新应用 `docker/patch/sglang/v0.5.15.post1.patch`。
:::

### Artifacts

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--artifact-root` | str | `None` | 媒体与 trajectory sidecar 的根目录。布局：`<root>/<task>/rollout_<id>/group_<idx>.safetensors` 以及 `..._sample_<slot>.png`；评测写在 `<root>/eval/<task>/` 下。必须是 rollout driver **和**每个训练 rank 都能读到的存储 —— sidecar 由单个 driver 进程写出，所有 FSDP rank 都要读回。 |
| `--artifact-retention-rollouts` | int | `2` | 保留多少个 rollout 的 sidecar 与生成媒体。更早的 rollout 目录在每轮 rollout 开始时由 `prune_stale_artifacts` 回收。`<= 0` 表示全部保留，会无上限增长（每组一个 safetensors sidecar，每个候选一张图，每步都有）。 |

### 采样

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--sampling-config` | JSON dict | `None` | 按任务给定的采样几何，用 `json.loads` 解析。由 adapter、rollout driver 和 actor replay 共同读取。 |
| `--generation-seed` | int | `1234` | 基础 seed。见下面 `driver_xt` 的说明。 |

#### `--sampling-config` 的键

| 键 | 默认值 | 读取方 | 说明 |
|---|---|---|---|
| `height` / `width` | `384` / `384` | adapter | 输出分辨率。engine 会在 response 里回显这两个值，它们是还原真实 latent grid 的唯一来源 —— 打包后的序列长度无法还原非正方形网格。 |
| `num_inference_steps` | `12` | adapter | **生成**使用的去噪步数。 |
| `guidance_scale` | `1.0` | adapter | **必须为 `1.0`。** 其他值直接报错：`replay_transition` 只跑一次正向条件前向，CFG rollout 会训练一个与采样时不同的策略。双前向 CFG replay 未实现。 |
| `eta` | `0.7`（rollout 侧） | adapter、actor | SDE 噪声强度，作为 `rollout_noise_level` 发送。 |
| `sigma_max` | `0.99` | actor | replay 在 `sigma == 1` 时使用的 clamp。不要改；见下面 SDE step 0 的说明。 |
| `sde_type` | `"sde"` | adapter、actor | `"sde"` 是 FlowGRPO 的系数 `sqrt(sigma / (1 - sigma)) * eta`；`"dance"` 是 DanceGRPO 的常数 `eta`。 |
| `driver_sigmas` | `true` | adapter | 随请求发送 driver 自己的 FlowMatch sigma 调度，而不是让 server 重新计算。 |
| `driver_xt` | `true` | adapter、driver | 由 driver 提供初始 latent `x_T` 的配方（`initial_noise_group_ids` / `initial_noise_latent_shape` / `initial_noise_seed` / `denoise_seeds`）。开启时每个请求的 engine seed 保持不变，候选多样性来自每个候选的 `sample_id`；设为 `false` 时 driver 退回用 `generation_seed + group_index * group_size + slot` 作为 engine seed。 |
| `init_noise_latent_shape` | 自动推导 | adapter | 覆盖 `x_T` 的 latent 形状；默认为 `[16, *latent_grid(height, width)]`。 |
| `sample_id_mode` | `"metadata"` | driver | 候选稳定 id 的构造方式。`metadata` 用 `prompt:<prompt_id-or-sample_id>:sample:<slot>`（取不到就退回 group index）；`positional` / `group_index` 用 `prompt:<group_index>:sample:<slot>`。也可用环境变量 `RELAX_NATIVE_GENERATION_SAMPLE_ID_MODE` 设置。 |

::: warning `eta` 有两个不同的默认值
adapter 构造 rollout 请求时 `eta` 默认 `0.7`，而 actor replay 默认 `1.0`。请务必在 `--sampling-config` 里显式写 `eta`（两个随仓库发布的配方都写了），否则 replay 的高斯分布不是采样时的那个。
:::

#### 训练的 SDE step

**被训练**的 SDE step 子集由 `relax.models.generative.resolve_sde_indices` 解析。它同时决定两件事：生成时在哪些步注入 SDE 噪声，以及哪些步会被 replay 求梯度。解析优先级：

| 键 | 默认值 | 说明 |
|---|---|---|
| `sde_resample_per_rollout` | `false` | **最高优先级。** 为真且拿得到 `rollout_id` 时，用 `numpy.random.default_rng(rollout_id)` 从池子里抽 `num_sde_steps` 个 —— 可复现，且在所有 engine 和 rank 上一致，无需任何通信。两个随仓库发布的脚本都开了它，所以训练时它会覆盖 `sde_indices`。 |
| `sde_pool` | fraction 窗口 | 上述按轮抽样的候选步集合。 |
| `sde_indices` | `None` | 显式列表，排序去重后原样使用。只在没有按轮重抽样时生效 —— 留出集评测（`rollout_id=None`）正是这种情况，所以它是那条确定性回退路径，保证评测数字跨 step 可比。 |
| `num_sde_steps` | `0` | 训练的步数，在 fraction 窗口内按 stride（`floor(k * len(window) / num_sde)`）取点。 |
| `sde_timestep_fraction` | `[0.0, 1.0]` | 调度的 `[lo, hi]` 归一化窗口。 |

完全没有 SDE 配置时解析结果为空，pipeline 会在打包轨迹时失败。生成始终跑满 `num_inference_steps`，只有解析出来的子集会被 replay 和训练 —— 这正是把 replay 控制在单卡显存预算内的关键。

::: danger `sde_type="sde"` 下拒绝训练 step 0
step 0 处调度的 `sigma == 1`，扩散系数 `sqrt(sigma / (1 - sigma)) * eta` 是奇异的，只靠一个人为的 `sigma_max = 0.99` clamp 才保持有限 —— 而那并不是采样器实际用的值。与 SGLang diffusion engine 实测对比：`std_dev_t` 为 7.0 而非约 2.92，这会翻转 `prev_sample_mean` 中 `sample` 项的符号并把方差放大 2.4 倍；其余每一步都能对齐到 1e-4。因此 `_validate_sde_schedule` 在预检阶段拒绝训练 step 0，并覆盖它可能被训练到的每条路径：显式 `sde_indices`、按轮抽样的 `sde_pool`，以及由 `sde_timestep_fraction` 推导出的窗口。

请使用 `sde_indices [1,3,5]` 搭配 `sde_pool [1,2,3,4,5]`，或让 `sde_timestep_fraction` 从大于 `1/num_inference_steps` 处开始。注意 `sde_timestep_fraction: [0.0, 0.5]` 在 12 步调度下解析为 `[0, 2, 4]`，那**会**训练到 step 0。`sde_type="dance"` 用常数系数、不除以 `1 - sigma`，不受影响。
:::

### FSDP2 训练

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--fsdp-trainable-mode` | `full` \| `lora` | `full` | 必须与 `--lora-rank` 一致：`lora` 要求 `> 0`，`full` 要求 `0`。 |
| `--fsdp-trainable-attr` | str | `transformer` | model bundle 上承载可训练模块的属性名。其余部分全部冻结，且都在 rollout engine 里。 |
| `--fsdp-param-dtype` | `bf16` \| `fp32` | `bf16` | FSDP2 混合精度的参数 dtype（all-gather 出来的计算副本）。 |
| `--fsdp-reduce-dtype` | `bf16` \| `fp32` | `fp32` | 梯度归约 dtype。 |
| `--fsdp-master-dtype` | `fp32` \| `bf16` | 未设置 | 仅**可训练**参数的存储 dtype，即优化器的 master 副本。不设置则沿用加载时的 dtype。LoRA 运行应设为 `fp32`；前向不受影响。 |
| `--no-fsdp-reshard-after-forward` | flag | 默认**开启** reshard | 关闭前向后重新分片（用显存换速度）。没有正向参数 —— `--fsdp-reshard-after-forward` 会是个空操作。 |
| `--fsdp-activation-checkpointing` | flag | `False` | 反向时重算每个 transformer block。基本是必开项：它把 SDE replay 的激活成本压回大约「块输入」量级。 |
| `--fsdp-cpu-offload` | flag | `False` | FSDP2 CPU 参数 offload。见下面的告警。在 `--fsdp-trainable-mode lora` 下被拒绝。 |
| `--fsdp-load-wave-size` | int | `0` | 每次只让 N 个 rank 加载基座模型，波次之间加 barrier，从而限制启动时的宿主机内存峰值与模型目录读竞争。`0` 表示所有 rank 同时加载。 |
| `--fsdp-lr-scheduler` | `constant` \| `linear` \| `cosine` | `constant` | warmup 之后的学习率形状。见下文「学习率」。 |
| `--fsdp-debug-fingerprint` | flag | `False` | 每步在优化器前后记录可训练权重的绝对值求和指纹以检测空转的优化器步，并输出 `[logp parity]` 行（replay 出来的 π_old 锚点对比 engine 自己的采样 log-prob）。默认关闭 —— 它每步都做一次全参数归约加一次宿主机同步，还会强制走显式锚点的 replay 路径。 |

优化器固定为 `torch.optim.AdamW`，没有切换参数。其超参由共享的 `--lr` / `--weight-decay` / `--adam-beta1` / `--adam-beta2` / `--adam-eps` 经 `_adamw_kwargs` 传入。

::: warning `--fsdp-cpu-offload`
开启 FSDP2 CPU offload 后，对被 offload 的 DTensor 做 FSDP 集合通信（`clip_grad_norm_`）会报 *"No backend type associated with device type cpu"*。两个随仓库发布的 Qwen-Image 配方都不开它，而是依赖 `--offload-train` / `--offload-rollout`（Relax 的 colocate 时分共享）加上一个小的 SDE 训练子集。
:::

#### Colocate offload

这些是框架级参数，但生成式路径依赖它们：

| 参数 | 说明 |
|---|---|
| `--colocate` | **必需。** actor 与 rollout 时分共享同一批 GPU。 |
| `--offload-train` | rollout 期间释放 actor 的参数/buffer 设备存储并把 AdamW 状态移到 CPU；训练与权重同步时唤醒。`offload_module_to_cpu` 把 DTensor 的*本地分片存储*resize 到 0 并保留一份 pinned 主机副本 —— 直接改写 `param.data` 在 `fully_shard` 下是空操作，一个字节都释放不掉。 |
| `--offload-rollout` | 训练期间 offload diffusion engine 的模块。权重同步只 onload engine 的 transformer（`tags=[WEIGHTS]`），下一轮 rollout 前再整体重载。`release_memory_occupation` / `resume_memory_occupation` 失败是**致命错误**而不是告警：吞掉失败会让 pipeline 一直驻留显存，colocate 的 actor 在很多步之后 OOM，而现场没有任何线索指回这里。 |

#### 学习率

`lr_at_step` 计算每个优化器步（从 0 开始计数）的学习率：

- warmup 在 `--lr-warmup-iters` 步内线性上升。
- warmup 之后的形状由 `--fsdp-lr-scheduler` 决定；`constant`（默认）保持已验证的行为并忽略衰减设置。
- `linear` / `cosine` 在 `--lr-decay-iters` 内从 `--lr` 衰减到 `--min-lr`。
- **这些单位都是优化器步，不是 rollout** —— actor 每轮 rollout 会走 `--num-updates-per-batch` 个优化器步。
- 步数以 `lr_scheduler_steps` 写入 `trainer_state.json`，resume 时恢复，warmup 不会从零重来。
- Megatron 的 `--lr-decay-style` 在这里无效；与默认的 `constant` 同时出现时预检会告警。

### FlowGRPO

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--num-updates-per-batch` | int | `1` | 每个 rollout batch 的 PPO 优化器更新次数。见下文。 |
| `--advantage-estimator` | str | — | 设为 `grpo`。 |
| `--eps-clip` | float | `0.2` | PPO clip 区间。FlowGRPO 用的值远小于文本 RL —— 两个随仓库发布的配方都用 `1e-4`。 |
| `--eps-clip-high` | float | `None` | 可选的非对称上界。`grpo_clip_loss` 支持它，但 actor 目前只传对称的 `--eps-clip`。 |
| `--rollout-batch-size` | int | — | 每轮 rollout 的 prompt 数（组数）。 |
| `--n-samples-per-prompt` | int | — | 每个 prompt 的候选数，即 GRPO 组大小。advantage 在组内归一化。 |
| `--global-batch-size` | int | — | `rollout_batch_size * n_samples_per_prompt`。这是 rollout batch，在 `--num-updates-per-batch > 1` 时**不是**优化器步边界。 |
| `--micro-batch-size` | int | `1` | 一次前向多少个本 rank 切片内的候选。见下文。 |
| `--clip-grad` | float | `1.0` | 每次优化器步前的梯度范数裁剪。 |
| `--disable-grpo-std-normalization` | flag | 默认**开启**归一化 | 不再用组内 std 去除组中心化后的分量 —— 即 Dr.GRPO 变体。`n_samples_per_prompt == 1` 时 `grpo_std_normalization` 也会被强制关闭。 |
| `--generative-advantage-std-mode` | `group` \| `batch` \| `none` | 旧布尔行为 | 显式指定 prompt 组内中心化之后的 std 除数。`group` 对齐 Relax/Text GRPO，`batch` 对齐参考 diffusion 配方的 global std，`none` 是 Dr.GRPO。LoRA 脚本默认设置为 `group`。 |

::: warning `--num-updates-per-batch` 是切分，不是复用
`_plan_micro_batch_updates` 把本 rank 的本地训练切片划分成 **N 份互不相交、样本数相等**的更新（CountPlanner 风格）。每个样本只参与一次优化器步，不存在跨 mini-epoch 的数据复用。因此优化器步的边界是 `global_batch_size / num_updates_per_batch`，而不是 `global_batch_size`。

π_old 锚点仍然是在任何优化器步之前、为所有 micro-batch 一次性冻结的 —— 这正是第一次更新 ratio 恰为 1 的原因；而后续更新发生在权重已被优化器步移动之后，于是偏离锚点，产生非零的 clipped loss 与真实的 PPO clipping。

`--num-updates-per-batch 1` 时 ratio 恒为 1，配合组中心化的 advantage，clipped loss 按构造为 0，clipping 永远不会触发。两个随仓库发布的配方都用 `2`。

这个划分是硬校验：`N` 必须整除本地样本数，且任何 micro-batch 都不得跨越更新边界，否则 `train()` 直接报错。
:::

::: tip `--micro-batch-size` 与 DP 切片
`hydrate_micro_batches` 在每个 rank 上构造相同顺序的组列表（一个 micro-batch 对应一个组），所以各 rank 的 FSDP 集合通信序列完全一致。当 `n_samples_per_prompt` 能整除 DP world size 时，每个 rank 只 replay 每组的 `[dp_rank::dp_world]` 候选；否则该组退化为每个 rank 全量 replay（结果正确，只是算力冗余）。

`--micro-batch-size` 把这个 per-rank 切片进一步等分成若干块，每块作为一个独立 micro-batch —— 只有块大小能整除切片时才会切分，因为不等长的块会静默改变均值的权重。请**尽量调大**：flops 完全一样，但每多一个 micro-batch 就多一次 FSDP all-gather 和一次 reduce-scatter。全参脚本用 `2`；LoRA 脚本用 `1`，因为 `n=8` 分到 8 个 rank 后切片本来就只有一个候选。
:::

目标函数在 `relax/models/flow_grpo.py`（`flow_sde_transition_std_dev_t`、`dance_sde_transition_std_dev_t`、`flow_sde_transition_std`、`flow_sde_transition_moments`、`flow_sde_log_prob`、`replay_transition_logp`、`grpo_clip_loss`、`normalize_grouped`、`combine_component_advantages`）。它是纯 `torch`、不引入任何框架依赖，由 rollout engine 与 actor replay 共用，所以第一次 on-policy 更新可以逐 bit 对齐。

clipped 目标按 **(sample, step)** 逐元素计算，再在两个轴上做 mean：`_replay_logp` 返回 `[B, S]` 矩阵（每个训练 SDE step 一列）而不是把各步求和，这样某一步漂出 clip 区间不会把其他步的梯度一起清零。

::: danger 没有 KL-to-reference
FlowGRPO 更新里没有 reference model，也没有 KL 项。预检会拒绝任何非零的 `--kl-coef` 或 `--use-kl-loss`，而不是静默忽略。
:::

### 奖励

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--reward-runtime` | `colocate` \| `cpu` \| `remote` | `cpu` | 打分器运行的位置。见下面的说明。 |
| `--reward-scorer-path` | str | `None` | 指向 `GenerativeRewardScorer` 的 dotpath。除非 `reward_runtime=remote` 且给了 endpoint，否则必填。 |
| `--reward-model-path` | str | `None` | 打分器的本地模型目录。 |
| `--reward-required-components` | JSON list | `None` | 必须存在的分量名，例如 `'["pickscore"]'`。缺失或非有限值会让整组失败。 |
| `--reward-component-weights` | JSON dict | `None` | `{component: weight}`，用于把归一化后的分量合成单个 advantage。必选分量没有权重时预检报错。 |
| `--reward-endpoint` | str | `None` | HTTP endpoint。`reward_runtime=remote` 时必填。 |

::: warning `cpu` 和 `colocate` 都是进程内打分
`post_process` 跑在 rollout worker 里，所以只有两种可能：在本进程内打分（`cpu`，禁用 CUDA；或 `colocate`，在 actor 已 offload 时共享 rollout GPU），或者 POST 给外部服务（`remote`）。没有独立的打分器 actor 池 —— 旧的 `dedicated` 只是 `colocate` 的同义词，还会额外打印一条关于它从未拿到的 Ray placement group 的告警，现在已经删除。manager 按打分器配置缓存，所以模型只加载一次，而不是每轮 rollout 重载。

两个随仓库发布的脚本都默认 `--reward-runtime colocate`：PickScore 只需要约 5.6 GB，它运行时 actor 已经 offload，而放在 CPU 上会主导后处理耗时（实测 256 样本下 reward + convert + TransferQueue 约 104 s/rollout）。`reward/score_time` 单独统计打分本身，所以这个选择是可验证的。
:::

#### 内置打分器

| 打分器 | Dotpath | 分量 | 说明 |
|---|---|---|---|
| PickScore | `relax.engine.rewards.pickscore.PickScoreScorer` | `pickscore` | CLIP-H 文图对齐（`required_tracks = ("image",)`）；已验证的 T2I 奖励。 |

只有 PickScore 在已验证的 T2I 闭环中跑过。

### 权重同步

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--weight-sync-mode` | `full` \| `adapter` | `full` | 是推导出来的，不是配置出来的：`--lora-adapter-mode` 蕴含 `adapter`，显式给一个与之矛盾的值是预检错误，而不是被校验器静默覆盖。 |
| `--weight-sync-wire-dtype` | `bf16` | `bf16` | 传输张量的线上 dtype。 |
| `--weight-sync-bucket-size-mb` | int | `512` | 全量张量分桶传输的桶大小。两个随仓库发布的脚本都调到 `2048`。 |

没有同步间隔参数：colocate 事务按构造每步都同步，发生在 `train()` 末尾。

每步 actor 用 DTensor `full_tensor()` 聚合完整 transformer，构建 `FullWeightManifest`（张量数、总字节、`ordered_name_shape_hash`），再逐桶通过 CUDA IPC 发给每个 engine。IPC 拓扑按 CUDA device UUID 匹配而不是 GPU 序号，因为 placement group 会重排 GPU。发送完成后校验 count/bytes 与 manifest 是否一致，rank 0 再交叉校验 `get_weights_checksum`，最后才调用 `commit_weight_version`。任何一步失败都抛 `WeightSyncError`；`update_weights` 会回滚 policy version（重试时重建同一个版本号）、给 `train/weight_sync_failures` +1，然后要么（开了 `--use-fault-tolerance`）恢复 engine，要么在每个 rank 上重新抛出。

::: tip activation checkpointing 的包装层名字
`apply_activation_checkpointing` 会在每个被包装 block 的参数名里插入 `_checkpoint_wrapped_module` 段。`strip_transport_wrappers` 在计算 manifest 与线上名字之前把它去掉。否则 SGLang 的权重加载器会用一句裸 `continue` 跳过这些不认识的名字，**却仍然报告成功** —— 每个 transformer block 的权重都被静默丢弃，而同步显示干净提交。
:::

桶大小决定 RPC 数量，而一次同步的成本几乎全在单次 RPC 开销上：一次 40.9 GB 的 merge 模式同步实测 `[wsync time]` 为 gather 0.5 s / serialize 0.3 s / transport 38.8 s，也就是每个桶约 0.5 s，用来打开一个设备内的 CUDA-IPC handle。512 MB 是 80 个桶，2048 MB 是 20 个。桶在 GPU 上物化，所以调大也会让同步期的瞬时显存多约 1.5 GB。

### 预检校验

`validate_generative_config` 在 `train_backend == "fsdp"` 时于启动阶段运行，收集**所有**违规后抛出单个 `ValueError`。它是轻量导入的（不引 torch、不加载模型），`scripts/training/diffusion/` 下的启动脚本也走同一个 `train.py` 入口，所以错误的生成式配置会在训练服务创建前失败。

**报错：**

- `generation_task` 不在 `GENERATION_TASKS`（`("t2i",)`）内。
- 缺 `model_adapter_path` 或 `model_path`。
- `fsdp_trainable_mode` 既不是 `full` 也不是 `lora`。
- `fsdp_trainable_mode='lora'` 但 `lora_rank <= 0`，或 `lora_rank > 0` 却没有 `--fsdp-trainable-mode lora`。
- `lora` 模式下：`lora_dropout != 0`，或开了 `--fsdp-cpu-offload`。
- `full` 模式下：`weight_sync_mode != 'full'`。
- 用了 `--lora-adapter-mode` 却显式给了矛盾的 `--weight-sync-mode`，或 `weight_sync_mode='adapter'` 却没开 `--lora-adapter-mode`。
- `sde_type='sde'` 下 `sampling_config` 可能训练到 SDE step 0（同时检查 `sde_indices`、`sde_pool` 和推导出的 fraction 窗口）。
- `weight_sync_wire_dtype != 'bf16'`。
- 开了 `fully_async` 或 `hybrid`；或没开 `colocate`。
- `kl_coef != 0` 或开了 `use_kl_loss`。
- `rollout_num_gpus_per_engine < 1`，或 `actor_num_gpus_per_node * actor_num_nodes` 不能被它整除。
- `reward_required_components` 里有条目在 `reward_component_weights` 中没有权重。
- 开了 `use_dynamic_batch_size`，或设了 `max_tokens_per_gpu`（diffusion latent 定形，没有 token 预算分批）。
- 设了 `autoscaler_config`（diffusion engine 没有 elastic router / scale-out API）。
- 设了 `save_hf`（改用离线的 `examples/diffusion/export_checkpoint.py`）。
- 设了 `load_debug_rollout_data`（生成式 actor 从 TransferQueue + 磁盘 sidecar 读 batch）。

**告警**（记录日志，不阻断）：

- `fsdp_trainable_mode='lora'` 但没有 `--fsdp-master-dtype fp32`。
- `async_save`（DCP save 是同步的；不能硬报错，因为 `slime_validate_args` 要求它与 `--rotate-ckpt` 同时出现）。
- 设了 `lr_decay_style` 而 `--fsdp-lr-scheduler` 仍是 `constant`。
- `save_debug_train_data`（生成式 actor 产出的是数值索引 + 磁盘 sidecar，不是 token 序列）。

#### 运行时校验

需要活对象的检查放在运行期：

| 检查 | 位置 |
|---|---|
| 可训练边界（全参：不能有冻结参数；LoRA：adapter 存在且 base 冻结） | `FSDPTrainRayActor._assert_trainable_boundary` |
| `guidance_scale == 1.0` | `QwenImageAdapter.build_rollout_request` |
| response 带有 `trajectory_latents`、`timesteps`、`sde_indices`、`height`、`width` | `QwenImageAdapter.validate_rollout_response` |
| `actor_world == sum(engine_gpu_counts)` | 首次权重同步（`_assert_sync_plan_aligned`） |
| DP rank 之间 micro-batch 数一致 | `_assert_microbatch_count_aligned`（MAX 归约，直接报错而不是死锁） |
| resume 时 adapter contract 匹配（模式、base-model hash、LoRA rank/alpha/dropout/目标模块） | `_assert_resumable` |

### 参考配置

两个配方都在 `scripts/training/diffusion/` 下。下面每个值都是脚本里可用环境变量覆盖的默认值；脚本才是唯一事实来源，这里只是阅读辅助。

#### 已验证的 LoRA 配方

`scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh` 已完成 100 轮 Qwen-Image / PickScore adapter 同步对齐验证（最终 eval 0.8643，参考基线为 0.8620）。

```bash
--resource '{"actor": [1, 8], "rollout": [1, 8]}'
--colocate

# 后端 / 任务 / 钩子
--train-backend fsdp
--generation-task t2i
--model-path ${MODEL_DIR}/Qwen-Image
--model-adapter-path relax.models.qwen_image.adapter.QwenImageAdapter
--rollout-engine-class-path relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine
--rollout-function-path relax.engine.rollout.native_generation.generate_rollout
--custom-convert-samples-to-train-data-path relax.engine.rollout.native_generation.convert_samples_to_train_data
--custom-reward-post-process-path relax.engine.rewards.generative.post_process

# 组几何：32 prompt x 8 候选 = 256 样本的 rollout batch，
# 再切成 2 份互不相交、各 128 样本的优化器更新
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256
--micro-batch-size 1
--rollout-seed 42

# 评测
--eval-prompt-data pickscore ${DATA_DIR}/processed/t2i/pickapic_eval256.jsonl
--eval-interval 10
--n-samples-per-eval-prompt 2

# 采样：生成 12 步，训练其中 3 步，每轮 rollout 从排除奇异 step 0 的池中重抽
--sampling-config '{"height":384,"width":384,"num_inference_steps":12,"guidance_scale":1.0,"eta":0.7,"sde_type":"sde","sde_indices":[3,4,5],"sde_resample_per_rollout":true,"num_sde_steps":3,"sde_pool":[1,2,3,4,5],"driver_xt":true,"sample_id_mode":"metadata"}'
--generation-seed 42

# LoRA
--lora-rank 64
--lora-alpha 128
--lora-dropout 0.0
--lora-adapter-mode          # 脚本默认；设置 ADAPTER_MODE=0 才验证 merge 同步路径

# FSDP2
--fsdp-trainable-mode lora
--fsdp-trainable-attr transformer
--fsdp-param-dtype bf16
--fsdp-reduce-dtype fp32
--fsdp-master-dtype fp32
--fsdp-activation-checkpointing
--fsdp-load-wave-size 2
--offload-train
--offload-rollout

# FlowGRPO + 优化器（默认 LoRA 配方）
--advantage-estimator grpo
--eps-clip 1e-4
--generative-advantage-std-mode group
--num-updates-per-batch 2
--lr 3e-4
--lr-warmup-iters 0
--adam-eps 1e-8
--weight-decay 0.0

# 奖励
--reward-runtime colocate
--reward-scorer-path relax.engine.rewards.pickscore.PickScoreScorer
--reward-model-path ${MODEL_DIR}/PickScore_v1
--reward-required-components '["pickscore"]'
--reward-component-weights '{"pickscore":1.0}'

# 权重同步 / engine 布局
--weight-sync-wire-dtype bf16
--weight-sync-bucket-size-mb 2048
--rollout-num-gpus-per-engine 1

# checkpoint
--save ${EXP_DIR}/runs/qwen-image-t2i-lora/
--save-interval 20
--max-actor-ckpt-to-keep 10
```

脚本启动时还会写出
`ARTIFACT_ROOT/recipes/qwen-image-t2i-lora-*.json`，记录 prompt/eval 集、
DataLoader 顺序开关、采样配置、`advantage_std_mode`、LoRA 同步模式、git
revision 与本次验证使用的 SGLang patch 版本。

#### 全参微调参考

`scripts/training/diffusion/run-qwen-image-t2i-8xgpu.sh` 是记录全参经验取值的地方。它没有复现参考曲线；除非确实需要全参更新，否则优先用 LoRA 脚本。它与上面的差异：

| 设置 | 全参 | 原因 |
|---|---|---|
| `--fsdp-trainable-mode` | `full`（无 `--fsdp-master-dtype`，无 `--lora-*`） | 全参更新。 |
| `--rollout-batch-size` / `--n-samples-per-prompt` / `--global-batch-size` | `8` / `8` / `64` | 没有省下冻结 base 的梯度与 AdamW 动量（约 25 GB/rank），`n` 就受显存限制。降到 GRPO 最低的 2 时 `reward/group_std_mean` 只有约 0.01，信号太弱学不动。 |
| `--micro-batch-size` | `2` | `n=8` 分到 8 个 rank、每组每 rank 只有 1 个候选时切分本来就会跳过；`2` 是「尽量调大」的记录值（实测 `B=1` 时 `mem_peak_gb` 63.6 / 96）。 |
| `sde_indices` | `[1,3,5]` | 池子 `[1,2,3,4,5]` 与按轮重抽样都与 LoRA 配方相同。 |
| `--lr` / `--lr-warmup-iters` / `--adam-eps` | `3e-5` / `20` / `1e-15` | 在留出的 256 prompt PickScore 评测上做过区间标定：`1e-5`+`1e-8` 冻住（129 轮 eval 平坦），`1e-4`+`1e-15` 过冲（第 19 轮 eval −0.06）。极小的 `adam-eps` 恢复了 AdamW 的尺度不变性 —— mean 归约的 flow-SDE log-prob 产生的逐元素梯度约 1e-8，默认的 `1e-8` 会主导分母。 |
| 评测集 / 保留数 | `pickapic_eval64.jsonl` / `--max-actor-ckpt-to-keep 2` | 全参 20B 的 DCP checkpoint（权重 + fp32 AdamW 状态）约 115 GB。 |

## LoRA

设置 `--fsdp-trainable-mode lora` 并同时给 `--lora-rank`。两者必须一致 —— 只给其一时预检会直接拒绝，而不是静默去训练一组错误的参数。这些参数就是 Megatron 文本路径用的同一组 `--lora-*`；FSDP/diffusers 侧的实现在 `relax/backends/fsdp/lora.py`。

```bash
bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
```

### LoRA 换来的是什么

**首先是显存收益，adapter 同步是已验证 rollout 路径。** 冻结 base 省掉了它的梯度与 AdamW 动量 —— 在 20B DiT 上约 25 GB/rank —— 这才是能提高 `--n-samples-per-prompt` 和训练 SDE step 数的原因。它*不会*减少激活。脚本默认 adapter 同步，因此每步只传 LoRA 张量；merge 模式仍然传完整 transformer，主要用于显式验证 transport 本身。

### 两种 rollout 路径

| 模式 | 参数 | 每步同步量 | SGLang 要求 |
|---|---|---|---|
| Adapter | `--lora-adapter-mode`（脚本默认） | 只传 adapter —— 已验证的 rank-64 运行为 1440 个张量、约 377 MB | `docker/patch/sglang/v0.5.15.post1.patch` |
| Merge | `--lora-merge-mode`（`ADAPTER_MODE=0`） | 完整 transformer，`B @ A` 折进每个 base 权重 | `docker/patch/sglang/v0.5.15.post1.patch` |

同一个 engine 上绝不能混用这两种模式。一旦 SGLang 完成 LoRA 转换，DiT 参数会被改名为 `*.base_layer.weight`，而它的权重加载器对不认识的名字用一句裸 `continue` 跳过、**却仍然报告成功** —— 向已 LoRA 化的 engine 做全量同步会静默丢弃每一个张量。因此 adapter 模式压根不做全量同步（LoRA 训练不改 base），补丁也把混用情形变成显式报错。

### 目标模块

不设置时，`--lora-target-modules` 回退到 model adapter 自己的 `lora_target_modules`。`QwenImageAdapter` 提供每个 transformer block 的八个 attention 投影 —— `attn.to_q/to_k/to_v/to_out.0`（图像流）与 `attn.add_q_proj/add_k_proj/add_v_proj/to_add_out`（文本流）。SGLang 的 Qwen-Image DiT 用完全相同的命名，所以这里训练出的 adapter 无需改名即可在那边加载。

名字按模块名**后缀**匹配。

### 关键设置

| 参数 | 取值 | 原因 |
|---|---|---|
| `--fsdp-master-dtype` | `fp32` | adapter 的优化器 master 副本。bf16 master 会把约 1e-4 的相对 RL 更新在尾数上抹掉：grad norm 与 loss 看着都健康，策略却停止移动，症状与学习率过低完全一样。LoRA 下未设置时预检会告警。前向不受影响 —— `--fsdp-param-dtype` 仍决定 all-gather 出来的计算副本。 |
| `--lora-dropout` | `0.0`（强制） | FlowGRPO 的 π_old 锚点是在 `model.train()` 下对活模型做的 replay，随机前向会破坏第一次更新 `ratio == 1` 的不变量，在 `--eps-clip 1e-4` 下几乎全部被 clip。 |
| `--lr` | `3e-4` | 约为全参 `3e-5` 的 10 倍；rank-64 adapter 从 `B = 0` 起步，自由度也少得多。 |
| `--fsdp-cpu-offload` | 被拒绝 | 对 CPU offload 的 adapter DTensor 做梯度裁剪没有可用的通信后端。 |

### Checkpoint 与导出

LoRA checkpoint **只**存 adapter（`ignore_frozen_params`），因此远小于 1 GB，而不是约 115 GB。base 每次 resume 都从 `--model-path` 重新加载，所以 `adapter_contract.json` 记录了 rank、alpha、dropout、目标模块、task type 与 base-model hash —— `_assert_resumable` 在任何一项不匹配时都会拒绝。特别是 alpha 无法从权重反推，取错了会静默地重新缩放策略。

两种导出形态：

```bash
# 把 adapter 折叠进 base -> 可直接加载的 diffusers pipeline
python3 examples/diffusion/export_checkpoint.py --form merged \
  --save-dir ... --task t2i --iteration 100 --output-dir ... \
  --model-adapter-path relax.models.qwen_image.adapter.QwenImageAdapter \
  --base-model-path ${MODEL_DIR}/Qwen-Image --verify

# 可移植的 HF-PEFT adapter 目录
python3 examples/diffusion/export_checkpoint.py --form adapter \
  --save-dir ... --task t2i --iteration 100 --output-dir ... --verify
```

对 LoRA checkpoint 做 `--form merged` 时 `--base-model-path` 是**必填**的：没有 base 就无处可折，导出会写出一个全部由 `lora_A`/`lora_B` 张量构成的「transformer」—— 非空且数值有限，粗略的健全性检查根本发现不了。

## 产出

### Artifacts

写在 `--artifact-root` 下，每轮 rollout 一个目录：

```
${ARTIFACT_ROOT}/t2i/
└── rollout_0000000/
    ├── group_00000000.safetensors          # 整组共享的轨迹 sidecar
    ├── group_00000000_sample_0000.png      # 解码出的候选图片
    └── group_00000000_sample_0001.png
```

评测过程写在 `${ARTIFACT_ROOT}/eval/t2i/` 下，布局相同。

sidecar 以扁平前缀保存三个命名空间：`common.*`（sigmas、SDE 索引、sample 索引、policy version、seed hash）、`conditions.*`（冻结的文本条件）与 `tracks.*`（逐候选 latent），写入顺序固定为 `.tmp → fsync → 原子 rename`。每个候选还在其 `train_metadata` 里带一份小 JSON manifest，记录 policy version、采样指纹与权重 manifest hash。

`--artifact-root` 必须放在每个训练 rank 都能读到的存储上：sidecar 由单个 driver 进程写出，所有 FSDP rank 都要读回。`--artifact-retention-rollouts`（默认 `2`）在每轮 rollout 开始时回收更早的目录；没有它，几百步就会悄悄把卷写满。

### Checkpoint

用 PyTorch Distributed Checkpoint（DCP）保存在 `<--save>/<task>/iter_<n>/` 下：

```
${SAVE}/t2i/iter_0000020/
├── fsdp/                      # DCP 分片（模型 + 优化器）
├── trainer_state.json         # rollout id、policy version、task、adapter、采样指纹、lr_scheduler_steps
├── weight_sync_manifest.json
├── adapter_contract.json
├── rng_rank0.pt ...           # 逐 rank RNG，best-effort
└── COMMITTED                  # 最后写入；resume 唯一信任的标记
${SAVE}/latest_checkpointed_iteration.txt
${SAVE}/t2i/latest_checkpointed_iteration.txt
```

写入顺序固定 —— 分片 → RNG → JSON sidecar → `COMMITTED` → 两级 `latest_checkpointed_iteration.txt` —— 因此中途崩溃不会留下会被 resume 信任的半成品。只有当 MIN all-reduce 确认**每个** rank 都写成功时才会落 `COMMITTED`。

`--rotate-ckpt` / `--max-actor-ckpt-to-keep` 生效；轮转由 rank 0 在写新 checkpoint 前执行，其结果会广播出去，所以轮转失败会在每个 rank 上抛错，而不是把其他 rank 挂在 barrier 上。DCP 写失败会被大声记录并计入 `train/checkpoint_failures`，但不会终止训练。

::: warning resume 需要 `--load`
`_maybe_resume` 读的是 `--load` 而不是 `--save`。两个随仓库发布的脚本都没设置它，所以重启会从 rollout 0 开始，除非你显式传 `--load <与 --save 相同的目录>`。设置之后，resume 会恢复模型、优化器、`policy_version`、`lr_scheduler_steps`（warmup 不会重来）和逐 rank RNG，并以 `latest + 1` 作为起始 rollout id。
:::

把一个已提交的 checkpoint 导出回 HF safetensors：

```bash
python3 examples/diffusion/export_checkpoint.py \
  --save-dir ${SAVE_DIR} \
  --task t2i \
  --iteration 20 \
  --output-dir /path/to/export \
  --model-adapter-path relax.models.qwen_image.adapter.QwenImageAdapter \
  --verify
```

## 监控

actor 在 rank 0 的进程内初始化 tracking adapter，所以常规的 `--use-clearml` / `--use-metrics-service` / TensorBoard 参数都能用。所有序列共用 `rollout/step` 横轴。生成式特有的指标：

| 指标 | 含义 |
|---|---|
| `train/loss`、`train/ratio_{mean,std,min,max}`、`train/clip_fraction`、`train/approx_kl` | FlowGRPO 裁剪目标的健康度。`ratio_min` / `ratio_max` 跨 rank 做 MAX 归约，其余取平均 |
| `train/grad_norm`、`train/lr`、`train/optimizer_steps`、`train/optimizer_skips`、`train/num_microbatches` | 优化器健康度；`skips == num_microbatches` 表示整轮因退化被丢弃 |
| `train/policy_version`、`train/weight_sync_tensors`、`train/weight_sync_gb`、`train/weight_sync_failures` | 权重同步溯源 —— 没有版本序列就无法把奖励回退与一次回滚过的同步关联起来 |
| `train/checkpoint_failures` | 非致命的 DCP 保存失败次数 |
| `reward/<component>_{mean,std,min,max}` | 每个分量的原始奖励（如 `reward/pickscore_mean`） |
| `reward/advantage_{mean,std}`、`reward/num_groups` | 合成后的 advantage 分布 |
| `reward/group_std_mean` | **关键诊断指标** —— 每组奖励 std 的均值 |
| `reward/degenerate` | advantage 方差塌到 `1e-6` 以下时为 1 |
| `reward/score_time`、`reward/score_samples_per_s` | 只统计打分本身，与 `perf/reward_time` 区分开 |
| `perf/mem_{allocated,reserved,peak}_gb` | GPU 显存水位，取最差 rank |
| `perf/train_transitions_per_s`、`perf/train_samples_per_s` | 吞吐。一个 "transition" 是一个候选的一个 SDE step 回放 |

::: tip 没有 MFU / TFLOPs
共享的 `FlopsCounter` 按 LLM `hf_config` 分派，对 DiT 没有解析模型，所以 `log_perf_data_raw` 传的是 `flops_counter=None`。生成式路径改报真正约束它的量：每秒回放的 transition 数与候选数。
:::

## 故障排除

**启动时 `Generative RL config validation failed:`。** 预检一次性收集所有违规项 —— 请通读整个列表。常见原因包括缺 `--model-adapter-path` / `--model-path`、`--fully-async` 忘了关、非零的 `--kl-coef`、`sampling_config` 可能训练到 SDE step 0，或者 actor world size 不能被 `--rollout-num-gpus-per-engine` 整除（每个 engine 的 GPU 必须能整齐铺满 FSDP rank，CUDA-IPC 同步才成立）。

**`reward/group_std_mean` 接近 0 且 `train/loss` 卡在 0。** 组内没有奖励方差，所有 advantage 都约等于 0，优化器步被跳过。请提高 `--n-samples-per-prompt`、检查 prompt 是否足够多样以让打分器区分候选，并确认 `--num-updates-per-batch` ≥ 2。

**训练步 OOM。** 卡上同时有分片参数、梯度、AdamW 状态和 replay 激活。按顺序处理：保持 `--fsdp-activation-checkpointing` 开启、缩小训练的 SDE step 子集（`num_sde_steps` / `sde_pool`）、降低 `--sampling-config` 里的分辨率，然后再减少 `--n-samples-per-prompt` —— 或者改用 LoRA 配方，它省掉了冻结 base 的梯度与 AdamW 动量（约 25 GB/rank）。请保持 `--offload-train` 与 `--offload-rollout` 开启；colocate 依赖 actor 与 engine 轮流占卡。

**第一步之前、启动阶段就 OOM。** 每个 FSDP rank 都要先在 CPU 上物化完整基座模型再分片。调低 `--fsdp-load-wave-size`（两个随仓库发布的脚本都用 `2`）以限制同时加载的 rank 数。

**怀疑优化器步是空操作。** `--fsdp-debug-fingerprint` 会在每步前后打印可训练权重的绝对值求和指纹，并输出一行 `[logp parity]`，把回放出来的锚点与 engine 自己的采样 log-prob 做对比，从而把「step/offload 路径坏了」与「奖励数学坏了」区分开。它默认关闭，因为这是每步一次的全参数归约加宿主机同步。

**任务停了但 GPU 显存仍被占用。** 即使 `ray serve status` 已经为空，也要在重新启动前检查是否有残留的 `sgl_diffusion::scheduler` 进程。

## 致谢

感谢 UniRL 作者提供 Qwen-Image / PickScore 参考 recipe 与指标，用于对齐检查。

## 下一步

- [架构设计](./architecture.md) —— 生成式路径接入的 controller / service / component 模型
- [OOM 排查](./oom-troubleshooting.md) —— 通用的显存压力排查手册
