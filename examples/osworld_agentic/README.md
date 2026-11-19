# OSWorld + Nex agentic 训练

本 recipe 使用 Qwen3-VL-8B-Thinking、Relax resident agentic rollout 和 Nex 桌面沙箱进行 GRPO 训练。它依赖外部 Nex 服务及可用的 OSWorld 桌面模板；只克隆仓库不能直接运行。

## 目录与配置

- [VCS](#vcs)：可选的追加式上下文与批量截图裁剪，连续多轮回答按片段训练，默认关闭。
- `env.example.sh`：公开的部署配置模板，不包含实际服务地址或密钥。
- `env.local.sh`：用户复制模板后填写的私有配置，Git 忽略，不能提交。
- `scripts/run-qwen3vl-8B-osworld-8xgpu.sh`：正式训练参数与提交入口；修改训练参数请改这里。
- `run_agent_app.sh`：Relax 为每个 session 调用的 agent 入口，不是独立训练入口。
- `app/`：桌面环境、模型调用适配与奖励；`app/osworld_config.yaml` 控制交互轮数等。
- `scripts/prepare_data.py`、`train_tasks.txt`：数据转换与固定 12 任务清单。

训练脚本和 agent 入口直接读取同目录的 `env.local.sh`，无需额外配置文件。启动时传入的三个根目录优先。

## 准备环境

1. 按 [Relax 安装文档](../../docs/zh/guide/installation.md)准备训练镜像、Megatron、SGLang、Ray 和模型权重。脚本通过 `ray status` 检测现有集群，并用 Ray API 识别 8 或 16 张 GPU；没有集群时通过 `local.sh` 启动单机 Ray。显存需求随轨迹变化。脚本固定 `MEGATRON=/root/Megatron-LM`，其他镜像需修改该行。所有 worker 必须能访问相同路径、相同版本的代码与依赖。

2. 准备 OSWorld fork。下面固定到本 recipe 检查的版本：

   ```bash
   git clone https://github.com/fxsc03/OSWorld-main-nex.git /data/OSWorld-main-nex
   git -C /data/OSWorld-main-nex checkout 50b4a85778cfc13ce00e3bf63a3cd08bd260fb96
   ```

   将 `DATA_DIR` 设置为所有节点同路径可访问的持久共享目录。首次运行训练脚本会在 `${DATA_DIR}/osworld-agent-venv` 创建 Agent 环境并安装依赖，完成后写入标记；后续启动不再安装。提交训练前，脚本将持久环境打包为可复用归档，避免跨节点逐个读取小文件，再解包到每个在线 Ray 节点的 `/tmp/relax-osworld-agent-envs/<版本>/`，所有节点准备完成后才提交任务；每个 Agent 使用节点本地 Python，不从共享盘加载 venv。环境未变化时复用本地副本，Pod 重启导致副本丢失时自动重新复制。手动更新共享环境依赖后，执行 `touch "${DATA_DIR}/osworld-agent-venv/.relax-agent-ready"`，使下次启动创建新版本副本。环境复用训练镜像已有的 PyTorch 包，其他依赖安装在 venv 中；各节点需使用兼容的 Python 和训练镜像。需要更换依赖时，准备新环境并用 `OSWORLD_AGENT_PYTHON` 显式指定其 Python。宿主环境安装不能替代沙箱内的依赖：Nex 模板仍需提供桌面应用、LibreOffice/UNO、MCP 工具及任务所需文件。请按该 fork 的 Nex 部署说明验证模板、网络和任务 reset/evaluator。

3. 在 recipe 目录复制配置，仅首次执行，不要覆盖已有私有文件：

   ```bash
   cd examples/osworld_agentic
   cp -n env.example.sh env.local.sh
   chmod 600 env.local.sh
   ```

   填写 `OSWORLD_REPO`、`NEX_API_BASE`、`NEX_WORKSPACE_ID`、`NEX_API_KEY` 和 `NEX_TEMPLATE`（或 `NEX_TEMPLATE_ID`）。`MCP_SRC_ROOT` 是包含 `mcp/` 的仓库根目录。外部 Nex client 含部署相关默认值，因此 agent 入口要求显式提供 Nex 配置，不依赖这些默认值。

   `OSWORLD_AGENT_PYTHON` 默认无需填写。若显式设置，脚本直接使用该已准备好的环境，不自动安装或复制；启动时传入的值优先于 `env.local.sh`。不要将需要跨任务复用的环境放在 `/tmp`。

   私有文件必须在所有 agent worker 的相同路径可读。若使用 Ray 上传代码，被忽略的文件不会随代码上传，需另外安全部署；也可预先在 worker 环境配置。不要通过 runtime-env JSON 或命令行传递密钥，相关参数可能进入 Jobs 元数据与日志。

4. 正式脚本默认开启 ClearML。按自己的部署配置训练节点的 ClearML（例如 `clearml-init`）；模板不携带任何账户、密钥或内部端点。

## 启动

配置好 `env.local.sh` 后，在 Relax 根目录运行：

```bash
MODEL_DIR=/data/models \
DATA_DIR=/data/rl_data \
SAVE_DIR=/data/outputs \
bash examples/osworld_agentic/scripts/run-qwen3vl-8B-osworld-8xgpu.sh
```

- 模型应位于 `${MODEL_DIR}/Qwen3-VL-8B-Thinking`。
- 数据读取 `${DATA_DIR}/osworld/train.parquet`；不存在时从外部仓库的 `evaluation_examples/examples` 和固定清单生成。清单包含未知任务或为空会报错。已有 parquet 会复用，修改清单后请明确重新生成，避免继续读旧数据。
- 自动生成的 train/eval 使用同一清单；这不是独立验证集，默认也未配置周期评估。
- 输出位于 `${SAVE_DIR}/qwen3vl-8b-osworld-8xgpu/`。

## 当前正式参数

| 配置                           | 值                                                                               |
| ------------------------------ | -------------------------------------------------------------------------------- |
| 初始模型                       | Qwen3-VL-8B-Thinking                                                             |
| 算法                           | GRPO，自定义轨迹级 advantage，TIS                                                |
| 每轮采样                       | 12 组 × 每组 8 条 = 96 条轨迹                                                    |
| 更新次数                       | 每 rollout 2 次，共 200 rollouts                                                 |
| 学习率                         | 5e-6，constant                                                                   |
| KL                             | k3，系数 0.001，ref 每 rollout 更新                                              |
| PPO clip                       | 下/上界均 0.2                                                                    |
| 训练并行                       | TP4 / PP1 / CP1，8 卡 DP2、16 卡 DP4                                             |
| 单卡训练 / log-prob token 预算 | 均为 24576；不是模型上下文长度                                                   |
| Rollout 上下文 / response 上限 | 16384 / 8192 tokens                                                              |
| SGLang                         | 每引擎 2 GPU，static memory fraction 0.7，consistent hashing                     |
| 显存策略                       | 冻结 vision trunk；selective recompute `core_attn`；optimizer CPU offload 未开启 |
| 沙箱                           | prelaunch 开启；active 与预启动共享 12 个 group permits，最多 96 个 session      |
| Dynamic filter                 | 关闭                                                                             |
| Agent                          | 最多 30 轮，图像历史 2 张，session timeout 21600 秒                              |
| Checkpoint                     | 关闭；每次启动从初始模型开始，不恢复优化器状态                                   |

Actor 与 rollout colocate，共用检测到的 8 或 16 张物理 GPU。session 并发限制不等于 Nex 服务端实例计数保证：创建重试、服务端回收延迟仍需结合 Nex 配额监控。

## 奖励、轨迹与指标

每条轨迹的 shaped reward 为：`(成功 ? 1 : -1) - 0.05 × T/30 - 0.2 × I(T>=30)`，其中成功判定为 evaluator outcome ≥ 0.5，T 为轨迹轮数。每组 8 条轨迹计算并标准化 advantage，再分配到该轨迹的训练导出记录；不是按导出记录数量组成 8 条一组。

96 条逻辑轨迹可以导出远多于 96 条训练记录。`rollout/raw_reward` 为对齐交接实现的**按导出记录数加权的 shaped reward**，不是轨迹成功率。判断任务能力应同时看成功率、轮数、异常比例；不能将 raw reward 图标成 success rate。全成功/全失败组的策略 advantage 被置零，但启用的 KL 项仍可能产生梯度。

基础设施与执行异常按 Session 终态统计：`agentic/sessions_completed`、`agentic/sessions_failed`、`agentic/sessions_cancelled`，以及 `agentic/session_error_rate = failed / (completed + failed)`。统计按采集区间累积，train/eval 隔离；选择性取消不进入异常率分母。首个模型请求前失败也计入，即使整组被丢弃、没有训练 Sample；原任务 evaluator 判负仍属于正常完成的 Session。丢弃日志保留 task 进程的 `stop_reason` 和 `last_error`，不从成功导出的 Sample 推算基础设施异常率。

baseline 的 `max_image_history_length=2` 限制单次模型上下文的图像历史，不表示一条轨迹只有两张截图。VCS 模式改用独立的 K/Δ 窗口，raw reward 仍按原始调用数加权。

## 三项可选优化

VCS 减少重复导出的训练 context，图片缓存减少冻结视觉编码器的重复计算，提前传输降低 rollout 侧 CPU 图片内存峰值。它们不会减少配置中的任务数或每个任务的独立尝试次数。

### 开关与环境传递

| 配置                                               | 默认值      | 用途                                                         |
| -------------------------------------------------- | ----------- | ------------------------------------------------------------ |
| `RELAX_VCS`                                        | `0`         | 开启 VCS；未显式设置时，OSWorld 沿用 YAML 的 `context_mode`  |
| `RELAX_VCS_KEEP_IMAGES` / `RELAX_VCS_PRUNE_IMAGES` | `2` / `2`   | VCS 的 K / Δ，均须大于零                                     |
| `RELAX_TRAIN_VISION_CACHE`                         | `0`         | 开启已适配模型的冻结视觉特征缓存                             |
| `RELAX_TRAIN_VISION_CACHE_MAX_BYTES`               | `268435456` | 每个视觉执行 rank 的缓存上限，默认 256 MiB                   |
| `RELAX_TRANSFER_BATCH_GROUPS`                      | `0`         | 非 fully-async 模式下每批传输的完整 Group 数；`0` 沿用原水位 |
| `RELAX_MM_RELEASE_AFTER_TRANSFER`                  | `0`         | 传输成功后释放 rollout Sample 的像素张量                     |
| `RELAX_MM_RELEASE_TRIM` / `RELAX_MM_RELEASE_LOG`   | `1` / `1`   | 释放后回收空闲页，并记录张量字节与 RSS 变化                  |
| `RELAX_MM_PIXEL_DTYPE`                             | `fp32`      | 可设 `bf16` / `fp16`，减少预处理像素的存储与传输体积         |
| `RELAX_MM_IMAGE_POOL`                              | `0`         | 同一 SessionForest 内相同图片只保留一份像素，导出时重组      |

当前正式脚本仍使用 **Qwen3-VL-8B-Thinking**。VCS 和提前传输可分别启用；**图片缓存当前只适配 dense `Qwen35VLModelProvider`，不能直接在此 Qwen3-VL recipe 上打开**。使用已适配的 Qwen3.5 训练配置、并设置 `--freeze-vision-model` 后，才将对应开关改为 `1`。

编辑启动脚本的 `RUNTIME_ENV_JSON` Python 片段，在现有 `env["RELAX_PROPAGATE_ENV_VARS"] = ...` 之前加入下面配置。现有转发语句会将变量传递给后续 Serve、rollout、训练 worker 和 agent；仅在提交端 shell 中 export 不保证远端可见。

```python
env.update({
    "RELAX_VCS": "1",
    "RELAX_VCS_KEEP_IMAGES": "2",
    "RELAX_VCS_PRUNE_IMAGES": "2",
    "RELAX_TRAIN_VISION_CACHE": "0",  # 适配的 Qwen3.5 配置才改为 1
    "RELAX_TRAIN_VISION_CACHE_MAX_BYTES": "268435456",
    "RELAX_TRANSFER_BATCH_GROUPS": "1",
    "RELAX_MM_RELEASE_AFTER_TRANSFER": "1",
    "RELAX_MM_PIXEL_DTYPE": "bf16",
})
# 保留脚本原有的 RELAX_PROPAGATE_ENV_VARS 构造语句。
```

变量在 worker 导入模块时生效，修改后需重启训练。缓存预算 `2147483648` 表示每 rank 2 GiB；增大缓存会占用训练显存，不能代替合理的动态微批次 token 预算。

### VCS

VCS 保留文字历史，批量裁剪旧截图，将连续回答导出成一个训练片段。baseline 仍使用上游两张图的窗口和逐调用导出；VCS 不改变任务执行、evaluator、reward shaping 或 GRPO 的逻辑轨迹分组。

也可在 `app/osworld_config.yaml` 配置；显式环境变量优先，`RELAX_VCS=0` 强制关闭：

```yaml
context_mode: vcs
context_policy: baseline
vcs_keep_images: 2
vcs_prune_images: 2
```

下一次请求的图片数 **大于 K+Δ** 时，先保存已提交片段，再移除最旧的 Δ 张图片。不是立即降到 K 张；每轮只加入一张图时，裁剪后为 K+1 张。计数按图片引用次数，同一截图再次观察也计一次。VCS 不能与上游截图跳过策略组合，实际窗口由 VCS 控制，`max_image_history_length` 只影响上游临时 prompt。

以 K=2、Δ=2、连续 8 轮、工具不变且无重试为例，`It` 为该轮截图、`At` 为真实回答；文字历史始终保留：

| 推理轮 | 请求中的图片         | 新回答归属     |
| ------ | -------------------- | -------------- |
| 1–4    | I1，逐轮增加至 I1–I4 | A1–A4 属于段 0 |
| 5–6    | I3–I5，增加至 I3–I6  | A5–A6 属于段 1 |
| 7–8    | I5–I7，增加至 I5–I8  | A7–A8 属于段 2 |

最终导出 3 条训练 context，分别监督 A1–A4、A5–A6、A7–A8。新段中的旧回答作为 prompt，`loss_mask=0`；本段新增回答正常监督。只在裁剪图片、上下文变化或任务结束等边界导出片段末态，原本应该训练的新回答不会因合段丢失。context 条目减少比例不能直接换算成训练加速比例。

`RelaxHybridAgent.call_llm` 在请求前调用 `VisualContext.prepare`，真实回复仍写入 `turn_records`。`agent.py` 显式导出片段的 messages、tools、chat_template_kwargs；Forest 使用已提交状态的真实生成 token 和 rollout logprob，不重新 tokenize 旧回答后伪造概率。新段在 system 后加入 `[Visual context segment N]` 标记，防止意外匹配旧分支。

所有片段保留同一 Session / `Sample.index`，共享轨迹级 advantage，沿用自定义 advantage、dynamic batching 和逻辑 index 下的有效 token 归一化。`OSWORLD_PER_TRAJ_NORM=0` 保持不变；旧配置 `=1` 时按原始调用数缩放，而非片段数。reward 只写入第一条 export，`rollout/raw_reward` 仍按原始调用数加权。

system、tools、模板参数或历史修正发生变化时先保存旧段，再用真实新上下文开段。上游解析失败仍对同一 prompt 最多重采样五次：非空失败回答保留为监督目标，但不加入重试输入；空回复或未收到结果的传输尝试不凭空生成目标。任务结束、步数耗尽及已处理的 predict/context-length 错误会保存非空尾段；取消和无法写出结果的异常遵循原 Session 生命周期，不保证导出，也不新增 exactly-once 协议。

通用实现是 `relax.utils.visual_context.VisualContext`，不绑定 OSWorld。其他 CUA adapter 按 Session 使用 `full_history` 模式，每轮调用 `begin_turn`，真实请求前 `prepare`、回复后 `commit(record, usage)`、结束时 `finish`，导出 `records`。应保留真实工具返回并关闭上游重复截图折叠；OSWorld 的兼容适配器使用 `latest_observation` 模式。具体训练导出约定见 [Agentic rollout](../../docs/zh/guide/agentic-rollout.md)。

`metadata.vcs_segment` 记录分段原因、调用索引、环境轮号、解析尝试、新回答数、图片及 token 数。`osworld/vcs_segments`、`osworld/vcs_pruned_images`、`osworld/input_tokens`、`osworld/answer_tokens` 走现有指标接口；精确训练 mask/token/probability 以 Forest 生成的 Sample/dump 为准。VCS 只限制图片，长文字历史仍可能触及上下文上限。

### 训练图片缓存

当前边界是 dense `Qwen35VLModelProvider` 的 **`vision_model.decoder` 冻结输出**。patch、position、rotary 准备与 merger 仍走原模型；未设置 `--freeze-vision-projection` 时，merger 继续正常前向与反向。Bridge 内部复用 `Qwen3VLModel` 类不表示其他 Qwen3-VL 配置也受支持。

图片身份使用通用 `Sample.session_id`（或 metadata 中的 Session ID）、CPU processor 像素 SHA-256、dtype 和 `(T,H,W)`，通过原 `multimodal_train_inputs` 传输。LRU 按字节预算淘汰，单图特征超过预算时正常计算但不存储。缺少 Session 身份、视频或混合视频输入走原路径；不改变任务数、轨迹数、图片数或 loss。

缓存建立在实际执行视觉计算的 rank 上，保存该接口的完整 pre-merger 特征 `[patches, 1, vision_hidden]`，不是语言模型 SP 分片。TP 成员以持久 Gloo 控制组交换身份和命中状态，任一成员 miss 时所有成员对同一图片执行原 decoder；特征不通过控制组传输，DP 副本不共享缓存。

TP/PP/CP/DP、LM sequence parallel 和原视觉通信保持不变。CP 使用实时 `model.pg_collection.cp`，metadata 按 Bridge 的连续图片分配对齐，保留原 CP gather；PP/VPP 无视觉模块的 stage 不建缓存或控制组。权重切换、offload、加载 state dict，以及冻结参数/buffer 的版本、设备、dtype 或 autocast 状态变化会使缓存失效。

开启时检查冻结与确定性条件。当前拒绝其他 provider、非空 deepstack、可训练缓存边界、非零 dropout、视觉 FP8/FP4、视觉 CUDA graph、视觉 SP 分片、分离式 encoder/decoder，以及无法对应源图片的自定义 CP padding；没有固定并行度白名单。

每次清空时，`Training vision cache: ... stats=...` 输出原因、rank 坐标及本区间统计并重置计数：

| 字段                                                | 含义                                                               |
| --------------------------------------------------- | ------------------------------------------------------------------ |
| `calls` / `cache_calls`                             | 入口调用数 / 身份信息完整的调用数                                  |
| `image_refs`                                        | 身份完整的图片引用次数，重复图片逐次计数                           |
| `hits` / `misses`                                   | 跨调用命中 / 未命中引用；TP 协同 miss 按 miss 计                   |
| `batch_dedup_images` / `encoded_images`             | 未命中图片的批内去重数 / 实际编码的去重后图片数                    |
| `patch_refs` / `encoded_patches`                    | 原本需要处理 / 实际编码的 patch 数，均包含绕过缓存的调用           |
| `cache_hit_patches` / `batch_dedup_patches`         | 跨调用命中 / 批内去重省下的 patch 数                               |
| `bypass_calls` / `bypass_patches`                   | 身份缺失或视频等绕过缓存的调用 / patch 数，不推算其图片数          |
| `evictions` / `oversized_skips`                     | 容量淘汰 / 单条超过预算而未存储次数；主动清空不算淘汰              |
| `tp_forced_misses`                                  | 本 rank 命中但其他 TP 成员 miss 而共同重算的图片引用数             |
| `entries` / `bytes` / `peak_bytes` / `budget_bytes` | 条目数 / 当前占用 / 区间峰值 / 预算                                |
| `image_hit_ratio` / `image_skip_ratio`              | 完整身份图片的命中率 / 同时计入批内去重的计算跳过比例              |
| `patch_skip_ratio`                                  | `1 - encoded_patches / patch_refs`，包括绕过缓存的调用；零分母记 0 |

例如空缓存输入 `[A,A,B]`，得到 `image_refs=3, misses=3, batch_dedup_images=1, encoded_images=2`，三次 miss 不等于编码三次。不同尺寸图片优先比较 patch 指标；TP 复制记录不能相加，每个 DP 副本选一个 TP 成员。Ray 日志去重可能折叠不同 DP 的行，完整比较读取 worker 原始日志。命中率或 patch 跳过比例不代表整步加速比例；这些统计不新增 GPU 计时同步或集体通信。

### 提前传输与图片内存

`RELAX_TRANSFER_BATCH_GROUPS=1` 允许非 fully-async 模式在一个**完整且已评分的 Group** 就绪时写入 TransferQueue，无需等满 rollout。Group 仍含同一任务的多次独立尝试及全部导出样本，不提前训练未完成的 Group，不改变 advantage、batch 或更新频率。

`RELAX_MM_RELEASE_AFTER_TRANSFER=1` 等待 `async_put` 成功后，把 rollout 本地 Sample 的像素张量替换为零行占位，保留 dtype、宽度与图片网格；训练读取队列中的完整图片。写入失败或取消时保留数据。设置 `--save-debug-rollout-data` 或 `--dump-details` 时也保留原像素，保证调试文件可回放。默认执行垃圾回收和 `malloc_trim`，`mm_release samples=... freed=... rss=... returned=...` 记录张量字节和实际 RSS 变化；`freed` 不等于操作系统即时回收字节。

`RELAX_MM_IMAGE_POOL=1` 另可在 SessionForest 内按图片指纹复用像素存储，导出时还原原张量。`RELAX_MM_PIXEL_DTYPE` 控制存储/传输精度，不是视觉特征缓存。当前 colocate 仍等整批 rollout 完成后训练，队列和运行中 Sessions 也占内存，提前传输不等于提前执行 GPU 更新。

### 验证范围

CPU 回归使用 mock 模型/环境和轻量视觉 fixture，不启动 Nex 或 GPU 训练：

```bash
python3 -m pytest -q \
  tests/cpu/examples/test_osworld_visual_context.py \
  tests/examples/test_osworld_agentic_advantage.py \
  tests/examples/test_osworld_agentic_monitoring.py \
  tests/cpu/test_agentic_rollout.py \
  tests/utils/test_visual_context.py \
  tests/utils/test_train_vision_cache.py \
  tests/cpu/agentic/test_cua_transfer_memory.py
```

覆盖真实 Forest 匹配、导出、mask 与训练转换，以及缓存命中/去重、隔离、淘汰、失效、merger/语言层梯度和传输打包。缓存控制组测试覆盖 TP=1/2/4；CP metadata 覆盖 CP=1/2/4、空分配和 LM SP 开关；PP 覆盖无视觉模块的 stage。CPU 结果不等于实际模型吞吐或 GPU 并行验证。

固定 fork 的真实 `predict`/动作解析回归需在 Agent 环境中设置 `OSWORLD_REPO` 指向已准备的外部仓库；未配置时该用例明确跳过。它读取上述固定提交的代码，以 HTTP 替身验证上下文超限和 Session 丢弃只请求一次，并保留普通解析重试，不启动模型服务或 Nex。

已有 Qwen3.5-9B MobileGym 实验在 TP4/PP1/CP1/DP2、LM SP 开启时完成过 18 次更新，观察到缓存命中和批内去重；不代表本 Qwen3-VL recipe 启用了缓存。PP>1、CP>1、动态 CP 等其余组合尚无 GPU 端到端验证。性能比较需固定模型、轨迹、硬件和预算，分别开关功能，计入缓存构建、rollout 和训练全周期；短实验不能证明独立加速比或最终成功率无损。

## 日志与隐私

- 提交日志：`examples/osworld_agentic/log/<experiment>-<timestamp>.log`。
- Agent 日志：`examples/osworld_agentic/log/agent/<run-tag>/`。
- 训练导出：`${SAVE_DIR}/<experiment>/rollout_result/<run-tag>/train/`。
- 轨迹奖励：同一 run 目录的 `reward_dump.jsonl`。
- TensorBoard：`${SAVE_DIR}/<experiment>/tensorboard_log/<run-tag>/`；ClearML 实验名带启动时间。

Git 忽略私有配置、常见日志/截图/缓存目录，但忽略规则不是数据脱敏机制。桌面截图、任务内容、agent 日志及导出 JSON 可能包含敏感信息，发布前应单独审查；不要强制添加私有文件。加载私有配置时禁用 shell tracing，仍不应把整个私有环境打印到日志。

本 recipe 的审查不代表外部 OSWorld fork、沙箱镜像或历史 Git 提交已完成隐私审计。尤其外部 fork 含部署相关默认端点，公开分发该依赖前应单独审查。

## CHANGELOG

以下按实验启动时间升序记录，时区为 UTC+8。日期后的时间对应实验名中的启动时间；“更新”指优化器更新，编号从 0 开始。只记录可由保留的提交参数和日志确认的结果，脚本修改不算已验证实验。

### 2026-10-11 18:33 — 初始集成实验

- **配置**：Qwen3-VL-8B-Thinking，12 任务 × 8 条轨迹；`num_rollout=10`、每轮 3 次更新；LR `1e-6` / cosine；KL 系数 `0.02`；训练预算和 rollout context 均为 32K；full/uniform recompute，optimizer CPU offload 开启。
- **结果**：日志记录 9 次更新，随后 SGLang engine 在 flush cache 时连接拒绝，任务失败，未完成计划。这里的 10 rollouts 最多对应 30 次更新；这个短配置不满足后续长期训练需求。
- **后续**：延长至 200 rollouts，并调整 offload/recompute。Agent 按逻辑 session 管理沙箱、按模型可见上下文导出训练记录；96 条轨迹可以对应 1400 多条训练记录，不代表同等数量的沙箱。

### 2026-10-11 22:58 — 200 rollouts，关闭 offload、减少重计算

- **相对上次**：`num_rollout: 10 → 200`；关闭 optimizer CPU offload；full recompute 改为 selective；训练预算与 rollout context 暂保留 32K。
- **结果**：首次训练发生 CUDA OOM，没有完成优化器更新。
- **后续**：降低单卡 token 预算，补齐长度配置之间的约束。

### 2026-10-12 01:15 — 训练预算降至 16K

- **相对上次**：`max_tokens_per_gpu: 32768 → 16384`，rollout context 仍为 32768。
- **结果**：启动参数校验失败：`max_tokens_per_gpu × CP` 小于 rollout context；没有进入训练。

### 2026-10-12 01:16 — Rollout context 降至 16K

- **相对上次**：`rollout_max_context_len: 32768 → 16384`。
- **结果**：再次在参数校验阶段失败：prompt 上限仍为 16384，与 context 相等，未给生成 token 留空间；没有进入训练。

### 2026-10-12 01:21 — 修正 prompt/context 约束

- **相对上次**：prompt 上限改为 16383，保持 16K context 和 16K 单卡训练预算。
- **结果**：完成 1 次更新后再次 OOM，尚不能稳定运行。
- **后续**：增加 MLP 重计算以降低训练峰值显存。

### 2026-10-12 02:01 — 增加 MLP 重计算

- **相对上次**：selective recompute 显式使用 `core_attn mlp`。
- **结果**：Actor 初始化阶段遇到 `uncorrectable ECC error`，没有进入训练。这次失败不是训练 loss 或奖励导致的。

### 2026-10-12 02:07 — ECC 报错后重新提交

- **相对上次**：保留 16K 预算、16K context、`core_attn mlp` selective recompute 和关闭 optimizer CPU offload 的组合。
- **结果**：完成 57 次更新（19 rollouts），最大 grad norm 2.854678，之后主动停止以修正奖励逻辑；不是跑满 200 rollouts。
- **问题**：当时的奖励口径尚未完整对齐交接版，不能将该曲线当成交接实验的复现结果。

### 2026-10-12 15:42 — 恢复交接版奖励口径

- **相对上次**：恢复轨迹 outcome shaping、8 条轨迹组内 advantage 标准化，以及按导出记录数加权的 `rollout/raw_reward`。保留 LR `1e-6` / cosine、每轮 3 次更新、16K 训练预算。
- **结果**：完成 21 次更新（7 rollouts），最大 grad norm 4.100425；随后主动停止，继续对齐学习率和训练配置。
- **口径**：shaped reward 为 `(成功 ? 1 : -1) - 0.05 × T/30 - 0.2 × I(T>=30)`。成功率另按 session 去重统计，不能和 `rollout/raw_reward` 混用。

### 2026-10-12 21:02 — LR 5e-6 / constant，32K，冻结 vision

- **相对上次**：LR `1e-6 → 5e-6`，cosine → constant；训练 token 预算 16K → 32K；冻结 vision trunk。保留每轮 3 次更新、KL `0.02`、reference 每 rollout 更新、selective recompute；未开启 sandbox prelaunch。
- **结果**：完成 147 次更新（49 rollouts）后主动停止，进入预启动实验。出现 4 次 `grad_norm > 10`：更新 11、35、71、110，均是所在 rollout 的第 3 次更新；最大 grad norm 491186.094，最大未乘系数的 `train/kl_loss` 14710.166。
- **发现**：大梯度尖峰与重复更新后的 k3 KL 异常值相关。grad norm 是裁剪前值，`clip_grad=1.0` 不代表该指标必须小于 1。

### 2026-10-13 22:15 — 开启 sandbox prelaunch

- **相对上次**：开启 `agentic-prelaunch`，active 和预启动共享 12 个 group permits，最多 96 个 session；其余保留 32K、每轮 3 次更新、KL `0.02`。
- **结果**：停止前日志记录 14 次更新。第 3 次更新再次出现尖峰：grad norm 189.610、未乘系数的 KL loss 2.846，KL 项约占该点 total loss 的 96.5%。预启动没有消除这个训练稳定性问题。
- **后续**：将 KL 系数降至 `0.001`，每轮更新次数减至 2。没有逐 token 或分离 PG/KL 梯度证据，暂不把 KL 异常归因到具体 token；也未得到控制其他变量后的 prelaunch 独立加速比。

### 2026-10-14 00:58 — KL 0.001、每轮 2 次更新、24K，开启 dynamic filter

- **相对上次**：KL `0.02 → 0.001`；每轮更新 `3 → 2`，每次更新的逻辑轨迹数 `32 → 48`；训练预算 `32K → 24K`；开启基于轨迹 outcome 的 dynamic filter。保留 LR `5e-6`、prelaunch、冻结 vision 和每 rollout 更新 reference。
- **结果**：完成 2 次更新，最大 grad norm 1.159800，未出现大于 10 的尖峰；随后按要求主动停止并关闭 dynamic filter。窗口过短，不能据此判断长期稳定性或过滤收益。

### 2026-10-14 01:48 — 关闭 dynamic filter，开始持续训练

- **相对上次**：仅关闭 dynamic filter；保留 KL `0.001`、每轮 2 次更新、24K、LR `5e-6` / constant、prelaunch 和冻结 vision。
- **结果快照：2026-10-14 13:20:39**：完成 27 rollouts、54 次更新（0–53）；grad norm 最小/平均/最大为 **0.561670 / 0.954595 / 1.337857**，`grad_norm > 10` 为 **0 次**；最大未乘系数的 KL loss **0.003474**。该窗口内原来的大幅梯度尖峰未再出现。
- **曲线**：rollout 0 → 26，raw reward `-0.403194 → 0.144336`，session 成功率 `48.96% → 80.21%`；最新 actor training MFU 约 `74.39%`。首尾两个点不代表单调上涨或独立评测成绩，actor MFU 也不代表整个 rollout 周期的利用率。
- **结论范围**：KL 系数、更新次数和 token 预算共同变化，不能把稳定性改善单独归因于 KL 系数；实验仍在运行，尚未完成 200 rollouts。

### 2026-10-14 — 当前实验期间的代码与文档整理，未重启

- **脚本修改，未实跑**：训练/log-prob token 上限改为 48K；当前运行仍是 24K。按 `create-training-script` 整理数组、直接启动和自动入口识别；固定 LR `5e-6`、200 rollouts、每轮 2 次更新，移除间接变量 `OSWORLD_UPDATES_PER_ROLLOUT`。保留默认阻塞提交、不保存 checkpoint 的要求。
- **监控与 dump**：区分 session 与导出记录，补齐 session/task/turn/export 和解析重试标识；修正 raw reward 图的名称与纵轴；正常完成时关闭 metrics adapters。这些改动不构成一次新的对照实验。
- **时间审计，12:15:20 快照**：24 个 `perf/step_time` 合计 10小时21分18.736秒，平均 **1553.281 秒/rollout**；加计时前初始化 3分57.812秒及尚未上报的尾段 1分27.839秒，与 Job 总时长 10小时26分44.387秒对齐。该指标每 rollout 一个点，包含两次更新；rollout 1–23 的累计时间与日志跨度仅差 0.458 秒。
- **开源整理**：提供 `env.example.sh`，直接读取忽略的 `env.local.sh`，删除多余 `env.sh` 及其忽略例外；更新中英文文档，缺失/空任务清单明确报错。未公开内部端点、密钥或私有路径。
- **验证结果**：提交前全部 pre-commit 通过，相关测试 61 passed、1 个 opt-in live Ray Serve 测试 skipped。隔离环境验证过脚本提交参数；48K 尚未完成真实 GPU 训练验证。共享前缀训练、广播和 GPU→CPU 拷贝仅做分析，没有实施新的优化算法。
