---
name: create-training-script
description: 新增 Relax 训练启动脚本或模型、数据集、卡数、训练模式变体时使用。覆盖 scripts/training 和 examples 下的脚本位置、模板选择、MODEL_DIR/DATA_DIR/SAVE_DIR、可恢复 checkpoint、ClearML 实验命名和 Ray 非阻塞提交。
---

# 新增训练脚本

从最接近目标的现有脚本复制修改，重点落实下面的路径、命名、保存和提交约定。旧脚本中的写法不一定符合这些约定，不要原样继承。

## 1. 放在哪里、参考谁

- 通用训练配置放 `scripts/training/<category>/`：文本 RL 用 `text/`，多模态 RL 用 `multimodal/`，监督微调用 `sft/`，另有 `genrm/`、`diffusion/`、`hpc/`。
- **属于某个 example 的训练脚本就放该 example 自己的 `scripts/` 下**，例如 `examples/nemo_gym_agentic/scripts/run-<recipe>-<model>-8xgpu.sh`；无需再放一份到 `scripts/training/`。保留 example 已有的数据转换、agent client 等辅助入口。
- 用 `rg --files scripts/training scripts/models examples/<example>` 找模板，优先匹配模型架构、训练任务、硬件、共卡/异步模式，再考虑卡数。先读模板、模型配置及其 source 的入口。
- 简单共卡 / 异步参考 `scripts/training/text/run-qwen3-4B-8xgpu.sh` 和同目录的 `run-qwen3-4B-8xgpu-async.sh`；SFT、多模态、diffusion 从各自目录选，不套文本 RL 参数。
- 沿用邻近命名，如 `run-<model>-<size>-<N>xgpu[-<recipe>][-<mode>].sh`。文件名卡数应对应实际物理卡数，NPU / KLX 选同硬件模板。

## 2. 路径和实验名：固定实验，区分每次运行

数据与 checkpoint **只使用 `MODEL_DIR`、`DATA_DIR`、`SAVE_DIR` 三个根目录**，不要新增 `EXP_DIR`、`ARTIFACT_ROOT` 等中间根目录。`SCRIPT_DIR` / `RELAX_ROOT` 仅用于定位仓库代码。目录默认值优先沿用已确认的可访问位置，未知时要求运行者提供，不猜个人路径或挂载点。

下面是变量与保存配置的示例；模型、数据集、保存频率按目标任务替换：

```bash
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/dapo-math}"
EXP_NAME="${EXP_NAME:-qwen3-4b-dapo-8xgpu-colocate}"
now=$(date "+%Y%m%d-%H%M%S")

: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B"
   --ref-load "${MODEL_DIR}/Qwen3-4B"
   --megatron-to-hf-mode bridge
   --load "${SAVE_DIR}/${EXP_NAME}"
   --save "${SAVE_DIR}/${EXP_NAME}"
   --save-interval 50
   --max-actor-ckpt-to-keep 2
)
# 数据参数放入对应 ROLLOUT_ARGS / SFT_ARGS：
# --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
```

- `MODEL_DIR` 放初始模型，`DATA_DIR` 放数据，`SAVE_DIR` 放持久化训练 checkpoint；多节点时确认相应节点可访问。保存目录不要指向模型源目录、临时目录或 Ray 上传的临时工作目录。
- `PROJECT_NAME` 是 ClearML 的任务分组，如 `Relax/nemo-gym/calendar`；`EXP_NAME` 是稳定实验标识，描述模型、数据/recipe、卡数和模式。两者均允许环境变量覆盖，`EXP_NAME` 用适合作为目录名的字符串。
- **`EXP_NAME`、`SAVE_DIR`、`--load`、`--save` 均不得包含每次启动新生成的 `${now}`。** 恢复同一实验时保持 `SAVE_DIR` 和 `EXP_NAME` 不变；新实验则显式更换 `EXP_NAME` 或 `SAVE_DIR`，避免误恢复旧实验。
- `${now}` 只用于单次运行的 ClearML task name、日志文件名和需要唯一性的 submission ID。不要写成 `EXP_NAME=model-${now}` 再拼保存目录，也不要在恢复时自动寻找“最新时间戳目录”。
- 对上述 Megatron 模板，首次启动时 load 目录不存在或为空会从 HF 初始化；已有有效训练 checkpoint 时从稳定 load 目录恢复。不要为了“首次运行”永久删掉 `--load`，也不要把 HF 权重目录当作完整训练恢复目录。其他后端先核对各自恢复实现。
- `--save`、`--save-interval`、`--max-actor-ckpt-to-keep` 一起显式配置。`50` / `2` 是示例：频率按运行长度与保存开销选择，保留数用正整数以限制磁盘占用；短验证也要考虑能否实际触发保存。间隔单位按训练循环核对，勿误写为秒或 microbatch 数。
- 恢复完整训练时，不要从模板误带 `--finetune`、`--no-load-optim` 等重置训练状态的选项；LoRA、PPO、FSDP 按对应模板核对需要保存和恢复的状态。

## 3. ClearML 和日志必须有明确命名

默认启用 ClearML 和 metrics service，凭证使用现有环境配置。参数数组可沿用模板的 `WANDB_ARGS` 名称，但实际参数应包含：

```bash
WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)
```

最终命令传入 `"${WANDB_ARGS[@]}"`。日志使用 `mkdir -p "${SCRIPT_DIR}/log"` 和 `tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"`，保留 `set -o pipefail`。ClearML task 和日志可以每次启动新建；**训练 resume 由稳定 checkpoint 路径决定**。当前 ClearML adapter 不会自动续写上一次 task，不要承诺恢复 checkpoint 就会续用同一 ClearML task。

## 4. Ray 提交默认非阻塞

- 脚本设置 `export RAY_NO_WAIT="${RAY_NO_WAIT-1}"`，最终提交保留 `ray job submit ${RAY_NO_WAIT:+--no-wait}`，使 AI 提交后能继续查看日志。调试命令显式带 `RAY_NO_WAIT=1`。
- `${RAY_NO_WAIT:+--no-wait}` 按“非空”判断，`RAY_NO_WAIT=0` **仍然启用** no-wait；用户需要等待时显式传空值 `RAY_NO_WAIT=`。此处默认表达式使用 `${RAY_NO_WAIT-1}`，不能用会覆盖空值的 `${RAY_NO_WAIT:-1}`。
- 保留 `--runtime-env-json="${RUNTIME_ENV_JSON}"` 和 `${WORKING_DIR:+--working-dir "${WORKING_DIR}"}`。新增 worker 环境变量时确认已进入 runtime env，不能只在提交机 export。
- 使用已确认的 Ray Jobs HTTP 地址，不能把 GCS 地址直接当作 Jobs 地址。本地模板的 loopback dashboard 地址只适用于相应本地提交场景。
- 保留提交返回的 job ID，随后用 `ray job logs --address="<Jobs HTTP 地址>" <job-id>` 检查训练进展。no-wait 下 `tee` 只保存提交阶段输出，**提交命令成功不等于训练成功**。

## 5. 其余结构和交付检查

- 保留 shebang、版权头、用途和运行示例。用 `BASH_SOURCE[0]` 定位代码；仅当 `RELAX_ENTRYPOINT_MODE` 未设置时 source 对应入口。example 的相对路径要重新核对，不能照搬 `scripts/training` 模板的层级。
- Megatron 架构参数 source `scripts/models/`；训练超参数直接写当前脚本的 `CKPT_ARGS`、`ROLLOUT_ARGS` / `SFT_ARGS`、算法、优化器、性能、SGLang、监控等数组，最终全部用 `"${XXX_ARGS[@]}"` 传入。example 可通过已有辅助入口调用训练，实际参数仍在新脚本中清楚列出。
- `--resource` 是 `[服务数量, 该角色总卡数]`，当前服务数量要求为 1；共卡共享卡不重复计数，全异步 / hybrid 的 actor、rollout 分别占卡。核对其他角色、并行度、每 engine 卡数、batch 和长度，不只修改文件名。
- 核对数据字段和奖励实现；多模态保留媒体映射；SFT 使用 `--loss-type sft`；diffusion 按其 FSDP 与专用引擎模板配置。不确定的参数查源码，不编造 flag。
- 用 `bash -n` 检查新增 shell 文件，核对引用、数组展开、路径和 JSON；不要通过 source 训练入口做静态检查。特别检查“两次启动只改变 now，load/save 路径仍完全相同”，以及新实验不会复用旧实验目录。
- 交付脚本位置、运行所需的三个目录、`PROJECT_NAME` / `EXP_NAME`、保存/保留策略、恢复方法和验证结果。提交 commit 前执行 `pre-commit run --all-files`。

只要求写脚本时完成静态校验即可；要求实际训练验证时按 `skills/dev/SKILL.md`，通过 `scripts/entrypoint/ray-job.sh` 提交，复用已有授权和集群信息。未运行 GPU / 多节点训练时说明原因。
