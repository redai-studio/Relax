# Code-generation sandbox grader — Draft demo

原始 0/1 判题和数据准备复用 [本目录 baseline](BASELINE.md)。先按该 recipe 准备模型与数据。

## 执行流程

`examples/gagar/run_code_generate_baseline.sh` 保留 baseline 的 SandboxFusion HTTP 判题方式。`run_code_generate.sh` 在同一 recipe 上启用 sandbox 模式：每份 Python 答案先在独立 Apptainer 实例里运行原 SandboxFusion 判题器，再由另一个实例里的 mini-swe-agent 比较本组通过的答案。

链路：`group_rm → 原始 0/1 判题 → 冻结 code/test_output → agentic grader → weights → compute_group_rewards → Relax 组内减均值`。

- 使用现有 `--group-rm` 和本目录 `group_rm.reward_group`；共享 `core/grs.py` 与 GAGAR 算法。两种任务分别加载各自目录的 根目录 `grader.yaml` 的 system / instance template；SWE 继续用 bridge，代码生成使用 grader 自己的 sandbox。
- 代码生成使用根目录 `grader.yaml` 中的英文代码生成配置：评价效率、简洁清晰程度；必要时读代码、构造输入并运行。直接返回每个通过候选的 `[0,1]` 质量权重和依据，允许 0、并列，也不要求权重和为 1。不使用 SWE 的五项 rubric、分档或 rank-to-weight。
- instance prompt 提供题目、候选 ID、原始 outcome 和证据读取方式。只提供代码和判题结果，不向 grader 提供 think 或完整 submission；原始回答保留在宿主审计文件。代码和判题结果可分页读取，代码挂在 `/candidates/<rollout_n>/solution.py`，实验写在 `/workspace`。
- grader 的 bash 只接收 `command`，直接在自身容器里执行；SWE bridge 的 bash 仍要求 `rollout_n`。GenRM 使用 Relax 原有 `/generate`，没有新增模型服务。
- 全通过或全失败组沿用 GAGAR 公式原有行为，最终中心化优势为 0。失败答案不进入质量评分；grader 不改写原始 outcome。

### 代码职责

- `group_rm.py`：Relax group RM 入口，准备整组输入、监督 worker，处理取消、质量评分回退与 reward 记录。
- `sandbox/worker.py`：独立 worker 内的原始判题、grader 容器编排、固定 GenRM 代理、实例租约和清理；也保留 `--cleanup-owner` 定向回收入口。
- `sandbox/apptainer.py`：实例启停、tracked child 和 subreaper。
- `grader.py`：容器内 bash 执行器，以及本任务证据范围、评分函数与共享 grader 工具的组合。
- `scoring.py`：代码生成质量权重的校验；根目录 `grader.yaml` 提供 system / instance prompt。

### 环境与镜像

构建、启动和调试 sandbox 只在分配的训练节点进行；开发机只做代码与 mock 测试。宿主安装 Apptainer，并创建本任务自己的 mini-swe-agent venv。无需在宿主另起 SandboxFusion 服务：

```bash
uv venv /local/path/code-grader-venv --python 3.12
source /local/path/code-grader-venv/bin/activate
uv pip install mini-swe-agent==2.4.1
export MINI_SWE_AGENT_VENV=/local/path/code-grader-venv
```

镜像内分别安装 SandboxFusion Python 判题环境和 mini-swe-agent。下面命令在 Relax 仓库根目录执行；`BUILD_DIR`、`GAGAR_SANDBOX_IMAGE` 由运行环境指定。

```bash
export BUILD_DIR=/path/to/grader-build
export GAGAR_SANDBOX_IMAGE=/path/to/grader.sif
mkdir -p "${BUILD_DIR}"
git clone https://github.com/bytedance/SandboxFusion.git "${BUILD_DIR}/SandboxFusion"
git -C "${BUILD_DIR}/SandboxFusion" checkout 469de0845cb7d665bc68044d541aa3aa28c5b30a
cp examples/gagar/app/code_generate/sandbox/grader.def examples/gagar/app/code_generate/sandbox/sandboxfusion.yaml "${BUILD_DIR}/"
(cd "${BUILD_DIR}" && apptainer build "${GAGAR_SANDBOX_IMAGE}" grader.def)
```

只有 Python 执行器；`sandbox_fusion.py` 调用原 `evaluate_single`，保持 final-answer 提取、provided_data 和 6 秒判题超时，仅把 Python runtime 从 conda 定位切换成镜像内 venv。禁止配置外部 CPU/GPU runner，否则会退出。

### 启动

沿用数据准备与模型配置。数据转换现在额外保留 `metadata.task`（原题目与 starter code），避免把 actor 的 chat template 带入 grader；旧数据仍兼容原 prompt，重新生成后才使用这个字段。下面是 split GenRM 模式：训练使用 8 卡 actor（DP4、CP1、TP2、PP1）；推理使用 4 卡 rollout 和 4 卡 GenRM，两者分别运行两个 TP2 引擎。训练与推理阶段分时复用 8 张 GPU；默认 grader 是 Qwen3.8-27B，context 为 65536，单次回复预算为 16384，只接收候选代码和判题证据，不接收候选的思考过程。

grader 自身启用 thinking、`reasoning_effort=xhigh`。历史思考与工具动作一起保留在 assistant `content` 的 `<think>...</think>` 中；`preserve_thinking=false` 避免 Qwen 模板再次包裹。GenRM 消息仍只有 `role/content`，不修改框架 schema。采用 [Qwen 官方 thinking 采样建议](https://huggingface.co/Qwen/Qwen3.8-27B#api-usage)：temperature=1.0、top_p=0.95、top_k=20；min_p=0、presence_penalty=0、repetition_penalty=1 沿用 SGLang 默认值。64k/16k 是本 recipe 的预算选择。客户端与 sandbox 代理的单次请求超时均为 600 秒，整组 grader 仍受原有 1500 秒 agent 预算和 1800 秒进程上限约束，超时沿用回退与清理逻辑。

```bash
export MINI_SWE_AGENT_VENV=/path/to/mini-swe-agent-venv
export GAGAR_MODEL_PATH="${MODEL_DIR}/Qwen3.8-27B"
export GAGAR_SANDBOX_TMPDIR=/path/to/local-scratch
export GAGAR_AUDIT_DIR=/path/to/persistent-grader-audit
RAY_NO_WAIT=1 bash scripts/entrypoint/ray-job.sh examples/gagar/run_code_generate.sh
```

`MODEL_DIR`、`DATA_DIR`、`SAVE_DIR` 与 baseline 相同；`SANDBOX_URL` 在该模式不需要。配置关闭 GRPO 标准差归一化，保留组内减均值。训练用 `score`，通过率用 `test_reward`，另记录 `quality_weight` 与 `grader_fallback`。

### 隔离、失败和回收

判题和 grader 都启用独立 PID/IPC/网络空间、只读输入挂载、私有临时文件系统；不挂宿主工作目录、home、数据盘或 GPU。instance 由宿主身份管理，执行器与 grader 降到 UID/GID 65534、移除 capabilities。宿主需要支持这些 Apptainer 选项；启动失败不会降级到宿主 bash。

grader 只有一个当前组的 Unix socket 可访问模型；代理固定转发消息到 GenRM，不接受目标 URL、命令或任意 HTTP 路径。容器内 bash 无外网。命令有时间、进程数、地址空间、输出文件大小限制。这是共享宿主内核的容器隔离，不等同于独立虚拟机；limits 也不是整组 cgroup 内存配额。

每组独立保存 `input.json`、`outcomes.json`、`trajectory.json`、`result.json` 与 `worker.log`。原始判题失败直接报错；只有质量 grader 失败才回退到原始 outcome，且标记 `grader_fallback=1`。

实例由本目录 `sandbox/apptainer.py` 管理，保留 owner 命名、tracked child、subreaper、优先 graceful stop 与停止校验。它不导入 SWE server、SIF cache 或 SWE 数据依赖。每次 start 前记录实例租约，正常退出删除；取消或异常后父进程再接管、重试清理。若平台调用持续失败，保留 `instances/` 租约和 scratch 并报错，不把清理失败伪装成成功评分。恢复后可在同一节点用记录中的 owner PID 定向重试，不触碰其他组或 SWE 实例：

```bash
"${MINI_SWE_AGENT_VENV}/bin/python" -m examples.gagar.app.code_generate.sandbox.worker \
  --output /path/to/group-audit/result.json --cleanup-owner OWNER_PID
```

部署到多节点时，各 reward worker 节点都必须具备相同的镜像、venv 和代码路径。

统一配置位于 GAGAR 根目录的 `grader.yaml`，本任务选择 `tasks.code_generate`。修改 system/instance template、工具超时和推理预算都在该文件中完成，原始 rollout 与判题配置不变。
