# Relax

Relax 是一个基于 Ray Serve 的大模型强化学习训练框架，支持 Megatron 训练后端、SGLang 推理引擎、以及 GRPO/PPO/DAPO 等算法族。

**Tech stack**: Python 3.8+ · PyTorch · Megatron · SGLang · Ray / Ray Serve

## Project Structure

```
relax/                   核心框架
├── agentic/             Agentic 层 — Session、pipeline、agent process 与训练导出
├── core/                编排层 — 训练循环、服务基类、全局注册表
├── components/          组件层 — RL 服务组件（Ray Serve Deployment）
├── engine/              引擎层 — Rollout 数据生成、奖励计算、请求路由
├── backends/            后端层 — Megatron 训练后端、SGLang 推理引擎
├── distributed/         分布式层 — Ray 集群管理、分布式 Checkpoint
├── entrypoints/         入口层 — 训练入口脚本
├── models/              模型层 — 模型专属实现与注册
└── utils/               基础设施 — 工具函数、指标监控、多模态处理
tests/                   模块测试与集成回归测试
├── cpu/                 无 GPU 环境执行的测试，内部保留模块层级
├── backends/megatron/   Megatron 后端测试（权重转换等）
├── backends/sglang/     SGLang 后端测试
├── components/          RL 服务组件测试
├── core/                Controller、Service 与注册表测试
├── data/                数据处理与 SFT 数据测试
├── distributed/checkpoint_service/  分布式 Checkpoint 测试
├── distributed/ray/     分布式 / Ray 测试（弹性伸缩等）
├── engine/rewards/      奖励函数测试
├── engine/rollout/      Rollout 引擎测试（预取、数据源等）
├── engine/sft/          SFT 引擎测试
├── entrypoints/         入口行为测试
├── examples/            示例级回归测试
├── integration/         跨模块集成测试
├── models/              模型专属测试
├── tools/               工具与辅助脚本测试
├── utils/               工具函数测试（HTTP、指标、流式数据集等）
└── test_agentic_rollout.py  Agentic runtime 与 Session 测试
docs/                   中英文用户文档
skills/                 仓库开发与运维工作流
docker/                 训练镜像与依赖 patch
examples/                用户级示例（deepeyes、OPD 等）
scripts/                 训练启动脚本 & 模型配置
configs/env.yaml         运行时环境配置
```

修改代码时重点关注：`relax/utils/utils.py`、`relax/components/`、`relax/core/controller.py`、`relax/entrypoints/train.py`。

## Architecture

基于 Ray Serve 的服务编排分为三层：

1. **Controller**（`relax/core/controller.py`）：顶层训练循环、服务编排与健康监控。
2. **Service**（`relax/core/service.py`）：生命周期管理、GPU placement group 与 Ray Serve deployment 封装。
3. **Components**（`relax/components/`）：Actor、Critic、Rollout、Advantages、GenRM、ActorFwd 等具体 RL 服务。

两种执行模式：

- **Colocate（Sync）**：Actor 与 Rollout 分时复用相同 GPU。
- **Fully Async**：各角色使用独立 GPU 集群，通过 TransferQueue 流式传输数据。

关键子系统：

- **Backends**（`relax/backends/`）：Megatron 训练后端（TP/PP/CP/EP）与 SGLang 推理引擎。
- **Engine**（`relax/engine/`）：Rollout 生成、可插拔奖励函数（`engine/rewards/`）、数据过滤与请求路由。
- **Distributed**（`relax/distributed/`）：Ray 集群管理与 DCS（基于 NCCL broadcast 的权重同步）。
- **Algorithm Registry**（`relax/core/registry.py`）：算法名称到组件角色的映射。

## Code Standards

- Ruff 格式化，行宽 119，双引号，`isort` 管理导入并将 `relax` 视为 first-party（配置见 `pyproject.toml`）
- `relax/` 下所有 `.py` 须含版权头：`# Copyright (c) 2026 Relax Authors. All Rights Reserved.`
- pre-commit 对 `transfer_queue/` 排除大部分检查
- 日志统一用 `relax.utils.logging_utils.get_logger(__name__)`，禁止 `print` / `logging.getLogger`
- 显式类型注解；禁止通配导入（`from x import *`）；重型可选依赖放函数内导入
- 热路径禁止 GPU-CPU 同步（`.item()`、`.tolist()`、`print(tensor)`）
- 组合优于继承，层级 ≤ 2；偏好委托

## Naming Conventions

| 类型        | 模式                    | 示例                                  |
| ----------- | ----------------------- | ------------------------------------- |
| Service 类  | `@serve.deployment`     | `Actor`, `Rollout`, `Critic`, `GenRM` |
| 后端 Actor  | `XxxTrainRayActor`      | `MegatronTrainRayActor`               |
| Actor 组    | `RayTrainGroup`         | 管理一组训练 Ray Actor                |
| Manager     | `XxxManager`            | `RolloutManager`, `GenRMManager`      |
| Reward 函数 | `engine/rewards/xxx.py` | `deepscaler.py`, `math_utils.py`      |
| 数据源      | `XxxDataSource`         | `RolloutDataSource`                   |

## Hard Rules

CRITICAL: 以下规则不可违反。

- 禁止通配导入、硬编码密钥/路径/端点
- 禁止跳过 pre-commit hooks
- 禁止猜测集群配置或重建 CUDA/驱动栈
- 调试时直接修改 `scripts/training/` 下的训练脚本来调整参数，再通过 `scripts/entrypoint/ray-job.sh` 提交
- 禁止做与目标无关的代码改动，如 code format 等
- 集成测试需要多节点 GPU 硬件——必须显式说明跳过原因
- 修改前先读相关文件；提交前跑 `pre-commit run --all-files`
- 遵循最小变更原则：只碰需求直接涉及的文件和行

## Ask First

修改以下内容前必须先征求确认：

- `relax/utils/arguments.py` 或 `relax/backends/megatron/arguments.py` 的参数解析
- 新增依赖
- Controller / Service / Launcher 逻辑
- 删除或重命名公开 API

不确定时，留 `TODO(agent)` 注释并在回复中说明约束。

## Development Workflow

```bash
pip install -e .                    # 以开发模式安装
pip install -r requirements.txt
pre-commit run --all-files           # lint + format（等同 make format）
pytest tests/                        # 测试
pytest tests/<test_file>.py::<test_name>  # 运行单个测试
make test                           # 等同 pytest tests/
make docs-dev                       # 启动 VitePress 文档开发服务器
```

- 提交遵循 Conventional Commits（`feat:`, `fix:`, `docs:`），详见 `git-commit` skill
- 默认仅创建本地 commit；用户授权 push 或创建 PR 时，遵循 `relax-github-workflow` skill
- 测试命名：`test_<module>_<behavior>()`，GPU 测试用 `@pytest.mark.skipif` 优雅跳过

## Test Scope & Placement

新增测试前先评估必要性，不要求每个函数、类或实现改动都配套新增单测。

- 先检查现有覆盖，并在变更说明中简述新增测试要防止的真实失败、覆盖缺口和 CI 成本；没有明确增量价值时不新增。
- 优先覆盖用户可见行为、真实缺陷回归、数值与梯度正确性，以及分布式通信、状态恢复和资源清理等高风险路径。
- 纯搬迁、重命名、文案或无行为变化的重构，优先复用现有回归与静态检查；避免重复实现逻辑、只断言 mock 调用或源码文本的测试。
- 优先扩展已有用例，以最小场景集覆盖不同失败机制；避免重复覆盖和无必要的参数组合。新增慢测试或多进程测试时，说明为什么现有轻量测试不足，并限制进程数和超时。
- 测试按实际运行需求归类：纯 CPU 测试放入 `tests/cpu/<module>/`，需要真实 GPU 的测试保留在对应的 `tests/<module>/`。CPU/GPU 混合文件应拆开，不在 workflow 维护文件清单或逐用例排除。
- CPU 分类必须检查间接调用和 fixture 的设备依赖，包括 CUDA tensor 搬运、设备探测和可见设备环境变量；不能仅凭测试名称或使用 mock 判断，也不能为迁入 CPU 而新增 skip/mock 掩盖真实 GPU 依赖。
- CPU 套件须支持隐藏 GPU 后执行 `pytest tests/cpu`；GPU 套件由 `pytest tests/ --ignore=tests/cpu` 执行。依赖不全导致的 skip 不算回归通过，应明确验证限制。

## Distributed Code Rules

作用域：`relax/backends/**`、`relax/distributed/ray/**`、`relax/distributed/checkpoint_service/**`

- 禁止在模块级创建全局进程组；始终显式传 `process_group`
- `dist.get_rank(group)` 而非 `dist.get_rank()`
- all-reduce 所有 rank 必须调用；broadcast 显式 `src`；barrier 仅用于调试
- 调试环境变量：`TORCH_DISTRIBUTED_DEBUG=DETAIL`, `NCCL_DEBUG=INFO`, `CUDA_LAUNCH_BLOCKING=1`

## Configuration & Launching

训练入口为 `python relax/entrypoints/train.py [args]`，通过 `relax/utils/arguments.py` 的命令行参数配置（扩展 Megatron-LM 解析器）。

## Domain experts & skills

Fire the appropriate **expert subagent** or **load a skill** based on what you're working on. Experts are read-only consultants with deep domain knowledge; skills are step-by-step implementation guides.

| Working on...                                | Fire subagent      | Load skill              |
| -------------------------------------------- | ------------------ | ----------------------- |
| Megatron backend (TP/PP/CP/EP)               | `megatron-expert`  |                         |
| RL algorithms (GRPO/PPO/DAPO/GSPO/SAPO)      | `algorithm-expert` |                         |
| Task Launch & service deployment             | `launcher-expert`  |                         |
| Ray framework (Core, Serve, Jobs, placement) | `ray-expert`       |                         |
| Development & code changes                   | `ray-expert`       | `dev`                   |
| Agentic rollout integration, runtime & hangs | `agentic-expert`   | `agentic-rollout`       |
| Distributed training hang debugging          | `ray-expert`       | `debug-hang`            |
| Code review of git changes                   | --                 | `code-review`           |
| Creating new skills                          | --                 | `creating-skills`       |
| Git commits                                  | --                 | `git-commit`            |
| GitHub contribution, PRs & review follow-up  | --                 | `relax-github-workflow` |
| GitHub required checks & CI operations       | --                 | `relax-github-ci`       |

**How to fire an expert**: task(subagent_type="megatron-expert", load_skills=\[\], run_in_background=true, prompt="...")
