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
├── gpux4/               需要 2-4 张 GPU 的测试
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
- 调试时直接修改 `scripts/training/` 下的训练脚本来调整参数，启动入口遵循下方 Configuration & Launching 规则
- 禁止做与目标无关的代码改动，如 code format 等
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
make docs-dev                       # 启动 VitePress 文档开发服务器
```

- 提交遵循 Conventional Commits（`feat:`, `fix:`, `docs:`），详见 `git-commit` skill
- 默认仅创建本地 commit；用户授权 push 或创建 PR 时，遵循 `relax-github-workflow` skill
- 测试要求与执行入口统一见下方 Testing。

## Testing

新增、修改或审查测试前，必须应用本节和 [测试质量准则](skills/code-review/references/test-quality-checklist.md)。以下要求同时适用于人工和 coding agent。

### 准入与有效性

- 验证改动不等于新增永久测试。先查已有覆盖；仅为真实 bug、可推导的独立性质或必要对外契约增加用例，并说明现有测试为何不足、维护与 CI 成本是否值得。
- 每个永久用例必须回答“哪个具体的生产代码错误会使它失败”。测试数量、断言数量、覆盖率和全绿都不能单独证明价值。
- 执行实际产品实现，以结果、状态、输出、必要交互或真实错误为证据。mock 只隔离外部或昂贵边界，不实现被声称验证的计算、恢复、调度或同步能力。
- 期望来自独立参考、手算、外部规范或已确认契约；不得同步复制当前产品算法、复用可能出错的关键 helper，或把实际输出抄为答案。固定历史版本的兼容性参考须注明来源与边界，不能冒充算法正确性证明。
- 不收录脚本 dry-run 后逐项核对默认参数、GPU 数、环境变量、路径或日志文本的配置镜像测试，也不以源码字符串、AST 抽取、补丁拼装、私有名字/布局快照替代行为验证。真实解析器、源码转换器或补丁工具自身的输入输出契约除外。
- 按独立故障模式选择用例和参数行，不机械穷举配置。优先保护数值与梯度、分布式通信、状态恢复、并发和资源清理等真实风险。

### 验证与维护

- bug 回归优先扩展已有用例；条件允许时确认旧实现因目标 bug 失败、新实现通过。需要时在隔离副本中注入具体错误验证断言；未实际运行的旧版本对照或 mutation 不得声称通过。
- 竞态使用事件、受控交错或真实进程协议触发；慢测试和多进程测试需要合理的成本、超时与清理边界。不用不确定等待或无有效断言的循环制造覆盖。
- 一次性 smoke、探索实验、原理演示和文案检查不默认进入永久测试库。记录实际版本、命令、结果、跳过原因与能力边界；硬件或依赖不足时明确未验证，不改用源码/AST 检查冒充运行。
- 删除或合并前，说明失去什么保护、哪个保留用例覆盖同一故障，或旧契约为何失效。混合文件按用例处理；不能因命中某个静态模式就整文件删除。
- 不为变绿或提速而删除有效断言、mock 掉被测能力或无依据放宽容差。无行为变化的搬迁、重命名和重构优先复用现有回归与静态检查。

### 分类与运行

- 测试命名为 `test_<module>_<behavior>()`。真实 GPU、多卡或多节点测试须声明所需资源，并在资源不足时用 `@pytest.mark.skipif` 等方式说明跳过原因；集成测试按实际依赖判断，不一概视为多节点 GPU 测试。
- 测试按实际运行需求归类：纯 CPU 测试放入 `tests/cpu/<module>/`，需要真实 GPU 的测试保留在对应的 `tests/<module>/`，需要 2–4 卡的测试放入 `tests/gpux4/`。CPU/GPU 混合文件应拆开，不在 workflow 维护文件清单或逐用例排除。
- CPU 分类检查真实调用及间接 fixture 的设备依赖，包括 CUDA tensor 搬运、设备探测和可见设备环境变量；不以名称或 mock 判断，也不为迁入 CPU 新增 skip/mock 掩盖真实 GPU 依赖。
- CPU 套件须支持隐藏 GPU 后执行 `pytest tests/cpu`。单卡 CI 运行 `pytest tests/ --ignore=tests/cpu --ignore=tests/gpux4`，多卡 CI 运行 `pytest tests/gpux4`；CI 另外执行的示例测试也纳入审查。依赖不全导致的 skip 不算回归通过。

按改动涉及的行为选择检查，不默认执行整个 GPU 套件：

```bash
pytest tests/<path>/test_<module>.py::test_<behavior>  # 定向验证
CUDA_VISIBLE_DEVICES="" pytest tests/cpu examples/chunkflow/test/cpu
pytest tests/ examples/chunkflow/test/gpu --ignore=tests/cpu --ignore=tests/gpux4
pytest tests/gpux4
```

`pytest tests/`（`make test`）覆盖不同资源需求的套件，只在环境满足对应要求时运行。

## Distributed Code Rules

作用域：`relax/backends/**`、`relax/distributed/ray/**`、`relax/distributed/checkpoint_service/**`

- 禁止在模块级创建全局进程组；始终显式传 `process_group`
- `dist.get_rank(group)` 而非 `dist.get_rank()`
- all-reduce 所有 rank 必须调用；broadcast 显式 `src`；barrier 仅用于调试
- 调试环境变量：`TORCH_DISTRIBUTED_DEBUG=DETAIL`, `NCCL_DEBUG=INFO`, `CUDA_LAUNCH_BLOCKING=1`

## Configuration & Launching

训练入口为 `python relax/entrypoints/train.py [args]`，通过 `relax/utils/arguments.py` 的命令行参数配置（扩展 Megatron-LM 解析器）。

单机启动训练时先检查本地 Ray 是否运行，未运行则直接 `bash <训练脚本路径>`（多数脚本会自动 source `scripts/entrypoint/local.sh`，缺少该逻辑时先 source 此入口），不要仅凭 `HOST_IP` / `MASTER_ADDR` 环境变量推断已有远程集群或要求用户提供 Ray 地址，只有明确使用已有集群时才走 `ray-job.sh`。

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
