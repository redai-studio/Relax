# 如何贡献

感谢你对 Relax 项目的关注！通过阅读本指南，你可以快速了解如何参与到 Relax 开源项目的贡献。

::: tip 从哪里开始

如果你想参与 Relax，但还没有确定要做什么，可以从以下入口寻找感兴趣的任务：

- [第二期 Hackathon](https://github.com/redai-studio/Relax/issues/321)：查看本期任务、参与规则和认领方式，选择感兴趣的方向，单人或组队参与。
- [Good first issue](https://github.com/redai-studio/Relax/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22)：适合初次参与项目的贡献者，可以从这里寻找入门任务。

动手前，请先阅读任务说明和已有讨论，了解任务要求及当前进展。Hackathon 任务请按活动页面中的规则认领。

:::

## 开发准备

### 1. 获取代码

在 GitHub 上 **Fork** [redai-studio/Relax](https://github.com/redai-studio/Relax)，再将你的 fork 克隆到本地。将 `<your_user_name>` 替换为你的 GitHub 用户名：

```bash
git clone https://github.com/<your_user_name>/Relax.git
cd Relax
git remote add upstream https://github.com/redai-studio/Relax.git

# 同步主仓库的 main 分支
git checkout main
git pull upstream main
```

`origin` 指向你的 fork，`upstream` 指向 Relax 主仓库。每次开始新的贡献前，先切回本地 `main` 并拉取上游更新，再创建工作分支。开发工作都在工作分支上进行，本地 `main` 只用于同步主线。

### 2. 设置开发环境

环境要求见[安装指南](./installation.md)。

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装并启用 pre-commit 钩子
pip install pre-commit
pre-commit install

# 以开发模式安装 Relax
pip install -e .
```

### 3. 运行示例实验

如需验证训练环境，可以按照[快速开始](./quick-start.md)准备模型和数据，再运行相应示例。多轮视觉语言训练可参考 [DeepEyes 示例](../examples/deepeyes.md)。

## 开发工作流

### 1. 创建工作分支

```bash
git checkout -b feat/your-change
```

建议根据改动类型选择分支名前缀，例如：

- `feat/`：新功能
- `fix/`：Bug 修复
- `docs/`：文档更新
- `chore/`：维护任务

其他前缀可参考下方的提交消息约定。

### 2. 开发与调试

在工作分支上实现功能或修复问题，并注意以下几点：

- 编写清晰、可读的代码
- 遵循现有代码风格
- 为新功能添加测试
- 根据需要更新文档

### 3. 运行单元测试

完成代码修改后，为新增或修复的行为补充测试，并根据改动选择测试范围：

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/utils/test_metrics_service.py

# 带覆盖率运行
pytest --cov=relax tests/
```

### 4. 提交更改

完成相应验证后，先查看改动，再暂存本次准备提交的文件（将 `<changed-files>` 替换为实际路径，多个路径用空格分隔）：

```bash
git status
git diff
git add <changed-files>
git commit -m "feat: describe your change"
```

如果 hook 自动修改了文件或报告错误，请查看并修正改动，再重新 `git add` 和 `git commit`，直到检查通过并提交成功。

提交消息遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

- `feat:` - 新功能
- `fix:` - Bug 修复
- `docs:` - 文档更改
- `style:` - 代码风格更改（格式化等）
- `refactor:` - 代码重构
- `test:` - 添加或更新测试
- `chore:` - 维护任务

### 5. 创建 PR

将工作分支推送到你的 fork：

```bash
git push origin feat/your-change
```

推送后，在 GitHub 上创建 PR：来源分支选择你 fork 中的工作分支，目标分支选择 **`redai-studio/Relax` 的 `main` 分支**，并填写 [PR 模板](https://github.com/redai-studio/Relax/blob/main/.github/PULL_REQUEST_TEMPLATE.md)。

根据 CI 结果和审查意见，在同一分支继续修改、验证、提交并推送，PR 会自动更新。

## 代码风格指南

### Python 风格

- 遵循 [PEP 8](https://pep8.org/)
- 使用类型提示
- 为公共函数编写文档字符串
- 保持函数职责单一、实现简洁

示例：

```python
def compute_reward(
    response: str,
    ground_truth: dict,
    reward_type: str = "f1"
) -> float:
    """
    计算生成响应的奖励分数。
    
    Args:
        response: 模型生成的响应
        ground_truth: 真实数据
        reward_type: 要计算的奖励类型

    Returns:
        0 到 1 之间的奖励分数
    """
    if reward_type == "f1":
        return compute_f1_score(response, ground_truth["answer"])
    else:
        raise ValueError(f"未知的奖励类型: {reward_type}")
```

### 文档风格

- 使用清晰、简洁的语言
- 包含代码示例
- 在有帮助时添加图表
- 保持文档更新

## 测试指南

### 编写测试

```python
from relax.utils.metrics.client import MetricsClient

def test_metrics_client_log_metric():
    """测试记录指标。"""
    client = MetricsClient(service_url="http://localhost:8000/metrics")

    # 记录指标
    client.log_metric(step=1, metric_name="test/metric", metric_value=0.5)

    # 验证
    assert client.get_buffered_metrics_count(step=1) == 1
```

### 测试覆盖率

- 以超过 80% 的代码覆盖率为目标
- 测试边界情况和错误条件
- 对外部依赖使用 mock

## 文档指南

### 添加文档

1. 在 `docs/en/guide/` 和 `docs/zh/guide/` 中分别添加英文和中文 Markdown 文件
2. 更新 `docs/.vitepress/config.mts`，将页面加入侧边栏
3. 包含代码示例和图表
4. 确保中英文内容一致

### 构建文档

请先安装 Node.js，再在仓库根目录执行以下命令：

```bash
# 启动文档开发服务器
make docs-dev

# 构建文档
make docs-build

# 预览构建的文档
make docs-preview
```

## Pull Request 指南

### 提交前

- [ ] 测试在本地通过
- [ ] 代码已格式化
- [ ] 文档已更新
- [ ] 提交消息遵循约定
- [ ] 分支与 main 保持最新

### PR 描述

请在 PR 描述中说明：

- **What（改动内容）**：进行了哪些更改
- **Why（改动原因）**：为什么需要这些更改
- **How（实现方式）**：如何实现这些更改
- **Testing（验证方式）**：做了哪些测试或验证

示例：

```markdown
## What
为 DeepEyes 示例添加自定义奖励函数支持

## Why
用户需要灵活定义任务的奖励逻辑

## How
- 添加 `custom_reward.py` 模块
- 更新配置以支持自定义奖励函数
- 添加文档和示例

## Testing
- 添加自定义奖励函数的单元测试
- 使用 DeepEyes 示例进行测试
- 验证向后兼容性
```

## CI

PR 会运行 CPU 检查、GPU 单元测试和 GPU/NPU 集成测试。操作 CI 时，在新 PR 评论的首行写一条指令。PR 作者及拥有仓库写权限的贡献者可以重跑、取消 CI。

| 指令 | 用途 |
| --- | --- |
| `/rerun` | 重跑当前提交对应的最新失败或超时 workflow 中失败的 job |
| `/rerun <target>` | 重跑单个 workflow 或检查；`all` 表示所有已完成的 CI workflow |
| `/cancel <workflow>` | 取消整个 workflow，包括其矩阵 job；`all` 表示所有运行中的 CI workflow |
| `/help` | 查看指令和 target |
| `/review` | 请求代码 review |

| Target | Workflow / 检查 |
| --- | --- |
| `ci` | 全部 pre-commit 与 CPU 检查 |
| `pre-commit` | Pre-commit 检查 |
| `cpu-310`、`cpu-311`、`cpu-312` | Python 3.10 / 3.11 / 3.12 CPU 测试 |
| `gpu-unit` | GPU 单元测试 |
| `integration` | 全部 GPU/NPU 集成测试 |
| `gpu-async` | Qwen3-4B GPU 异步训练 |
| `gpu-vl` | Qwen3-VL-4B GPU 训练 |
| `npu-async` | Qwen3-4B NPU 异步训练 |

取消操作接受 workflow target（`ci`、`gpu-unit`、`integration` 或 `all`）。重跑只操作当前提交已有的 run；若 run 仍在运行，请等待结束或取消后再重跑。

## 交流与反馈

讨论时请就事论事，尊重不同意见。提出批评或建议时，尽量说明具体问题和理由；对刚接触项目的贡献者，也请多一些耐心。

### 提问与讨论

遇到问题时，可以先搜索已有的 Issue、PR 和讨论，看看是否已有解答。如果仍未解决，可以在 GitHub Discussions 中提问，或加入微信群交流。

### 报告 Bug

请在 Issue 标题中简要说明问题，并提供以下信息，方便其他人定位和复现：

- 复现步骤
- 预期结果和实际表现
- 报错信息及相关日志
- 运行环境，例如操作系统、Python 版本和硬件配置

## 感谢！

你的每一份提交都在让 Relax 框架越来越完善，感谢你为 Relax 做出的贡献！
