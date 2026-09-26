# 如何贡献

感谢你对 Relax 项目的关注！本指南将帮助你开始贡献。

## 开发流程

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

`origin` 指向你的 fork，`upstream` 指向 Relax 主仓库。后续贡献前，切回本地 `main` 并拉取上游更新，再创建工作分支。请在工作分支上开发，保持本地 `main` 用于同步主线。

### 2. 设置开发环境

环境要求见[安装指南](./installation.md)。

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装
pip install -e .
```

### 3. 运行示例实验

```bash
# 运行基础示例
python relax/entrypoints/train.py

# 运行 DeepEyes 示例
cd examples/deepeyes
bash run_deepeyes.sh
```

### 4. 开始开发

```bash
git checkout -b feature/your-change
```

安装 pre-commit 和 Git hooks：

```bash
pip install pre-commit
pre-commit install
```

安装后，每次 `git commit` 都会自动运行检查。

### 5. 执行单元测试

完成代码修改后，为新增或修复的行为补充测试，并根据改动选择测试范围：

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/utils/test_metrics_service.py

# 带覆盖率运行
pytest --cov=relax tests/
```

### 6. 提交更改

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

### 7. 创建 PR

```bash
git push origin feature/your-change
```

在 GitHub 上，从你 fork 中的工作分支向 **`redai-studio/Relax` 的 `main` 分支**创建 PR，并填写 [PR 模板](https://github.com/redai-studio/Relax/blob/main/.github/PULL_REQUEST_TEMPLATE.md)。如果使用了其他分支名，请相应替换命令中的名称。

后续根据 CI 结果和审查意见，在同一分支修改、检查、提交并推送，PR 会自动更新。

## 代码风格指南

### Python 风格

- 遵循 [PEP 8](https://pep8.org/)
- 使用类型提示
- 为公共函数编写文档字符串
- 保持函数专注和简洁

示例：

```python
def compute_reward(
    response: str,
    ground_truth: dict,
    reward_type: str = "f1"
) -> float:
    """
    计算响应的奖励。
    
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

- 目标 >80% 代码覆盖率
- 测试边界情况和错误条件
- 对外部依赖使用 mock

## 文档指南

### 添加文档

1. 将 markdown 文件添加到 `docs/en/guide/` 或 `docs/zh/guide/`
2. 更新 `docs/.vitepress/config.mts` 以添加到侧边栏
3. 包含代码示例和图表
4. 提供中英文两个版本

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

包含：

- **What（什么）**：进行了哪些更改
- **Why（为什么）**：为什么需要这些更改
- **How（如何）**：更改如何工作
- **Testing（测试）**：如何测试更改

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

## 审查流程

1. **自动检查**：CI/CD 运行测试和 linter
2. **代码审查**：维护者审查代码
3. **反馈**：处理审查意见
4. **批准**：获得维护者批准
5. **合并**：PR 合并到 main 分支

## 社区指南

### 尊重他人

- 对他人友善和尊重
- 欢迎新人
- 提供建设性反馈
- 心怀善意

### 寻求帮助

- 使用 GitHub Discussions 提问
- 加入我们的微信群
- 检查现有 issues 和 PRs

### 报告问题

报告 Bug 时：

- 使用清晰、具有描述性的标题
- 描述复现步骤
- 包含错误消息和日志
- 说明运行环境（操作系统、Python 版本等）

## 贡献领域

### 代码

- 新功能
- Bug 修复
- 性能改进
- 代码重构

### 文档

- 改进现有文档
- 添加新指南
- 翻译到其他语言
- 修复错别字和错误

### 示例

- 添加新示例
- 改进现有示例
- 添加教程

### 测试

- 添加新测试
- 提高测试覆盖率
- 添加集成测试

## 获取帮助

- **GitHub Issues**：报告 bug 和请求功能
- **GitHub Discussions**：提问和讨论想法
- **微信群**：加入我们的社区
- **Email**：联系维护者

## 许可证

你贡献的代码和文档将按 Apache 2.0 开源许可证发布。

## 感谢！

你的每一份提交都在让 Relax 框架越来越完善，感谢你为 Relax 做出的贡献！
