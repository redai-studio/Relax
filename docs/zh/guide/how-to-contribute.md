# 如何贡献

感谢您对 Relax 项目的关注！本指南将帮助您开始贡献。

## 开始

### 1. 设置开发环境

创建虚拟环境并安装依赖：

```bash
# 克隆仓库
git clone https://github.com/redai-studio/Relax.git
cd Relax

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装
pip install -e .
```

### 2. 启动 Ray 并部署服务

```bash
# 启动 Ray 集群
ray start --head

# 部署所有服务
python -m relax.core.controller deploy --config configs/env.yaml
```

### 3. 运行示例实验

```bash
# 运行基础示例
python relax/entrypoints/train.py

# 运行 DeepEyes 示例
cd examples/deepeyes
bash run_deepeyes.sh
```

## 开发工作流

### 1. 创建分支

```bash
# 创建功能分支
git checkout -b feature/your-feature-name

# 或创建修复分支
git checkout -b fix/your-bug-fix
```

### 2. 进行更改

- 编写清晰、可读的代码
- 遵循现有代码风格
- 为新功能添加测试
- 根据需要更新文档

### 3. 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/utils/test_metrics_service.py

# 带覆盖率运行
pytest --cov=relax tests/
```

### 4. 格式化代码

```bash
# 使用 black 格式化
black relax/

# 排序导入
isort relax/

# 运行 linter
flake8 relax/
```

### 5. 提交更改

```bash
# 暂存更改
git add .

# 使用描述性消息提交
git commit -m "feat: add new feature"
# 或
git commit -m "fix: resolve bug in metrics service"
```

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

- `feat:` - 新功能
- `fix:` - Bug 修复
- `docs:` - 文档更改
- `style:` - 代码风格更改（格式化等）
- `refactor:` - 代码重构
- `test:` - 添加或更新测试
- `chore:` - 维护任务

### 6. 推送并创建 Pull Request

```bash
# 推送到您的 fork
git push origin feature/your-feature-name

# 在 GitHub 上创建 pull request
```

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

## 测试指南

### 编写测试

```python
import pytest
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

1. 将 markdown 文件添加到 `docs/guide/` 或 `docs/zh/guide/`
2. 更新 `.vitepress/config.mts` 以添加到侧边栏
3. 包含代码示例和图表
4. 提供中英文两个版本

### 构建文档

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
- 假定善意

### 寻求帮助

- 使用 GitHub Discussions 提问
- 加入我们的微信群
- 检查现有 issues 和 PRs

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

通过为 Relax 做出贡献，您同意您的贡献将根据 Apache 2.0 许可证授权。

## 感谢！

感谢您为 Relax 做出贡献！您的贡献帮助这个项目变得更好。
