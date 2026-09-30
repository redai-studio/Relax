# GitHub CI

## 快速开始

Relax 会在 PR（包括 draft PR）上运行 CPU 检查、GPU 单元测试和 GPU/NPU 集成测试。每条 PR 评论只发送一条指令：

```text
/rerun
```

这会重跑当前 PR head 对应的最新失败或超时 workflow 中失败的 job。重跑单项检查可使用 `/rerun cpu-310`。指令不会创建尚不存在的 workflow run，也不会批准 fork workflow。

## 指令

| 指令 | 行为 | 使用权限 |
| --- | --- | --- |
| `/rerun` 或 `/rerun failed` | 重跑当前 head 最新失败或超时 workflow 中失败的 job | PR 作者或仓库 write/maintain/admin |
| `/rerun <target>` | 重跑整个 workflow，或单项检查及其依赖 job | 同 `/rerun` |
| `/rerun all` | 重跑所有已完成的配置内 workflow | 同 `/rerun` |
| `/cancel <workflow>` 或 `/cancel all` | 请求取消整个运行中的 workflow | 同 `/rerun` |
| `/help` | 展示指令与 target | 所有人 |
| `/bypass <workflow>` 或 `/bypass all` | 留下供 ci-bypass 在 workflow 运行时判断的评论 | 配置的 CI team |
| `/review` | 请求 Nyanpasu review | 由 Nyanpasu 控制 |

指令 action 处理 `/rerun`、`/cancel` 和 `/help`。ShigureLab/ci-bypass 读取豁免标签或评论；Nyanpasu 响应 `/review`。三者入口独立。`/bypass` 不会产生 bot 回执，也不会自动重跑 CI。

统一使用 `/rerun`，不支持 `/re-run`。没有 `/ci` 或 `/unbypass`。只有新建评论才会执行 rerun/cancel/help，编辑评论不会再次执行。指令后允许附加标准 agent 自动回复声明。

## Target

| Target | Workflow / 检查 |
| --- | --- |
| `ci` | `ci.yml`：pre-commit 与 CPU 测试 |
| `pre-commit` | Pre-commit Checks |
| `cpu-310`、`cpu-311`、`cpu-312` | Tests (Python 3.10 / 3.11 / 3.12) |
| `gpu-unit` | `unittest.yml`：GPU 单元测试 |
| `integration` | `integration-test.yml`：GPU/NPU 集成测试 |
| `gpu-async` | Qwen3-4B-4xgpu-async |
| `gpu-vl` | Qwen3-VL-4B-2xgpu |
| `npu-async` | Qwen3-4B-4xnpu-async |

`/cancel` 和 `/bypass` 接受 workflow target：`ci`、`gpu-unit`、`integration` 或 `all`。取消整个 workflow 会同时取消其兄弟矩阵 job。部署 workflow 不在操作范围内，`all` 也不包含它。

## 豁免

CI team 在 `.github/ci/config.json` 中配置：`SigureMo`、`NINGBENZHE` 和 `Yangruipis`。成员可以添加 `ci-bypass: <workflow>` / `ci-bypass: all` 标签，或发表评论：

```text
/bypass gpu-unit
```

理由可省略，例如 `/bypass gpu-unit --reason runner maintenance` 也可以。之后使用 `/rerun gpu-unit` 重跑整个 workflow，让 gate 重新判断。只重跑单个 job 可能复用旧 gate 结果。如果 workflow 尚未结束，请等待或显式取消后再重跑。

沿用 Paddle 的 workflow 结构，ShigureLab/ci-bypass 直接核对标签或评论作者是否在配置的 team 中。独立的 `synchronize` workflow 会在新提交推送后移除 `ci-bypass:` 标签。此清理只影响标签：历史 `/bypass` 评论仍按 ci-bypass 的评论规则参与判断。撤回评论豁免时，需要删除或修改该评论，再重跑 workflow。

## 配置与维护

| 文件 / 配置 | 用途 |
| --- | --- |
| `.github/ci/config.json` | 共享的 CI team 与 workflow/job 映射 |
| `.github/actions/ci-commands` | 使用 `GITHUB_TOKEN` 执行 rerun、cancel 和 help |
| `.github/actions/pr-welcome` | 欢迎模板与评论发布 |
| `.github/workflows/check-bypass.yml` | 在 `ubuntu-slim` 上运行的可复用 ShigureLab/ci-bypass gate |
| `.github/workflows/remove-ci-bypass-labels.yml` | 新提交触发的独立标签清理 |
| Secret `WELCOME_BOT_TOKEN` | 欢迎 bot 的 token，初始账号为 `rai-studio-bot` |
| Variable `WELCOME_BOT_LOGIN` | Bot 改名时覆盖预期的欢迎账号 login |

JavaScript 直接读取 JSON。Team 不能为空，用户名应使用 GitHub 实际 login 的大小写，ci-bypass 会进行大小写敏感比较。新增 target 时核对 workflow/job 名称；含 `job` 的 target 只支持 rerun。

只有欢迎 workflow 需要 `WELCOME_BOT_TOKEN`，其仓库权限需允许写 PR 评论，并会在发布前校验 token 所属账号。Rerun/cancel 使用 `GITHUB_TOKEN` 的 Actions 写权限，状态回复使用其 Issues 写权限。两个评论处理 workflow 均 checkout 可信的默认分支代码，并禁用凭据持久化。Nyanpasu 的 `/review` 唤醒词单独配置。

欢迎内容维护在 `.github/actions/pr-welcome/comment.md`。重新打开 PR 不会再发一条欢迎评论。需要使用的 bypass 标签可在仓库设置中创建，gate 仅读取标签和评论。

模块检查使用与 `actions/github-script` 一致的 Node.js 24。Pre-commit hook 安装固定版本的 oxfmt PyPI 包来格式化 CI JavaScript/JSON，无需 npm 解析依赖：

```bash
node --input-type=module -e "await import('./.github/actions/ci-commands/index.mjs'); await import('./.github/actions/pr-welcome/index.mjs')"
pre-commit run --all-files
```

合并到默认分支并配置 bot 后，使用临时 draft PR 验证欢迎评论、指令权限、rerun/cancel、豁免判断和标签清理。本地模块检查不能验证真实 GitHub 权限或 GPU/NPU 执行。

## Stacked PR

使用 `gh stack` 注册原生 GitHub stack。Stack base 为 `main` 时，现有 `pull_request.branches: [main]` 根据 stack base 判断是否触发。仅手动串联 PR 的 base 分支没有此行为。详见 [GitHub stacked PR 文档](https://docs.github.com/en/pull-requests/reference/stacked-pull-requests)。

## 故障排除

- **找不到匹配的 run：** 核对 head SHA、workflow 触发条件、stack 注册及 fork workflow 审批状态。`/rerun` 无法创建 run。
- **Run 仍在运行：** 等待结束或显式取消，取消请求是异步的。
- **Head 已变化：** 检查新提交的 CI 后重新发送指令。
- **Processing 回执 / API 超时：** 重试前先检查关联 workflow 和目标 run。重复事件或重跑指令 workflow 不会再次执行同一条评论。
- **Bypass gate 出错：** 常规测试继续执行，错误不会变成豁免。
- **欢迎 token 缺失或不匹配：** 配置 bot token 和对应 login；这不会影响 rerun/cancel 或 bypass。

## 下一步

- [贡献指南](./how-to-contribute.md)
- [调试指南](./debugging.md)
