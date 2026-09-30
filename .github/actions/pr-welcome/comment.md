Thanks for contributing to Relax, @{{author}}! 感谢你为 Relax 做出贡献！

<details>
<summary>Contribution guide / 贡献指南</summary>

Describe the problem, your changes, and how you validated them. Keep each PR focused and run `pre-commit run --all-files` before submitting.

请说明问题、改动和验证方式，保持 PR 聚焦，并在提交前运行 `pre-commit run --all-files`。

[English contribution guide]({{docs}}/en/guide/how-to-contribute) · [中文贡献指南]({{docs}}/zh/guide/how-to-contribute)

</details>

<details>
<summary>CI guide and commands / CI 指南与指令</summary>

| Command / 指令       | Usage / 用途                                                  |
| -------------------- | ------------------------------------------------------------- |
| `/rerun`             | Rerun failed CI on the current commit / 重跑当前提交失败的 CI |
| `/rerun <target>`    | Rerun a workflow or check / 重跑指定 workflow 或检查          |
| `/cancel <workflow>` | Cancel an entire workflow / 取消整个 workflow                 |
| `/bypass <workflow>` | CI team exemption via ci-bypass / CI team 通过 ci-bypass 豁免 |
| `/help`              | Show commands and targets / 查看指令和 target                 |
| `/review`            | Request Nyanpasu review / 请求 Nyanpasu review                |

Post one command per comment. Workflow targets are `ci`, `gpu-unit`, `integration`, or `all`. Rerun/cancel require PR authorship or repository write access. Bypass is restricted to the configured CI team; a reason is optional. After requesting bypass, rerun the entire workflow to evaluate the gate.

每条评论只写一条指令。Workflow target 为 `ci`、`gpu-unit`、`integration` 或 `all`。PR 作者或有仓库写权限的成员可重跑、取消 CI。豁免仅限配置的 CI team，理由可省略；提交豁免后重跑整个 workflow，让 gate 重新判断。

[English CI guide]({{docs}}/en/guide/github-ci) · [中文 CI 指南]({{docs}}/zh/guide/github-ci)

</details>

<details>
<summary>CI configuration and maintenance / CI 配置与维护</summary>

CI team members and targets are configured in [`.github/ci/config.json`]({{source}}/.github/ci/config.json). Welcome comments and CI commands have separate [actions]({{source}}/.github/actions). Exemptions use [ShigureLab/ci-bypass](https://github.com/ShigureLab/ci-bypass); new commits automatically remove `ci-bypass:` labels. `/review` is handled by Nyanpasu.

CI team 和 target 集中配置在 [`.github/ci/config.json`]({{source}}/.github/ci/config.json)。欢迎评论和 CI 指令由独立的 [actions]({{source}}/.github/actions) 处理。豁免使用 ShigureLab/ci-bypass；新提交自动清除 `ci-bypass:` 标签。`/review` 由 Nyanpasu 处理。

</details>
