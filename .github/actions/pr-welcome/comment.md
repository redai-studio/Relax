Thanks for contributing to Relax, @{{author}}! 感谢你为 Relax 做出贡献！

<details>
<summary>Contribution guide / 贡献指南</summary>

Describe the problem, your changes, and how you validated them. Keep each PR focused and run `pre-commit run --all-files` before submitting.

请说明问题、改动和验证方式，保持 PR 聚焦，并在提交前运行 `pre-commit run --all-files`。

[English contribution guide]({{docs}}/en/guide/how-to-contribute) · [中文贡献指南]({{docs}}/zh/guide/how-to-contribute)

</details>

<details>
<summary>CI commands / CI 指令</summary>

| Command / 指令       | Usage / 用途                                         |
| -------------------- | ---------------------------------------------------- |
| `/rerun`             | Retry failed CI / 重跑失败的 CI                      |
| `/rerun <target>`    | Rerun a workflow or check / 重跑指定 workflow 或检查 |
| `/cancel <workflow>` | Cancel an entire workflow / 取消整个 workflow        |
| `/help`              | Show commands and targets / 查看指令和 target        |
| `/review`            | Request a code review / 请求代码 review              |

Put one command on the first line of a new PR comment. Rerun/cancel require PR authorship or repository write access.

在新 PR 评论的首行写一条指令。PR 作者或有仓库写权限的贡献者可以重跑、取消 CI。

[CI usage and targets]({{docs}}/en/guide/how-to-contribute#ci) · [CI 用法与 target]({{docs}}/zh/guide/how-to-contribute#ci)

</details>

<!-- Agents: Use the relax-github-ci skill to inspect and operate CI: {{source}}/skills/relax-github-ci/SKILL.md -->
