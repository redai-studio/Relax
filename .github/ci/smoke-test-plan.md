# CI 评论指令在线测试计划

本 draft PR 是一次性在线测试环境，不合并。最终 diff 仅保留本计划；临时失败探针已在收尾时移除，不修改生产 workflow 或 action。

## 关联 PR

- #392：共享 CI 配置、target 映射和维护权限。
- #393：独立的 `/help`、`/rerun`、`/cancel` 评论指令。
- #394：workflow 豁免 gate 与 synchronize 标签清理。
- #395：贡献者欢迎评论及 CI 使用说明。
- #396：CI 运维技能和直接 API 操作说明。
- #398：oxfmt 文件覆盖及格式检查。

## 执行约定

- 所有操作只针对本 PR；每次写操作前确认 PR 仍开放、head 未改变。
- 评论指令使用已部署到默认分支的实现；直接 API 仅用于取证、测试准备、去重重放或收尾。
- 每项测试完成后发布独立评论，记录通过/失败、实际现象、命令评论、回复和 run/job 证据；失败项同时记录影响和改进建议。
- PR 描述中的状态表会链接对应测试记录。预期失败探针不算产品缺陷；API 接受请求也不等于测试通过。
- GPU/NPU workflow 仅验证路由、取消及豁免；本轮不以重新验证训练正确性为目标。
- 无独立身份的非作者/无写权限用户、非 CI team、Bot 类型评论，以及 fork、并发竞态，不宣称已在线实测。

## 测试用例

| 编号 | 步骤                                             | 预期结果                                                                                                   | 状态 / 记录 |
| ---- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ----------- |
| T01  | 创建本 draft PR                                  | 配置的欢迎账号发布且仅发布 1 条双语欢迎评论，文档链接和指令正确                                            | 待测试      |
| T02  | 新评论 `/help`                                   | 返回与配置一致的 workflow/job target 列表                                                                  | 待测试      |
| T03  | `/rerun unknown-target`、`/cancel ci/pre-commit` | 明确拒绝无效 target 和 job 级取消，CI attempt 不变                                                         | 待测试      |
| T04  | CI 运行中执行 `/rerun ci`                        | 提示仍在运行，不创建新 attempt                                                                             | 待测试      |
| T05  | `/cancel gpu-unit`、`/cancel integration`        | 只取消本 PR 对应的完整 workflow，状态进入终态                                                              | 待测试      |
| T06  | 确认失败探针执行后评论 `/rerun failed`           | 同一 head 的失败 job 实际重跑，成功的 sibling 不重跑；核对执行时间和 step 记录                             | 待测试      |
| T07  | CPU 重跑开始后评论 `/cancel ci`                  | 整个 CI workflow 进入 cancelled 终态                                                                       | 待测试      |
| T08  | CI 完成后评论 `/rerun ci/pre-commit`             | 只重跑指定 job 及其依赖后继；按执行时间和 step 记录确认其他 job 未重跑                                     | 待测试      |
| T09  | 创建普通评论，再编辑为 `/help`                   | 编辑不触发命令执行，无对应回复或新 command job                                                             | 待测试      |
| T10  | 重跑 T08 对应的 CI Commands workflow             | 同一 source comment 的回复 ID 和数量不变，目标 CI attempt 不增加                                           | 待测试      |
| T11  | 添加 `ci-bypass: ci`，再 `/rerun ci`             | gate 重新计算且成功，CI 下游检查 skipped                                                                   | 待测试      |
| T12  | 添加 `ci-bypass: all`，再 `/rerun all`           | 3 个配置的 workflow 均创建新 attempt，下游检查全部 skipped                                                 | 待测试      |
| T13  | 移除豁免标签，评论 `/bypass ci`，再 `/rerun ci`  | 评论豁免生效；验证后撤销该评论的指令                                                                       | 待测试      |
| T14  | 没有 failure/timed_out run 时评论 `/rerun`       | 不产生新 attempt                                                                                           | 待测试      |
| T15  | 带豁免标签推送移除失败探针的新 commit            | synchronize 清除所有 `ci-bypass:` 标签；分别验证新 head 的 gate 和命令选择，不把清理标签等同于 gate 未跳过 | 待测试      |
| T16  | 重跑欢迎 workflow                                | 仍然只有 1 条欢迎评论                                                                                      | 待测试      |

表中的状态仅表示初始计划；执行过程以 PR 描述的状态表和逐项测试评论为准。

## 收尾

移除临时失败探针，撤销所有测试用豁免标签和有效 `/bypass` 评论，确认本 PR 没有测试任务继续运行，并保持 draft。同步清理标签与 CI gate 并行执行，须分别验证标签消失和新 head 的 gate 行为。

## 首轮执行调整

- 评论指令实际在创建初始回复时返回 HTTP 403，T02～T05 记录为失败。
- 依赖该入口的 T06、T07、T08、T10、T14 标为阻塞，等待修复部署后重测。
- T11～T13 改用直接 Actions API 重跑整个 workflow，仅验证豁免 gate；原计划中的评论重跑链路仍为阻塞，因此整体状态记为部分通过。
- T15 分别记录 synchronize 标签清理、新 head gate 的实际结果，以及尚未验证的评论路由。
- 单 job 重跑可能复制其他 job 的历史记录，因此不能仅用 job ID 或 workflow attempt 判断实际执行次数。
- 每项的最终状态和证据链接以 PR 描述、独立测试评论为准。

## #401 合入后的第二轮复测

- 默认分支修复 PR 评论写入权限后，用新的评论事件重测 T02～T08、T10～T15；保留第一轮记录并链接新证据。
- 临时 `tests/000_ci_command_smoke/test_ci_command_smoke_probe.py` 仅在本仓库、本 draft 分支、CI workflow 的 CPU job 生效。Python 3.10 产生明确的预期失败；Python 3.11/3.12 提前退出成功，作为重跑选择的合成控制组，不代表完整单测通过。
- 以日志确认探针确实执行；测试收集错误或依赖安装失败不能冒充探针失败。复测结束后删除探针。
- T17 补充检查豁免后的 required contexts；当前 job 级豁免使 matrix 未展开，缺少 5 个具名检查，保持失败。该缺陷独立于 #401 的权限修复。
