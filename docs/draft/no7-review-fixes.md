# No.7：PR #377 评审修复记录

日期：2026-09-26。评审基线：`1ab8a1e`。本记录对应本轮工作树修复。

[评审总览](https://github.com/redai-studio/Relax/pull/377#issuecomment-5831768394)包含下述五项问题；总览本身不是第六项独立缺陷。

## 逐项处理

| 意见 | 根因与修复 | 回归保护 |
| --- | --- | --- |
| [F1 / P1](https://github.com/redai-studio/Relax/pull/377#discussion_r4104234833) | 公共 abort 无条件读取仅 overlap 循环创建的 `result_queue`。补丁改为 `getattr(self, "result_queue", ())`，缺省为空，仍扫描实际 overlap 在途结果。 | 直接执行真实 `Scheduler.abort_request`，覆盖 non-overlap、PP=2、overlap × managed/unmanaged 六种组合；检查精确/前缀取消、已完成请求不修改、在途请求不提前释放。 |
| [F2 / P1](https://github.com/redai-studio/Relax/pull/377#discussion_r4104237607) | 纯 fully-async 不初始化权重备份器，bootstrap 却无条件切换权重。将启动导出集中到小型私有方法，仅存在备份器时恢复 actor；纯异步直接导出现有 actor 权重。 | 覆盖 sync、hybrid、fully-async 和无导出意图；检查恢复在导出前发生、导出 actor 而非 ref、意图仅消费一次。 |
| [F3 / P2](https://github.com/redai-studio/Relax/pull/377#discussion_r4104238567) | READY 校验已有 digest 比较，但缺错误摘要回归。新增 E2 错误 digest，并增强所有非法 READY 用例。 | 先发布 A，再发布 B；验证 A 默认身份保持、B ABORTED、两引擎均 ABSENT 且 B 各卸载一次，A 不卸载，占用回到 1。 |
| [F4 / P2](https://github.com/redai-studio/Relax/pull/377#discussion_r4104239507) | 测试把逻辑引用释放顺序误当 GPU 安全条件，依赖事件循环调度时序。改为断言 fence 未完成、资源仍驻留、无卸载、close/KV close 未完成。 | 两项测试分别加入 0/3 次调度让步；保留完成后的精确引用计数、恰好一次释放、重复 close 幂等与另一会话不受影响。 |
| [F5 / P2](https://github.com/redai-studio/Relax/pull/377#discussion_r4104248271) | 使用 `object.__new__` 的握手测试夹具缺少新访问的 config。补齐 `lora_publication_config=None`，生产构造逻辑不变。 | 保留原有异常释放 gate 测试，新增 publication 返回 409 且不调用旧握手、不暂停、不改变 gate 的测试。 |

## 本轮验证

环境：Python 3.12.3；CPU 测试使用安装在 `/workspace/sglang-no7` 的完整 SGLang 运行时，已同步本轮 abort 补丁。没有启动 GPU 生成或训练。

```bash
RELAX_SGLANG_SOURCE=/workspace/sglang-no7 python -m pytest -q \
  tests/engine/lora \
  tests/backends/sglang \
  tests/backends/megatron/test_lora_export_lifecycle.py \
  tests/components/test_actor_lora_publication.py \
  tests/components/test_rollout_weight_update_handshake.py \
  tests/distributed/ray/test_lora_membership.py \
  --disable-warnings
```

结果：**448 passed、6 skipped、42 warnings**。六个 skip 是需要显式启动 GPU 实验的验收入口；本轮未执行 GPU、多节点、多 rank 或完整训练实验。警告包括依赖弃用及测试 shell 的 Ray Serve 异步析构提示。

格式与静态检查：`pre-commit run --all-files` 全部通过；新增修复记录单独运行 pre-commit 通过；`git diff --check` 通过。

额外验证：

- 在独立测试进程中删除 digest 比较：非法 READY 五个用例中 **1 failed、4 passed**；错误摘要用例准确捕获 B 被错误发布。生产文件未作此变异。
- 在独立测试进程中恢复旧的直接 `self.result_queue` 访问：新增 abort 六个用例中 **4 failed、2 passed**；non-overlap/PP 均复现 AttributeError，overlap 通过。生产文件未作此变异。
- 使用临时 Git index 对干净 SGLang `v0.5.17`（`b6a09f38fcc5e96574324b4acc19d421c539cfc6`）检查整个补丁，`git apply --cached --check` 通过。

## 历史 GPU 证据范围

`tests/engine/lora/evidence/2026-09-25/` 的压缩结果和 `source_sha256` 保持不变，继续标识原始验收版本。本轮修改了源码和测试，因此历史摘要不能作为本轮源码完全匹配或新 GPU 复跑的声明。原始数值实验仍可复核；本轮新增结论限于上述 CPU 回归、变异验证与补丁应用检查。

Python 3.10/3.11 CI、远端 PR 检查和导师复审需在更新分支后确认；本地通过不等同于远端评审意见已标记解决。
