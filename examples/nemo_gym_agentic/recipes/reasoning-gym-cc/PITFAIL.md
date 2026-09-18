# Reasoning-Gym-CC PITFAIL：接入踩坑记录

本文件记录 reasoning-gym-cc recipe 中容易产生错误结论的地方。命令以 [README.md](README.md) 为准。

## 1. bwrap × 内核不兼容 → Bash tool 全崩，reward 集群 0

### 症状识别（组合出现 = 就是这个 bug）

- `rollout_result/train/*.jsonl` 里 `<tool_response>` 大量出现下面两串：
  - `apply-seccomp: prctl(PR_SET_SECCOMP): Unknown error 524`（内核不认 bwrap 塞的 seccomp filter）
  - `bwrap: Can't find source path /opt/nemo-gym/responses_api_agents/claude_code_agent/.claude/settings.json`
  - 或更早期：`bwrap: No permissions to create new namespace`（容器无 SYS_ADMIN / seccomp 拦）
- 用 tool 的样本 100% reward=0（或 puzzle24 之类的 format 分 0.01），`<tool_call>` 打满 `max_turns=30` 后 truncated
- 训练 log 里 `claude-code exited 1: --dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons`
- reward 均值靠 pure-think 硬推题（矩阵旋转、Tsumego）撑到 0.2~0.5，看起来"能训"但 Bash tool 实际零成功

### 根因

Claude Code CLI 用 [bubblewrap](https://github.com/containers/bubblewrap) 给 Bash tool 起沙箱。
在 kernel 5.10.134-16.3.an8 (TencentOS) + bwrap 0.9.0 的组合上，bwrap 塞进内核的 seccomp
filter 用了这个内核不支持的 action → `prctl(PR_SET_SECCOMP)` 返 EINVAL/ENOTSUPP (524)。
容器起时若又缺 `--cap-add SYS_ADMIN`/`--security-opt seccomp=unconfined`，还会先在
namespace 创建这一步就挂。

绕过 sandbox 需要**三处**同步配合，缺一处都白搭：

1. **`docker create` 层**：`--cap-add SYS_ADMIN --security-opt seccomp=unconfined`
   —— 允许 `clone(CLONE_NEWUSER)`。不加会报 `bwrap: No permissions`。
2. **`app.py` 里的 env dict**（`/opt/nemo-gym/responses_api_agents/claude_code_agent/app.py`
   `_run_claude_code`）：
   - `IS_SANDBOX="0"` —— CLI 不再自启 bwrap
   - `CLAUDE_CODE_BUBBLEWRAP="1"` —— 骗过 CLI 的 `getuid()==0 && IS_SANDBOX!="1" && !CLAUDE_CODE_BUBBLEWRAP` 三条件 root check（源码搜 `cannot be used with root` 定位）
3. **`configs/claude_settings.json` 的 `sandbox` 块**：`enabled=false`、
   `failIfUnavailable=false`、`allowUnsandboxedCommands=true` —— settings.json 是 CLI 的
   **独立 gate**，即使 env 关了，settings 里 `enabled=true` 仍会强制走 bwrap 并因
   `failIfUnavailable=true` 直接报 `sandbox required but unavailable`

已经落地在 checkout 里：

- `start_reasoning_gym_cc_gym_remote.sh` docker create 加了 (1)
- `start_reasoning_gym_cc_gym.sh` 启动前 sed patch 加 (2)，幂等，pattern 匹配不到会 hard-fail
- `configs/claude_settings.json` 已按 (3) 关闭 sandbox

### 排查方法

- 直接看 rollout jsonl 的 `<tool_response>` 分类：

  ```python
  import json, re
  from collections import Counter
  rows = [json.loads(l) for l in open("rollout_result/train/N.jsonl")]
  def cls(t):
      if "apply-seccomp"          in t: return "seccomp_524"
      if "bwrap: Can"             in t: return "bwrap_missing_path"
      if "bwrap: No permissions"  in t: return "bwrap_no_ns"
      if "bwrap"                  in t: return "bwrap_other"
      if "sandbox required"       in t: return "sandbox_required"
      if "cannot be used with root" in t: return "root_check_fail"
      return "clean"
  tot = Counter()
  for r in rows:
      for t in re.findall(r"<tool_response>(.*?)</tool_response>", r["response"], re.S):
          tot[cls(t)] += 1
  print(tot)
  ```

  修好后 `clean` 应 ~= 100%。

- 直接验容器：

  ```bash
  docker exec <container> bash -c '
    unshare --user --pid --fork -- echo ok   # 验 (1)
    grep -E "IS_SANDBOX|CLAUDE_CODE_BUBBLEWRAP" \
      /opt/nemo-gym/responses_api_agents/claude_code_agent/app.py  # 验 (2)
    grep -A2 "\"enabled\"" \
      /opt/relax-integration/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/configs/claude_settings.json  # 验 (3)
  '
  ```

- 端到端验 bwrap 真的不被调（不用等训练）：wrap `/usr/bin/bwrap` 记录调用，然后跑一发
  CLI 有 Bash tool 的 prompt，看 `/tmp/bwrap-invocations.log` 是不是 0 行。

### 一次修完后必须做

- **重启 gym 容器**（`docker restart` 或 `docker rm -f` + 重跑 `_remote.sh`）：`app.py` 是
  Python 长驻进程，字符串字面量在 import 时编译进 bytecode 常量池，磁盘 sed 完必须重新
  import。**热修 sed 不重启无效**，容易骗自己。
- `settings.json` 是每次 claude subprocess 起动时现读磁盘，不需要重启即可生效 —— 但
  spawn 中的 subprocess 已经读过了，得等下一批 rollout。所以修完后**跳过下一份 jsonl，
  从再下一份开始信数据**。

### 安全权衡

关掉 bwrap 之后同容器里 64 并发的 claude Bash tool 都跑在同一个 root fs 上：

- **没了的隔离**：filesystem、network、`/tmp`、subprocess 之间的 FS/进程视图
- **已有的目录分离**：每次 request 有独立 `CLAUDE_CONFIG_DIR`（`~/.claude_code_agent/<uuid>/`），
  Claude 子进程的 `cwd` 指向其 `workspace/`，`TMPDIR`、`TMP`、`TEMP` 指向其 `tmp/`，
  这些目录随 request 退出清理。补丁在镜像构建时应用，更新后必须重建镜像并重建容器。
- **工具限制的边界**：`--bare` 减少自动发现，`permissions.deny` 禁用列出的工具，
  但禁用 Read/Write 并不能阻止 Bash 读写文件。
- **剩余风险**：独立 `cwd` 和临时目录减少同名相对文件、遵循临时目录环境变量的文件发生串扰；
  显式写 `/tmp/foo`、共享 HOME 或其他绝对路径仍可能互相覆盖、读取其他 trial 的结果。
  即使是无对抗的数学题，模型生成的脚本也可能意外碰撞。需要严格隔离时仍须提供实际文件系统沙箱。

### 长期方向

- 内核升到能过 bwrap 0.9+ 的 seccomp filter 的版本
- 或降到 bwrap 0.6/0.8（用不了那么新的 seccomp action）
- 或换 firejail / systemd-run 之类的替代
- 保持 sandbox 关闭时，独立工作目录只能减少意外串扰，不能据此认定训练不存在交叉污染

参考：2026-09-09 rollout 3/4 对比 —— rollout 3 (`19:55`) 修 settings 前 572/572 seccomp_524
→ rollout 4 (`19:59`) 修完后 296/296 clean，reward 均值 0.28 → 0.42。
