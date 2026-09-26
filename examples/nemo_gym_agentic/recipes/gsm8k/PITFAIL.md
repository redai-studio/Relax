# GSM8K PITFAIL：接入踩坑记录

本文件记录 GSM8K recipe 中容易产生错误结论的地方。命令以 [README.md](README.md) 为准。

## 1. 这是 test split，不是正式 RL train set

固定 Gym commit 的 prepare 生成 1319 条 GSM8K test 题。用它跑一两条 E2E smoke 可以，但扩大
`NEMO_GYM_DATA_LIMIT` 做训练会污染 benchmark。正式训练应换独立数学训练集，并保留 GSM8K test
做 held-out evaluation。

## 2. Gym 原始输入和 Relax 转换文件不能混用

训练 wrapper 的输入应是：

```text
gsm8k_benchmark.jsonl
```

不是：

```text
gsm8k_relax.jsonl
```

wrapper 会在 Ray job 内再次调用 `convert_dataset.py`。把已转换文件作为源输入会因 schema 不匹配而
报错或丢失 verifier 字段。

## 3. `expected_answer` 必须转换为字符串

Gym prepare 可能写出数值：

```json
{"expected_answer": 18}
```

但 pinned `math_with_judge` 的请求模型要求字符串。converter 会规范成：

```json
{"metadata": {"expected_answer": "18"}}
```

绕过 converter 或手工删改 metadata 会导致 verifier schema 错误。

## 4. 答案格式影响 reward

reference prompt 要求最终答案放在 `\boxed{}`。普通文本里出现正确数字，不保证
`math_with_judge` 能抽取到正确答案。先运行 `verify_gsm8k.py`，把 verifier contract 与模型能力
分开。

## 5. 没有 tool call 是正常的

GSM8K 当前 graph 是 simple agent + math verifier，没有 calculator/tool sandbox。dump 中
`agent_turns` 通常为 1、`<tool_call>` 数量为 0，不是 session 断链。需要检查工具回灌时用
Workplace Assistant。

## 6. 一条 sample 不会产生有效 GRPO 学习信号

`NEMO_GYM_N_SAMPLES_PER_PROMPT=1` 时，同一 prompt 没有组内相对差异，GRPO advantage/grad norm
可能为 0。这只能验收 rollout 和训练代码能走到 step，不能证明参数在学习。正式更新至少用 4 个
sample，并确认 Actor metrics。

## 7. `/readyz` 只证明服务存活

Gateway ready 不代表数学 verifier 正确。必须额外看到：

```text
correct_reward=1.0 incorrect_reward=0.0
```

## 8. 两套 Ray 不能混

Gym 私有 Ray 使用 `:6381`，Relax Ray 通常使用 `:6379`。训练的 `RAY_ADDRESS` 永远指向 Relax
Ray；`start_gsm8k_gym.sh` 会主动清除继承的 `RAY_ADDRESS` 并启动自己的 Ray。

## 9. callback networks 必须覆盖实际 IP

`NEMO_GYM_CALLBACK_ALLOWED_NETWORKS` 接收逗号分隔的严格 CIDR，不接受 wildcard 或 `/0`。网段必须覆盖
Relax callback URL 中真实出现的 IP。代理环境还应把 Gym/Relax 内网 IP 加入 `NO_PROXY`。

## 10. 不要只看 Ray Job 状态

Ray driver 和 Actor 是不同进程。Actor OOM 或训练异常未必立即让外层 job 状态体现为失败。验收时
同时检查：

- `rollout_result/train/*.jsonl`；
- Actor traceback/OOM；
- optimizer/metrics step；
- Gateway `active_trials` 是否归零。
