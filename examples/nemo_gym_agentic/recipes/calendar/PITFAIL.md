# Calendar 接入踩坑

01. **必须交付可直接执行的脚本。** Calendar 训练脚本应像 R2E Gym 一样直接包含训练参数，不能再包装一层通用训练启动脚本，让用户追踪多层 `exec`。

02. **训练参数直接写在训练脚本里。** `num-rollout`、`n-samples-per-prompt`、batch size、context 长度和 parser 等参数不能再包装成 `NEMO_GYM_NUM_ROLLOUT` 一类环境变量。环境变量只用于机器、端口和路径配置。

03. **所有部署变量统一写入 `env.sh`。** 至少包括 `NEMO_GYM_IMAGE`、`MODEL_DIR`、`GYM_HOST`、`NEMO_GYM_GATEWAY_PORT` 和 `NEMO_GYM_SOURCE_DATA`，不能要求用户每一步重新拼环境变量。

04. **数据准备脚本必须直接生成训练脚本检查的文件。** `prepare_calendar.sh` 执行结束后，`${NEMO_GYM_SOURCE_DATA}` 必须真实存在，并且位于所有远程 Ray 节点可见的共享文件系统。只生成在 Docker volume 中没有意义。

05. **不能把 smoke 配置当正式默认值。** 之前默认只转换 32 条数据并输出 `calendar_smoke.jsonl`，导致正式训练只使用 32 条。现在默认使用 `calendar_train.jsonl` 的全部 3872 条数据。

06. **不要让两次数据转换产生歧义。** prepare 阶段保存的是 NeMo Gym raw 数据和用于检查的 `_relax.jsonl`；训练阶段会从 raw 数据生成 `${EXP_DIR}/data/calendar_train.jsonl`。日志里的 `Converted ...` 不是重新下载数据。训练源始终应传 raw `calendar_train.jsonl`，不能传 `_relax.jsonl`。

07. **禁止默认 `--no-wait`。** 只有显式设置 `RAY_NO_WAIT=1` 时才允许异步提交；默认启动必须等待并持续输出训练日志。

08. **不要混淆两个 Ray。** `RAY_ADDRESS` 只指向远程 Relax 训练集群；NeMo Gym Docker 使用自己的私有 Ray。`NEMO_GYM_CALLBACK_ALLOWED_NETWORKS` 是 callback 白名单，不能代替 `RAY_ADDRESS`。

09. **启动端口和训练端口必须完全一致。** `CALENDAR_PORT_BASE` 决定 Gym Gateway 端口，`NEMO_GYM_GATEWAY_PORT` 决定训练访问端口。本次实际使用 29103；只修改其中一个会直接访问错误端口。

10. **修改镜像后必须重建并替换运行中的容器。** 复用相同 image tag 不会更新已经启动的 `nemo-gym-calendar`；每次修改 NeMo Gym 集成代码后都要重建镜像、重建该容器并重新检查 `/readyz`。

11. **Qwen3 必须配置 reasoning parser。** Calendar grader 看到 `<think>` 会直接返回 0，即使后面的 JSON 正确。训练脚本必须包含：

    ```bash
    --agentic-reasoning-parser qwen3
    ```

12. **自然语言回答得到 0 reward 是正常的。** Calendar 要求最终回答包含完整 JSON list；“calendar is all set”之类的自然语言回答不满足 verifier。

13. **HTTP 500 必须有 traceback。** NeMo Gym 原实现会把 inner `ClientResponseError` 转成 500 而不打印栈，Gateway 也曾只记录 `agent_error`。当前镜像已补 traceback，排障时直接查看：

    ```bash
    docker logs -f nemo-gym-calendar
    ```

14. **本地 verifier 通过不等于训练链路通过。** 最终验收必须看到真实 rollout 完成、出现非零 reward、执行 optimizer step，并确认 Gateway 的 `active_trials` 回到 0。
