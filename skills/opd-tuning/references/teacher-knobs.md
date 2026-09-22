# OPD teacher 调参规则目录

每条规则字段：`Trigger` / `Why` / `Fix` / `Verify` / `Skip when`。
引用代码只给**文件 + 符号名**，不给行号（行号会随开发漂移）。
所有默认值都标了「自行复核」—— 用下面的自验证配方现查，不要相信本文档里的数字过期与否。

## 自验证配方（先会用这三条，再看规则）

**① 定位运行时真正 import 的那份 sglang**（镜像里常常躺着不止一份源码副本，
只信这条命令的输出）：

```bash
SGL=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
python3 -c "import sglang; print(sglang.__version__)"
```

**② 确认某个 sglang flag 是否存在 / 它的默认值与合法取值。**
新版 sglang 的 flag 由 `ServerArgs` dataclass 自动生成，
**grep `"--flag-name"` 会假阴性** —— 必须构建真 parser：

```bash
python3 -c "
import argparse
from sglang.srt.server_args import ServerArgs
p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p)
for a in p._actions:
    if any('cuda-graph' in s for s in a.option_strings):   # 换成你要查的关键词
        print(a.option_strings, 'default=', a.default, 'choices=', a.choices)
"
```

**③ 确认 Relax 镜像出来的 `--teacher-sglang-*` 形式真的存在。**
Relax 把 `ServerArgs` 的每个 flag 加上 `--teacher-sglang-` 前缀重新注册
（`relax/utils/opd/opd_utils.py` 的 `_mirror_teacher_sglang_server_args`），
但有一张跳过名单（`_SGLANG_PASSTHROUGH_SKIP_ARGS`，含 `tp_size` / `enable_memory_saver` /
`model_path` / `port` / `base_gpu_id` 等），名单里的**不会**被镜像出来：

```bash
python3 -c "
import argparse
from relax.utils.opd.opd_utils import _mirror_teacher_sglang_server_args
p = argparse.ArgumentParser(add_help=False); _mirror_teacher_sglang_server_args(p)
for a in p._actions:
    if any('cuda-graph' in s for s in a.option_strings):
        print(a.option_strings, 'dest=', a.dest, 'choices=', a.choices)
"
```

**④ sglang 的 env var 默认值**都集中在 `srt/environ.py`：

```bash
grep -n "SGLANG_VLM_CACHE_SIZE_MB\|LOGITS_PROCESSER" "$SGL/srt/environ.py"
```

---

## 因果链（先看这个，再看规则）

teacher 请求是 `max_new_tokens=0` 的**纯 prefill**
（payload 构造见 `relax/utils/opd/opd_main_worker.py`），
`input_ids = prompt + response`、`logprob_start_len = prompt_len - 1`
（`relax/engine/rollout/on_policy_distillation.py` 的 `OpdManager`）。由此推出整条链：

```
纯 prefill ─┬─► 请求一完 KV 立刻释放，KV 峰值 = 在飞请求
            │      → mem-fraction-static 可以压低                    [R-T01]
            │
            ├─► decode CUDA graph 永远不会被 replay
            │      → 关掉是纯赚；prefill graph 另说                   [R-T05]
            │
            └─► 每个 response 位置都要 logprob
                   → [N_positions, vocab/TP] fp32 是真正的显存主导项  [R-T02]
                   → 先开 logits chunking 解耦                        [R-T03]
                   → 解耦之后才敢拉大 chunk / prefill token 预算      [R-T04]
```

**顺序不能反。** R-T04 在 R-T03 之前做 = OOM。

---

## R-T01 — teacher `mem-fraction-static` 偏高

- **Trigger**：`--teacher-sglang-mem-fraction-static` ≥ 0.7，或与同脚本 student 的
  `--sglang-mem-fraction-static` 取同一个值
- **Why**：student 整个 decode 阶段攥着 KV，teacher 请求一结束 KV 立刻释放。
  两者方向**相反**，抄同一个数一定有一边错。
  teacher 这边把 KV 池撑大只会把显存从 logits gather / ViT / attention workspace 手里抢走。
  （某次多图实测参考：student `full token usage` 峰值 0.63–0.67，teacher 0.00–0.10。）
- **Fix**：**算，别抄。** KV 需求上界 ≈
  `max_running_requests × 单请求 token 数 × per-token KV`，
  其中 `per-token KV = 2 × layers × kv_heads × head_dim × dtype_bytes`（从 `config.json` 取，
  注意 `kv_heads < TP` 时 KV 头是复制而非切分）。
  再对照 `mem_fraction_static × 卡容量 − 权重/TP` 得到的池子大小，留 3–5× 冗余就够。
- **Verify**：teacher engine 日志里的 KV pool 大小与 `full token usage` 峰值。
  持续远低于 1.0 说明还能降。
- **Skip when**：teacher 在独立 PG（非 colocate）且那些卡本来就空 —— 压低没有受益方，
  只是把显存闲置。memory saver 也只在 colocate 打开
  （`opd_utils.py` 的 `build_teacher_overrides`：`enable_memory_saver = colocate_sync`）。
- **注意**：这条经常"触发但不值钱"。没有 OOM 压力时收益接近零，别机械改。

## R-T02 — teacher TP 由 logits 显存决定，不是由模型大小决定

- **Trigger**：teacher OOM 且栈顶在 `logits[input_logprob_indices]` / `logits_processor`；
  或给 teacher 配 TP 时只按"权重塞得下"算
- **Why**：普通推理只要最后一个位置的 logits，OPD teacher 要**每个 response 位置**的。
  sglang 的 `logits_processor` 默认一次性 materialize `[N_positions, vocab/TP]` fp32。
- **算法**：

  ```
  logits 峰值 = N_positions × (vocab_size / TP) × 4 B
  ```

  `vocab_size` 从 `config.json` 取（多模态模型在 `text_config` 下）。
  `N_positions` 受 `chunked-prefill-size` 约束 —— 这就是 R-T04 的耦合点。

  实测样例（某 9B 教师，vocab≈2.5e5，单卡约 95 GiB）：
  chunk 16384 + TP1（约 15 GiB 单次分配）→ CUDA OOM；同配置 TP2 → 跑通。
- **Fix**：抬 TP，或者开 R-T03 把这一项压到与 chunk 无关。
  **优先 R-T03** —— 抬 TP 是拿通信换显存，开 chunking 是白给。
- **Skip when**：已开 R-T03 且 chunk size 已压住峰值。

## R-T03 — logits chunking 没开

- **Trigger**：脚本 env 块里没有 `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1`，
  **且** 计划把 chunk 拉到让 R-T02 的峰值超出预算
- **Why**：它把 R-T02 的一次性 materialize 换成分块处理
  （sglang `srt/layers/logits_processor.py` 的 `process_input_logprobs_by_chunk`），
  峰值降到 `CHUNK_SIZE × vocab/TP × 4B`，**与 prefill chunk size 解耦**。
  这是 R-T04 的前置条件。
- **Fix**：

  ```bash
  export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
  export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=8192
  ```

  ⚠ **拼写是 `PROCESSER` 不是 `PROCESSOR`**（sglang 上游拼错了；写对反而无效）。
  用配方 ④ 复核这两个名字和默认值仍然存在。
  两个变量都必须进 `RELAX_PROPAGATE_ENV_VARS`（见 R-T09）。
- **CHUNK_SIZE 怎么选**：它**直接**决定 logits 峰值 —— 把 R-T02 公式里的 `N_positions`
  换成 `CHUNK_SIZE` 即可。显存紧就往回调，代价是分块循环次数变多。
- **Verify（三个条件，缺一就静默走老路径）**：读
  `logits_processor.py` 里 `should_skip_chunking` 的判断，通常是
  ① env 为真；② `positions > CHUNK_SIZE`（**短序列本来就不分块，正常**）；
  ③ DP-attention all-gather 没开。OOM 时先逐条确认这三点。
- **Skip when**：序列短到 R-T02 的峰值本来就不成问题 —— 此时它是无害但也无用的保险。

## R-T04 — `chunked-prefill-size` / `max-prefill-tokens` 太小

- **Trigger**：单请求 token 数（`prompt + response`）远大于 `chunked-prefill-size`
- **Why**：单请求远超每轮 chunk 预算时，sglang 的 `PrefillAdder` 每轮只收一条序列、
  抽干本轮预算就 break。请求被切成 `ceil(tokens / chunk)` 趟，每趟的 matmul 都太小，
  纯粹浪费 kernel 效率。
- **Fix**：两个一起抬，且 `max-prefill-tokens ≥ chunked-prefill-size`：

  ```
  --teacher-sglang-chunked-prefill-size <≥ 单请求 token 数>
  --teacher-sglang-max-prefill-tokens   <同上>
  ```

  `--max-prefill-tokens` 有自己的默认上限（用配方 ② 查当前值），不抬它就是新的天花板。
- **前置**：先用 R-T02 公式算新 chunk 下的 logits 峰值。超预算就先做 R-T03，
  否则 chunk 变大 = logits 张量变大 = 直接 OOM。
- **Skip when**：单请求 token 数本来就小于 chunk —— 抬了没意义。

## R-T05 — CUDA graph：decode 该关，prefill 不要盲关

- **Trigger**：脚本用 `--teacher-sglang-disable-cuda-graph`
- **Why**：
  - 较新的 sglang 里 `--disable-cuda-graph` 已被**弃用**，取而代之的是**按 phase 分开**的
    backend 开关（decode / prefill 各一个，取值含 `disabled`）。旧 flag 通常还留着兼容
    shim，能用但会打告警。**用配方 ②/③ 确认当前版本的确切写法与合法取值。**
  - 较新的 sglang **默认同时捕获 decode 图和 prefill 图**。teacher 永不 decode，
    decode 图捕了没人 replay → **关掉是纯赚**（省启动时间 + capture 显存）。
  - 但 prefill 图 teacher 是**会**用的，一刀切关掉可能损失吞吐。
  - colocate 下 `enable_memory_saver` 会强制关掉 prefill 图
    （见 `relax/backends/sglang/sglang_engine.py` 里组装 teacher server args 的那段），
    所以 colocate teacher 上旧 flag 实际只多关了 decode —— **语义没错，只是写法过时**。
- **Fix**：只关 decode phase，prefill 保持默认；只有实测更快时才关 prefill。
- **Verify**：teacher 启动日志 grep `Capture target decode CUDA graph end` /
  `Capture target prefill CUDA graph end` —— 这两行自带 `elapsed=... s, mem usage=... GB`，
  **是该模型上的真实节省额**，别用经验值。也可能看到
  `Disable prefill CUDA graph because ...`（某些层类型会自动关）。
- **⚠ 省下来的显存不会自动变成 KV**：`mem_fraction_static` 一旦显式指定，
  sglang 那套按 cuda-graph 预留显存的启发式整个不跑，关图省下的显存只是闲置。
  想变成 KV 必须**同时**上调 `mem-fraction-static`。另外大显存卡通常还有一个保留下限，
  常把 cuda-graph 那几百 MB 直接吞掉 —— 所以别承诺固定收益。
- **Skip when**：无 —— 但按 Verify 的日志读数说话。

## R-T06 — 副本数 vs TP

- **Trigger**：在「1 副本大 TP」和「多副本小 TP」之间做选择
- **Why（实测）**：teacher 是 **compute-bound**，副本数不增加 FLOPs。
  同卡数下 TP2×2副本 vs TP1×4副本 A/B **完全中性**。
- **高 TP 的真实好处**：单请求延迟下降 → `--opd-teacher-timeout-s` 是**每请求**超时、
  步时间取决于最慢那条，尾部更安全；以及 R-T02 的 logits 分片。
- **⚠ 不要用「cache 命中率」当理由，先去看代码。**
  历史上确实有过「多副本按 round-robin 打散同 prompt 的样本、毁掉 radix 前缀复用」的阶段，
  但 `on_policy_distillation.py` 的 `_pick_replica` 后来加了
  **按 `(teacher_key, group_index)` 的 group 亲和记忆** —— 同一 prompt 的 n 个样本落同一副本，
  前缀只 prefill 一次。**判断前先读这个函数确认亲和逻辑还在**，
  也不要引用早于该改动的分析文档。
- **多副本的好处**：每个 engine 只有一个 scheduler、每轮 prefill 预算有界，
  多副本 = 多个独立 scheduler。这一条目前**没有 A/B 支撑**，只是设计论证。
- **Fix**：吞吐上中性 → 按**显存和尾延迟**选。teacher 大 / logits 峰值紧 / 超时告警多
  → 抬 TP；teacher 小且塞得下、想要多 scheduler → 多副本。定不下来就抬 TP（下限更明确）。
  `replicas = teacher_gpus / num_teachers / --teacher-num-gpus-per-engine`
  （推导在 `opd_utils.py` 的 `maybe_start_managed_opd_teacher` / `_start_managed_multi_teacher`）。
- **Skip when**：`--teacher-num-gpus-per-engine` 不写 = 1 副本吃满 teacher 卡数，
  小规模下这个默认通常就够 —— 但建议写显式，否则改 teacher 卡数时布局会静默漂移。

## R-T07 — `--opd-teacher-timeout-s` 与「没有重试」

- **Trigger**：用默认值，或长上下文 / 多图场景下设得偏小
- **Why**：CLI 默认值很小（用 `--help` 复核），而一条多图 teacher logprob 要重新 prefill
  整个上下文。更关键的是 **`OpdManager._post_logprob` 不重试**：一次异常就返回 `None`。
  这是 `aiohttp.ClientTimeout(total=...)`，connect+read+body 全算在内。
- **失败语义**（不知道会误判）：
  - 部分失败 = **静默**，这些样本没有 teacher logprob，只在日志里出现
    `OPD ... failed: status=` / `Teacher log-prob length mismatch`；
  - 只有一个 batch 里**所有**非空样本都失败才抛错（`_raise_if_all_failed`）。
- **Fix**：按「最慢那条请求」定，不是按平均。同时把 teacher 日志里的 OPD failure
  计入检查项 —— **没有任何 metric 会告诉你这件事**。
- **Skip when**：短文本 OPD 且实测 p99 远低于默认值。

## R-T08 — 多图 VLM：`SGLANG_VLM_CACHE_SIZE_MB`

- **Trigger**：多模态 OPD + 数据集多图/大图，且该变量未设或停在 sglang 默认值
- **Why**：这是 sglang 的 ViT 输出（embedding）缓存，默认值很小（配方 ④ 复核）。
  单图占用 ≈ `image_token_num × vision_out_hidden_size × dtype_bytes`
  （`out_hidden_size` 在 `config.json` 的 `vision_config` 下）。
  **先算这个数再和 cache 上限比** —— 一旦单图就超过上限，命中率是结构性的 0，
  LRU 全程 thrash。
- **实测参考**：某多图 MOPD 场景把它从默认调到 1 GiB，单独一项
  `rollout_time` −35.4%、`step_time` −13.5%；同批 A/B 里其它候选项（dp-encoder、
  预处理去重）都测出 0。
- **Fix**：调到至少能装下"一波在飞请求"的图。记得进 `RELAX_PROPAGATE_ENV_VARS`（R-T09）。
- **代价**：cache 是 module global 里的 GPU 张量，engine sleep 时**没人释放**，
  colocate 下训练期间一直占着。
- **Skip when**：纯文本 OPD；或单图远小于默认上限的小图数据集。

## R-T09 — 环境变量没真的传到 engine

- **Trigger**：任何 `SGLANG_*` / `RELAX_OPD_*` 被 `export` 了但没进
  `RELAX_PROPAGATE_ENV_VARS`；或用 `ray job submit` 提交却没把它塞进 runtime env
- **Why**：Relax 给每个 actor 重建 runtime env，只把 `RELAX_PROPAGATE_ENV_VARS` 里
  **点名**的变量拷进去（`relax/utils/utils.py` 的 `post_process_env`，逗号分隔）。
  两个静默失败点：名字已存在于目标 env 里会被跳过；**driver 上没 export 那个名字也会被
  静默跳过** —— 只 export 名单不 export 值 = 无声无效。
  用 `ray job submit` 时 driver **不继承提交 shell**，所以还要额外并进 runtime env JSON
  （入口脚本里的那个 JSON 是**显式白名单**，不是全量透传）。
- **Fix**：用仓库里那个保序追加惯用法，两步都做：

  ```bash
  export FOO=1
  export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}FOO"
  # ray job submit 的脚本还要把 FOO 合并进 runtime-env JSON
  ```
- **Verify**：`grep "Ray runtime env" <log>` —— 要看 **actor** 那一行（job env 会被覆盖）。
  OPD patch 类的变量还可以直接看 engine 启动日志里那行
  `Launching SGLang server with independently-gated patches: ...`。
- **Skip when**：`TeacherManager` 硬编码转发到 teacher engine 的那几个变量，
  这一跳不用再列 —— 但它们仍然要先到达 driver，`ray job submit` 那层照样要管。

## R-T10 — `--teacher-sglang-max-running-requests` 与客户端并发

- **Trigger**：服务端上限与实际在飞请求数严重不匹配
- **Why**：客户端侧连接池上限写在 `on_policy_distillation.py` 里
  （`opd_teacher_connector_limit`）—— **它只有 `getattr` 读取、没有注册成 CLI flag，
  实际改不了**。所以服务端这个数是唯一能调的闸。另外每次 `prefill()` 都新建
  `ClientSession`，连接池是短命的。
- **Fix**：长上下文 / 大 mm 请求（单条就吃满 chunk 预算）→ 调**小**，
  让 scheduler 别把一堆半成品序列同时挂着；短请求 → 调大。
  与 R-T04 一起看：`max-running-requests × 单请求 token` 撑不满 `chunked-prefill-size`
  是浪费，远超则只是排队。
- **Skip when**：teacher 日志里 `#running-req` 显示实际在飞数远低于设定值 —— 瓶颈不在这。
