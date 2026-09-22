# student / teacher 产能平衡

colocate OPD 里 student rollout 和 teacher 占**互不重叠**的 GPU bundle，整个 rollout 阶段
两边同时驻留。谁先干完，谁的卡就在剩下的时间里 0% 空转。
这份文档讲怎么量、怎么调、以及哪些改法是**还不存在的**。

代码引用只给文件 + 符号名；判断前请实际读一眼那个符号，行为可能已经变了。

## 0. 先确认这确实是产能失衡，不是缺 overlap

不是调度 bug —— `relax/engine/rollout/sglang_rollout.py` 的 `generate_and_rm_group`
给每个 sample 建一个 asyncio task 再 `asyncio.gather`，task 内部是
`generate → rm → await teacher prefill` 串行，但 N 个 task 一次性建好，
所以 **sample i 的 teacher fetch 与 sample j 的 student 生成本来就物理重叠**。
teacher 请求还不受 student 那个 inference 信号量限流（信号量只包生成派发那一段，
在 teacher prefill 之前就释放了）。

实测参考（某 9B MOPD colocate 8 卡、多图数据集、student 4 卡 / teacher 4 卡）：

- student 空闲 = rollout step 的 **约 50%**（4 张卡半个 step 全 0%，连续采样无例外）
- teacher/student 耗时比 **约 2.1×**
- teacher 每 token 慢 **1.6–1.9×**，两个叠加因素：
  ① token 更多（teacher prefill `prompt+response`，student 只 prefill prompt）；
  ② 全词表 logits（主因，见 `teacher-knobs.md` R-T02）

这些数字是**那一个配置**的观测，不是普适常数 —— response 越短、图越少，失衡越轻。

## 1. 怎么量

**teacher 侧没有任何 metric。** 仓库里不存在 teacher 耗时 / 失败数 / 请求数的指标，
也没有 overlap ratio 之类的东西（有些早期分析文档提到过，属陈旧）。只能靠下面三条：

| 方法 | 怎么做 | 注意 |
|---|---|---|
| **GPU 占用直采** | 训练跑起来后连续 `nvidia-smi` 采样，比 rollout bundle（GPU `0..rollout_num_gpus-1`）与 teacher bundle（紧随其后）的利用率 | 最直接。bundle 布局由 `teacher_manager.py` 的 `_resolve_teacher_gpu_index` 决定 |
| **engine 日志时间窗** | 以 `Starting rollout step N` 为统一起点，取各 engine `Prefill batch` / `Decode batch` 行时间戳的**首末**作为活动窗口 | ⚠ Ray 会把大量日志行折叠成 `[repeated Nx]`，**按行求和的任何指标都不可信**；只用首末时间戳，或设 `RAY_DEDUP_LOGS=0` |
| **`perf/wait_time_ratio`** | = `perf/train_wait_time / perf/step_time`（`relax/utils/training/train_metric_utils.py`） | colocate 下 actor 在 rollout 期间睡着，所以这个比值 ≈ rollout 阶段占整步的比例。是**整体**信号，区分不出是 student 还是 teacher 拖的 |

其它有用的 metric：`perf/rollout_time`、`perf_detail/rollout/generate_time/{mean,max}`。
MOPD 还有 `rollout/by_source/<source>/{logp_gap,rkl_approx,accuracy}`
（`opd_utils.py` 的 `compute_mopd_metrics`，需要 `--opd-teacher-key`，
`--use-opd` 时会自动回填成 `data_source`）。

**⚠ 先量噪声带再量效果。** 多图 OPD 的**单步** `rollout_time` 波动可以很大
（某次实测达 ±27%），5 步合计才收敛到约 ±7%。任何 A/B 至少跑 5 步，
并先确认两臂的 `rollout/image_count/mean`、`rollout/multimodal_token_count/mean` 一致。

## 2. R-B01 — 重新分卡

- **Trigger**：一侧 GPU 长时间 0%，两侧耗时比明显偏离 1.0
- **平衡点**：设 teacher/student 工作量比 `r`，总卡 `G`，则

  ```
  teacher_gpus ≈ G·r/(1+r)      rollout_gpus ≈ G/(1+r)
  ```

  例：r=2.1、G=8 → 理论最优约 2.6 : 5.4。
- **硬约束**（`opd_utils.py` 的 `validate_managed_opd_teacher_colocate_args` 与
  `_start_managed_multi_teacher` 各校验一次）：

  ```
  rollout_num_gpus + teacher_gpus == actor_total_gpus
  ```

  再叠加两侧的整除要求：`rollout_gpus % rollout-num-gpus-per-engine == 0`，
  `teacher_gpus / num_teachers % teacher-num-gpus-per-engine == 0`。
  **可达点通常很稀疏** —— 上例在 8 卡上落不到 2.6:5.4，只能在 2:6 / 3:5 / 4:4 里挑。
- **别只算比例，要算过没过头**：上面那次实测评估的 2:6 点只把 step 压了约 9%，
  因为 2:6 **过头**了，student 反而成了新瓶颈。
  **正确做法：把可达点列出来，各自估两边耗时，取 max 最小的那个。**
- **额外代价**：student 卡变少 → KV 池变小。student 整个 decode 阶段攥着 KV，
  挤太狠会触发 Retract 重算。
- **不影响的东西**：`--rollout-batch-size` 只改绝对时长，**不改空闲占比**（两侧等比缩放）。

## 3. R-B02 — 先想想能不能把绝对工作量砍掉

调分卡是零和的（一边的卡是另一边给的），砍工作量不是。
多图场景里 `--image-max-token-num` 往往是唯一能大幅缩短绝对时间的改法
（**两侧同时降**，代价是图片有效分辨率下降，顺带缓解训练侧的 full-vocab logits 压力）。
`examples/on_policy_distillation/mopd/` 下的脚本里有完整的取舍推导注释可参考。

判据（多图 MOPD 多臂 A/B 的教训）：**只有卡在关键路径上的浪费才值钱。**
按「消除了多少重复工作」排序会排错 —— 有一项把 CPU 预处理砍掉 7/8，`step_time` 收益是 **0**，
因为瓶颈在 engine 内部，Relax 侧早点做完只是让请求更快堆在 engine 门口。
先问「这在关键路径上吗」，再问「这浪费了多少」。

## 4. R-B03 — 「student 跑完就地 offload、原地起 teacher」

**判断这个之前先自己复核一遍**，下面是写这份文档时的状态：**没有实现**。
不是部分实现，是不存在。原语齐了，但没有任何东西把它们接起来。

现有的 offload 是 **train ↔ rollout 的 step 级** lock-step，teacher 和 student rollout
在整个 rollout 阶段**同时驻留**在各自的 bundle 上：

| 时机 | 动作 | 位置 |
|---|---|---|
| 启动、actor init 前 | teacher `offload()` | `opd_utils.py` 的 `maybe_start_managed_opd_teacher` / `_start_managed_multi_teacher` |
| `train()` 开头（rank 0） | teacher `offload()`，在 actor `wake_up()` **之前** | `relax/backends/megatron/actor.py`，经 `append_managed_opd_teacher_offload_handle` |
| 紧接着 | gloo `barrier` 把所有 rank 挡在 rank-0 的 offload 之后，防 `cuMemCreate` OOM | 同上 |
| `update_weights()` 里（rank 0） | teacher `onload()` 全量恢复，与 rollout 的 `onload_weights()` 同批 | 同上，经 `append_managed_opd_teacher_onload_handle` |

缺的东西（用 grep 复核这四条是否仍然成立）：

- **rollout 阶段内没有「学生全部跑完」这个相位边界**可挂钩子 ——
  rollout 的屏障是一个 `asyncio.gather`，每个 task 自带尾部的 teacher prefill。
- **rollout 路径里没有任何 `teacher_manager.offload/onload` 调用**
  （grep `teacher_manager` 的 offload/onload 调用点，应当只在上表那几处）。
- **teacher 没有分阶段 resume**：`TeacherManager.onload(tags)` 支持 tags，
  但所有调用方都传空；没有 student 那样的 `onload_weights` / `onload_kv` 包装
  （对照 `relax/distributed/ray/rollout.py`）。
- **卡的分区是结构性写死的**：`_resolve_teacher_gpu_index` 把 teacher 放在
  rollout 之后，两处校验硬要求 `rollout + teacher == actor`。
  做成时分复用必须把它放宽成 `max(rollout, teacher) <= actor`。

如果要提这个方案，必须一并说清四个拦路石：

1. 上面那条分区校验要改（两处）。
2. `SGLangEngine.release_memory_occupation` 会先 `flush_cache()` ——
   rollout 中途 offload student 会**摧毁 student 的 radix cache**，
   只对单轮 rollout 安全，多轮 / agentic 会付出重新 prefill 的代价。
3. teacher 的 `mem_fraction_static` 现在是按「共存」调的。
   时分复用之后 teacher 可以独占整张卡 —— **真正的收益在这里**，不在回收那点空闲。
4. 那个 gloo barrier 的存在本身说明「teacher 驻留 + actor 唤醒」并发会 `cuMemCreate` OOM。
   per-rollout 的切换需要同样的 barrier 纪律，但 rollout 跑在单个 Ray actor 的
   asyncio loop 里，**目前没有可挂的集合通信**。

## 5. R-B04 — fully-async

colocate 下 teacher 与训练共卡，无法重叠；fully-async 模式下 teacher 可以与训练重叠。
但注意两条限制（在 `relax/utils/arguments.py` 与 `opd_utils.py` 里各有一处显式拒绝）：
`--opd-type=megatron` 在 fully-async 下不支持；
MOPD（`--opd-teacher-routes`）**只支持 colocate**。
