# Train Actor profiling 与性能分析

适用于 Megatron Train Actor。命令从仓库根目录执行；采集在用户已有的 GPU 集群上完成，离线处理只需 Python 标准库。先核对当前源码，不把一次实验的 rank 编号、层布局或瓶颈当作其他任务的默认配置。

## 1. 开启当前已接入的采集路径

在实际使用的 `scripts/training/` 启动脚本中加入以下参数（已有同名参数时修改原值）。若使用数组，确保最终训练命令展开了 `"${PROFILE_ARGS[@]}"`：

```bash
PROFILE_ARGS=(
   --use-pytorch-profiler
   --profile-target train_overall
   --profile-step-start 3
   --profile-step-end 3
   --profile-with-stack
)
```

这是一次短窗口栈诊断示例，不是所有任务的最佳采样位置。`--profile-with-stack` 默认关闭，定位 Python 调用来源时开启；`--profile-with-memory`、`--profile-with-flops` 也默认关闭，按需添加。`record_shapes=True` 已在实现中启用。采集开销会影响耗时，性能收益需用关闭 profiler 的同条件运行验证。

核对已有 `--tb-experiment-name`，为该次采集使用可辨认且不与旧任务混淆的名称。获用户授权启动实验后，在已确认的远端环境用现有入口提交，不在本地 Mac 上执行：

```bash
bash scripts/entrypoint/ray-job.sh scripts/training/<实际启动脚本>.sh
```

入口会清理旧 worker / 任务，不能为了采集擅自在正在使用的集群执行。这里只提供启动方式，不自动提交任务。

### target 与 step 的真实含义

- `train_overall` 是默认 target，`TrainProfiler.on_init_end()` 启动 profiler，Actor 的 `self.prof.step(...)` 推进采样。它可以覆盖供数、log-prob、训练和同步，不能把整个窗口都算作 forward/backward。
- `train_actor`、`train_log_probs` 是合法选项，且定义了 iterator 包装器，但当前仓库没有调用 `iterate_train_actor` / `iterate_train_log_probs` 的位置。不要推荐单独选择它们来采集当前 Megatron 训练；以后若接入，重新核对推进粒度。
- `profile_step_start/end` 控制 profiler 的本地推进次数，end 包含在内；不是 microbatch 编号，也不保证等于业务 rollout_id 或 optimizer step。不同 Actor 路径的调用点不同，log-prob 路径也调用 `self.prof.step`。
- 当前 schedule：`wait=max(start-1, 0)`、`warmup=1 if start>0 else 0`、`active=end-start+1`、`repeat=1`。例中 #0/#1 等待、#2 预热、#3 记录；需运行到该 active 区间结束后的 `step()`，才能触发导出。确保 `0 <= start <= end` 且任务有足够的推进次数。
- `_create_torch_profiler` 没有 rank 筛选判断；不要假设上游同名 profiler 的 rank 参数在这里生效。多 rank trace 从实际输出中选择。

依据：[profile_utils.py](../../../relax/utils/profile_utils.py) 的 `TrainProfiler` / `_create_torch_profiler`，[actor.py](../../../relax/backends/megatron/actor.py) 的 `self.prof` 调用点，[参数定义](../../../relax/utils/arguments.py) 的 `--profile-target`；总开关和 step 参数由 [Megatron 参数解析器](../../../relax/backends/megatron/arguments.py) 引入，已有 [训练脚本示例](../../../scripts/training/text/run-qwen3-30B-A3B-8xgpu.sh)。

## 2. 找到并收集原始 trace

输出在 **worker 工作目录** 下的 `traces/<tb-experiment-name>/train_trace/`；没有 experiment name 时使用时间戳。`--tensorboard-dir` 不决定这条路径。文件名形如：

```text
train_overall_rank0_dp0_tp0_pp0.<timestamp>.pt.trace.json.gz
```

用每个节点实际的 worker 路径或已有共享目录收集到本地同一目录，保留原文件名。不要假设文件都在 head 节点或仓库根目录。PP 分析先收齐同一次采集、同一 DP/TP 切片的所有 PP stage；用文件名、`distributedInfo` 和训练拓扑核对 rank，不能固定按 rank/16 推算。怀疑 TP/DP 不均衡时再扩展到对应组成员。

## 3. 修复、筛选并默认合并

使用仓库脚本 [scripts/profile/fix_trace_stacks.py](../../../scripts/profile/fix_trace_stacks.py)：

```bash
python3 scripts/profile/fix_trace_stacks.py tmp/profile-input/*.trace.json.gz
```

**无需加 `--merge`。** 默认写出每个输入旁的 `fixed/<原名>.fixed-stacks.trace.json.gz`，以及第一个输入目录下的 `fixed/merged.trace.json.gz`。只有更换合并输出路径才使用：

```bash
python3 scripts/profile/fix_trace_stacks.py tmp/profile-input/*.trace.json.gz \
  --merge tmp/profile-merged.json.gz
```

- 输入是原始 gzip JSON 文件列表，不是目录或 tar 包；不要把不同实验、旧 merged 文件或 `fixed` 文件再次加入输入。原始输入不改写，已有同名输出会被完整新输出替换。
- 按 `process_name` 保留 `MegatronTrainRayActor.train` / `ray::MegatronTrainRayActor.train` 进程，保留其内部 Python 栈、CPU 算子和 GPU 活动；它不是按某个同名 Python slice 筛选。找不到目标进程会报错，应检查实际采集的 Actor 方法名。
- 修复越过父栈结束时间的 Python duration；只修复可视化层级，不代表原始 CPU 函数实际耗时已被准确恢复。GPU kernel duration 不因此改变。
- **合并文件**左侧按 CPU、GPU stream 数值分组，组内按 rank 数值排序，各 rank / 原始线程仍隔离。单 rank 的 fixed 文件保持原轨道布局。
- 按 `baseTimeNanoseconds` 对齐，保留相对时间差并隔离事件 ID；不能校正跨机器时钟偏差。脚本流式处理大文件，但元数据预扫描和合并需要额外遍历。
- 在可读取 Chrome trace JSON 的查看器中打开合并结果；若查看器不接受 gzip，可用 `gzip -dc tmp/profile-input/fixed/merged.trace.json.gz > tmp/profile-merged.json`。保留原 trace 用于定量复核。

## 4. 从现象到证据

1. **固定比较窗口**：列出源文件、step、时间范围、PP/TP/DP rank、GPU pid 和 stream；确认各 stage 的 F/B 次数相同、采样完整。CPU annotation 与异步 GPU 执行的边界可能不同，统计窗口内区间交集，并用完整 F/B 或整份 trace 复核边界效应。
2. **拆开计算与通信**：`ph="X", cat="kernel"` 中按名称识别 NCCL，另列 memcpy/memset；annotation（例如 `ProfilerStep#3`）不是持续计算。计算每个 `(pid, tid)` 的区间并集，跨 stream 求总忙碌时间也要做并集，不能把重叠 duration 相加。名称排除 NCCL 只是“非 NCCL kernel”口径，仍可能含 copy、optimizer 等，并非纯模型 FLOPs。单 stream 时间不是整卡利用率/MFU。
3. **算子问题还是负载问题**：按 kernel 名称统计调用次数、总耗时、平均/分位耗时，再用 CPU correlation / External id 和父栈核对 shape、所属层及重计算。总时间翻倍但次数翻倍、单次耗时相近，支持执行工作量更多；相同 shape 和调用次数下单次明显更慢，才支持 kernel 效率、硬件状态或资源争用问题。仅有 stage 总时间差不能断定分层有错。
4. **PP schedule**：同时看所有 stage 的 warmup、稳态、cooldown。标准非交错 1F1B 的 warmup 深度为 `min(M, PP-1-pp_rank)`；首个 backward 从后向前返回、后段先结束均属预期。均衡且忽略通信时气泡占比约 `(PP-1)/(M+PP-1)`，这是理论估算。M 是每条 pipeline 的 microbatch 数，动态 packing 时从实际调用/配置核对，不直接拿全局样本数代替。正常依赖等待不能作为 schedule bug 的证据。
5. **长 NCCL**：检查 CPU 发起时间、GPU 执行区间、group/参与 rank、张量大小和结束时刻。后段提前进入、各 rank 同时结束的 collective 往往包含等迟到 peer 的时间。尾部检查 `finalize_model_grads`、共享 embedding 梯度同步、token-count broadcast 的父栈；不要仅凭 broadcast 字样推断 PP 激活通信或带宽不足。
6. **数据问题**：区分上游取数阻塞、microbatch token/shape 差异、模型 stage 工作量不同。关联 `get_batch` / 数据迭代器、CPU launch 间隙、token/shape 和 GPU 时间，不能把所有空白称为“数据负载不均”。

报告给出证据表和可复现过滤条件，把“已测量”“推断”“待验证”分开。建议按瓶颈选择一个变量做 A/B：microbatch 数、PP 层布局、供数或特定算子；核对 batch 语义、显存和版本约束后再实施。不要未经测量把某次 PP8 实验的比例或优化收益推广到其他任务。
