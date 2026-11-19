# 磁盘增量权重同步（第一阶段）

本实现让 Megatron 完整导出 HF 权重，再与上一版 HF 权重做无损差分。接收端通过共享文件系统或 TCP 获取差分，在磁盘重建完整 HF checkpoint，最后调用 SGLang 的普通 `update_weights_from_disk` 加载。SGLang 不直接把差分写入 GPU。

这是 [RFC #410](https://github.com/redai-studio/Relax/issues/410) 的第一阶段。默认 NCCL/IPC 路径保持原行为；只有传入 `--disk-weight-sync-config` 才启用本路径。

## 当前范围

- 支持 Megatron、Dense、BF16、同步训练与推理使用独立 GPU。
- 每个 SGLang 引擎使用单张 GPU，TP/PP/DP 均为 1；当前要求关闭 CUDA graph。
- 暂不支持 fully-async、colocate、offload、MoE、MTP、LoRA、量化或动态成员。
- 本阶段仍做完整 HF 导出和完整磁盘加载，没有优化 Megatron 转换、源端汇聚或 GPU 局部更新。

## 调用过程

1. 在训练更新边界停下 optimizer，由所有训练 rank 参与完整 HF 导出。
2. 读取导出的 HF 参数，保存独立 CPU 快照，比较上一版已发布权重的原始字节。
3. 不变块复用旧内容；变化块选择稀疏位置和新值、完整字节或 XOR 后压缩。没有浮点相减再相加。
4. 先提交完整可读的版本，再查找推理引擎；发布成功与推理加载完成是两个状态。
5. 接收端下载缺失对象并校验，重建独立目录下的完整 safetensors checkpoint。
6. 固定引擎组暂停生成，普通磁盘加载并清理缓存。当前验证实现还包含完整 HF 比较、GPU 读回、参考权重重装和逐位比较，检查和生成探针通过后才返回成功。

第 6 步的完整审计会产生额外时间和磁盘开销。性能测试必须单独说明是否包含这些操作，不能将测试程序的去审计重放时间当作当前训练路径的实际同步时间。

## 训练入口

使用已有的小模型脚本，并提供包含 `Qwen3-0.6B` 的模型根目录、包含 `dapo-math-17k/dapo-math-17k.v2.jsonl` 的数据根目录，以及新的运行目录。该短验证脚本不保存常规训练或优化器 checkpoint。

```bash
mkdir -p "$SAVE_DIR"
cp task_delta_weight_sync/configs/normal-ab.json "$SAVE_DIR/sync.json"

RELAX_SKIP_CLUSTER_CLEANUP=1 \
MODEL_DIR="$MODEL_DIR" DATA_DIR="$DATA_DIR" SAVE_DIR="$SAVE_DIR" \
RAY_ADDRESS="$RAY_ADDRESS" RAY_JOBS_ADDRESS="$RAY_JOBS_ADDRESS" \
DISK_SYNC_CONFIG="$SAVE_DIR/sync.json" NUM_ROLLOUT=10 \
bash scripts/entrypoint/ray-job.sh \
  scripts/training/text/run-qwen3-0.6B-8xgpu-disk-sync.sh
```

上述 MODEL_DIR、DATA_DIR、SAVE_DIR 和配置文件路径均应使用两节点可见的绝对路径。外层入口必须设置 `RELAX_SKIP_CLUSTER_CLEANUP=1`，避免清理同集群其他作业。

配置中的相对路径以配置文件所在目录为基准。`store_root` 存放需要交付的编码对象；`canonical_root` 存放发布端完整恢复材料，必须位于前者之外。初始版本全量发布，默认每 20 版再发布一次全量；`full_only: true` 用于同一路径的全量对照。

每个新实验使用新的目录和 stream，避免复用其他训练的版本状态。脚本使用四张训练 GPU 和四张推理 GPU，并默认采用普通 GRPO 优化器参数。两节点需要能访问相应文件路径；路径相同本身不代表跨集群隔离。

## 独立发布和获取

训练进程之外也可以使用冻结的完整 HF checkpoint 发布版本。先在配置中填写实际目录和容量：

```bash
python -m relax.distributed.weight_sync --store "$STORE" publish \
  --config "$CONFIG" --checkpoint "$FROZEN_HF_CHECKPOINT" \
  --update-id "$UPDATE_ID" --result "$PUBLISHED_JSON"

python -m relax.distributed.weight_sync --store "$STORE" prepare \
  --stream "$STREAM" --version "$VERSION" \
  --cache "$CACHE" --checkpoints "$CHECKPOINTS" \
  --disk-budget-bytes 137438953472 --result "$PREPARED_JSON"
```

`prepare` 只获取和重建文件，不代表推理引擎已加载。TCP 获取时增加 `--endpoint tcp://HOST:PORT`；服务端使用 `serve --host HOST --port PORT`。当前 TCP 接口用于可信网络或已有认证隧道。

## 一致性和恢复

差分比较使用 HF 参数名、shape、dtype 和原始位模式，保留正负零及 NaN payload。全量与差分可以表示同一目标权重，但交付对象不同。接收端校验目标内容，拒绝错误 base、损坏对象和不完整 checkpoint。

新消费者、缺少 base 或损坏对象时可以请求同目标全量；生产者退出前可通过 `seal` 预先保存恢复材料。重复安装不能改变目标权重。组内加载或检查失败后保持暂停，不能以部分成功恢复生成；无法确认暂停的成员会被停止。

文件逐位比较入口如下；差异或无效文件会返回非零退出码：

```bash
python -m relax.distributed.weight_sync.tensor_diff \
  "$FULL_CHECKPOINT" "$RECONSTRUCTED_CHECKPOINT" --result "$DIFF_JSON"
```

磁盘预算与主机内存预算需要覆盖完整 HF 导出、发布端恢复材料、接收端缓存、重建文件及审计 dump。当前不自动删除正在使用或历史重建的 checkpoint 目录；发布对象的引用释放和 `gc` 不等于这些目录的清理。
