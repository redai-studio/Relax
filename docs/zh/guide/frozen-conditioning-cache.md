# 冻结条件缓存

图像 SFT 可在有效输入相同时复用冻结编码器的确定性特征。SenseNova 复用理解分支前缀 K/V 和图像索引，Qwen-Image 复用文本特征及确定性的 VAE 后验分布参数，两者共用有界的节点本地缓存。后验采样、扩散噪声、timestep 及可训练分支仍在线执行。本期不接入 RL。

## 启动

按 [SenseNova 图像编辑 SFT](./i2i-sft.md) 或 [Qwen-Image 文生图 SFT](./t2i-sft.md) 准备模型与数据。设置 worker 可见的 `MODEL_PATH`、`DATA_ROOT`、`METADATA_PATH`、`DATASET_SIZE`、`SAVE_DIR` 和已有集群的 `RAY_DASHBOARD_ADDRESS`。`DATASET_SIZE` 必须等于元数据条数。

```bash
RELAX_SKIP_CLUSTER_CLEANUP=1 \
RELAX_CONDITION_CACHE=1 RELAX_CONDITION_CACHE_MAX_BYTES=4294967296 \
bash scripts/entrypoint/ray-job.sh \
    scripts/training/diffusion/run-sensenova-i2i-sft-8xgpu.sh
```

Qwen 使用 `scripts/training/diffusion/run-qwen-image-sft.sh`，设置相同的缓存变量即可。两者均使用普通训练入口，无需安装验收工具或设置测量目录。`RELAX_CONDITION_CACHE=0` 恢复在线编码。缓存适合冻结条件重复出现的训练；低复用负载仍承担查找和构建缓存的开销，可能变慢。

## 配置

两份普通训练脚本会将下列变量传入 Ray worker。自定义启动脚本需将它们加入 `RUNTIME_ENV_JSON.env_vars`。

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RELAX_CONDITION_CACHE` | `0` | 启用复用；`0` 保留在线编码。 |
| `RELAX_CONDITION_CACHE_MAX_BYTES` | 8 GiB | 每节点 payload 上限，启用缓存时须为正。 |
| `RELAX_CONDITION_CACHE_DIR` | `/dev/shm/relax-conditioning` | 可写节点本地根目录，也支持本地磁盘。 |
| `RELAX_CONDITION_PREFETCH_DEPTH` | `0` | SenseNova 条件预取条目数，要求同时开启图像预取；Qwen 须为 `0`。 |
| `RELAX_CONDITION_PREFETCH_MAX_BYTES` | 512 MiB | 每 worker 的预取预留和借用字节上限。 |

## 复用与失效

键覆盖 checkpoint/tokenizer 文件内容、适配层 schema、有效输入内容、预处理、dtype 和影响输出的设置，不按文件名复用。随机裁剪、增强和 caption dropout 保持执行与随机数流顺序，使用实际结果构键。Qwen 将实际图像像素和 VAE 设置纳入键；缓存后验分布参数时保留 dtype 和影响采样的内存布局，后验采样仍按原随机数顺序每次在线执行。在线与缓存路径均将冻结 VAE 保留为副本，避免重复聚合权重。后验复用支持 contiguous 与 channels-last-3d 布局，其他布局需关闭缓存。解冻副本编码器需重建其 FSDP 所有权。

适配器声明冻结参数边界；参数替换、正常原地修改、解冻、buffer 或语义配置变化后停用复用并回退在线。SenseNova 检查 RoPE 等条件状态，同时允许确定性的懒初始化。生成侧 LoRA 更新不影响与它无关的冻结条件。状态检查位于 FSDP forward 外，避免将临时参数视图误判为替换。该防护不能识别 `.data` 等绕过版本计数器的写入；运行期间 checkpoint 文件也须保持不变。

跳过编码器前，参数分片组全部 rank 必须一致同意。任一 rank 未命中或恢复失败，该 microstep 全组在线编码，保持 collective 一致。文件缺失、checksum 失败和 schema/shape 错误会使对应条目失效并回退。可训练条件保留 autograd 图，不缓存。

## 容量、所有权与释放

同一训练 attempt 的 worker 在节点内共享不可变 safetensors payload；每次新查找验证文件 SHA，张量 dtype 不变。调用方借用 lease，使用中的条目不可淘汰。GPU 副本属于当前 forward；运行时保留源 lease，查询到搬运完成后归还，无需逐次命中等待，并在训练步结束或退出时排空。每个 rank 的待完成复制数量不超过其有效 batch 样本数，源张量字节不超过 payload 预算；容量不足时回退在线。不建立 GPU 常驻特征缓存。

Payload 上限覆盖已发布、借用和待发布文件，LRU 只淘汰未借用条目；条目过大或容量不足时在线计算。Writer staging 有独立上限，数值等于 payload 预算，在 CPU 复制前检查。SenseNova 可选预取另有每 worker 字节/深度限制，关闭存储前先排空预取。预取提示只安排读取，实际 forward 仍重建依赖并检查冻结状态。

这些预算不等于进程 RSS 总上限；输入、GPU 临时副本和 metadata 也占内存。tmpfs 使用 host RAM，磁盘缓存也可能占用 OS page cache。并发训练 attempt 各有独立预算，节点容量规划需相加。

正常结束、取消和被捕获异常会关闭 lease/client，最后一个 client 删除 namespace。Worker 突然退出后由 OS 释放锁，下次缓存初始化在确认 client/payload 锁均空闲后回收遗留 namespace；没有后台清理 daemon。恢复 checkpoint 时重新构建缓存，缓存文件不属于训练 checkpoint。

## 模型适配

[`ConditioningSpec` 与 `ConditioningResolver`](../../../relax/models/generative_sft.py) 保持模型解耦。适配器声明 schema/版本、有效依赖和 `cacheable`，在正常模型 forward 中调用 `resolve(spec, compute, restore)`。`compute` 返回扁平张量字典，`restore` 验证字段/shape/dtype 并构建当前 forward 的模型对象；不能修改借用张量或在 lease 结束后保留它。可用 `conditioning_state()` 声明需检查的参数、buffer 和语义状态。未来 RL 接入还需定义其所有权和 collective 边界，本期没有 rollout 特征传递。

固定版本的数值与性能验收工具、实验记录见[独立验证仓库](https://github.com/Religious-J/relax-condition-cache-validation/tree/117d4fb037d93e2123734ccacf3e411e5f5b7b1e)。
