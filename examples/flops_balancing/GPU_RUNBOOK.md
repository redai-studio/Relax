# Task 16 上机验证操作手册

**最新四卡实测：Qwen3-4B 在三档长度混合负载上提升15.61%，在预先固定的一长十五短负载上提升26.84%；每组均为三对独立训练循环。** 两组完整梯度与实际FP32更新比较仅在双方共同的稳定数学配置下通过；默认原生BF16数值未通过。后者的纯长度可变条数控制与FLOPs调度分配相同，不能将26.84%归因于FLOPs平方项。Qwen3.5大模型候选尚未完成多卡验证。

## 最新NVLink四卡复测（11月11日）

使用固定版本`Qwen/Qwen3-4B`（`1cfa9a7208912126459214e8b04321603b3df60c`）、单机4×A800-SXM4-80GB、TP1/PP1/CP1/DP4、有效batch16和每卡23552 token预算。从固定版本公开OpenMathReasoning-mini样本池按预先声明的长度范围无放回选择，每场景400条、25个完整batch；未截断或复制样本。所有模式使用相同模型权重、样本顺序、随机种子、优化器、数值后端与容量约束。每次训练25步，前5步预热，后20步按原生`train_wait+train`计时；统计窗口内的等待和编译仍计入。每次独立启动，前三对按AC/AC/CA顺序运行。

| 场景       | 每batch样本长度构成              | A监督token/s（3次）            | C监督token/s（3次）            | 各侧吞吐中位数之比 |
| ---------- | -------------------------------- | ------------------------------ | ------------------------------ | ------------------ |
| 三档长度   | 2条10–16k、4条5–8k、10条1.5–2.5k | 7526.902 / 7527.807 / 7527.011 | 8701.845 / 8700.948 / 8705.632 | **+15.608%**       |
| 一长十五短 | 1条10–16k、15条1.5–2.5k          | 5146.013 / 5139.832 / 5175.878 | 6519.125 / 6527.072 / 6530.185 | **+26.837%**       |

A为现有长度均衡与动态binpack，C为FLOPs权重、允许不同DP副本拿不同样本数且保留原有容量保护。三档场景每对计时窗口均处理1401802个真实输入token；长短场景每对均处理847139个。逐步输入与监督token计数在每对A/C之间一致。两场景都通过独立的A1/A2基线重复和A1/C1完整数值检查，每份比较包含四副本的4022468096个参数元素。正确性结论限定共同稳定数学后端；实验使用的参数梯度与FlashAttention反向稳定化算子随证据包提供，**不是生产默认实现**。

另对三档场景运行一次E控制：保留C的可变样本数策略，仅令工作量系数为`(1,0)`，即按长度分配。E的监督token吞吐为8035.586 token/s，C三次中位数比E高8.291%。E只有一次、未做独立完整数值检查，所以只作为机制线索，不是正式三次配对的独立FLOPs收益声明。一长十五短场景的CPU精确重放表明E和C分配完全相同，其超过20%的主要可证收益来自可变条数重排。该场景是公开真实样本的预选混合负载，是否计作任务额外一星由评审判断。

提交证据时应一并提供冻结数据及生成规则、完整运行命令和环境版本、模型权重与数据摘要、三组A/C逐步原始日志、两份完整数值比较、长度控制和失败的原生BF16报告。新NVLink场景资料已打成`task16-nvlink-stratified-evidence-20261111.tar.gz`，SHA256为`b06cbcd45f0a9e03c46b1262f1b9a31eec741ce360021a6e1bdf93bd7acfde32`；包内`verify_nvlink_stratified_evidence.py`会从原始训练日志重算两组收益，并核对每次运行的源码、环境及逐步token覆盖。历史完整源码与默认后端失败记录在`task16-reproduction-and-evidence-20261111.tar.gz`，SHA256为`6d948416aecaa804fd89f02134a8b2e2835d586d7891d715bad3f0ce839fd7bf`。两包需在提交时以公开附件或可访问地址提供；仅本机路径不能作为评审证据。模型权重不包含在包内。

## 历史PCIe四卡实验（11月11日较早阶段）

### 当时实际运行范围

下文保留原始大模型候选流程；已上机验证的配置采用4×A800 80GB PCIe，同机包含两个NUMA节点，无NVLink。实际模型与候选不同，不能把小模型结论写成Qwen3.5大模型验证通过。

| 项目                   | 实际Dense实验                                     | 实际MoE实验                                |
| ---------------------- | ------------------------------------------------- | ------------------------------------------ |
| 模型                   | Qwen/Qwen3-4B                                     | allenai/OLMoE-1B-7B-0924                   |
| 固定版本               | `1cfa9a7208912126459214e8b04321603b3df60c`        | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` |
| TP/PP/普通DP/CP/EP/ETP | 1/1/4/1/1/1                                       | 1/2/2/1/1/1                                |
| 有效全局batch          | 16                                                | 32                                         |
| 数据                   | 同一公开数学数据的2048条自然样本，最长19161 token | 同一公开数据，2048条、最长4096 token       |
| 微批预算               | 最终对照共同23552 token                           | 8192 token                                 |
| 基线                   | 长度KK + 原生binpack                              | 长度KK + 原生balanced容量保护              |
| CPU offload            | 关闭                                              | 关闭                                       |

Dense的变长DP分配只在受控的同步、固定有效batch、TP1/PP1/CP1/EP1/ETP1路径开启；其他布局保留等样本数的FLOPs分配。容量约束仅在KK结果超预算时尝试已有打包器，装不下则保留原生多微批处理。不能把这一范围外推为TP2/CP2/EP2等组合已通过本轮验证。

**数值范围必须声明：默认原生BF16后端未通过冻结的实际更新容差。** 原因包括归约顺序改变后的低精度参数梯度舍入，以及FA2反向dQ分片数量随包内序列数变化。当前全面验收使用双方共同的、独立归档的数学配置：Dense保留原生前向和TN/NN矩阵路径，对参数梯度累加与FA2反向分片做稳定化；MoE使用K128补偿累加及固定算子配置。它们是复现实验条件，不属于本次调度核心的默认行为。不能拿该配置的正确性证明另一后端的吞吐，也不能声称默认后端已数值验收通过。

独立复现需使用证据包中的`install_reproduction_sources.py`恢复实际源码，再使用随包的`reproduce_task16.py`与数学配置及完整依赖模块；旧候选脚本不会自动重现实验配置。模型权重不随证据包分发，按固定版本下载后核对全部分片SHA256。证据包不包含凭证。

恢复权重时，先复制证据包中的小配置和tokenizer，再仅下载固定版本的权重分片，避免覆盖OLMoE的已冻结问答模板：

```bash
cp -a "$EVIDENCE/assets/Qwen3-4B" "$QWEN3_WEIGHTS"
hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --include '*.safetensors' --local-dir "$QWEN3_WEIGHTS"
```

MoE对应`allenai/OLMoE-1B-7B-0924`及上表版本，先复制`assets/OLMoE-1B-7B-0924`。为MoE使用新的源码目录并选择`--profile moe-final`，恢复容量约束加入前的实际源码；Dense使用默认`dense-final23k`。MoE参数为`--model moe --candidate D --max-tokens-per-gpu 8192`，数据目录为`assets/workload-moe`。两类模型各自独立完成正确性和性能阶段。

```bash
python install_reproduction_sources.py --root "$EVIDENCE" --repo "$REPRO_REPO"
python "$EVIDENCE/reproduce_task16.py" \
  --root "$EVIDENCE" --repo "$REPRO_REPO" --address "$RAY_ADDRESS" \
  --checkpoint "$QWEN3_WEIGHTS" --model dense --candidate C \
  --workload "$EVIDENCE/assets/workload-dense-accumulation-step15" \
  --phase correctness --max-tokens-per-gpu 23552 \
  --output "$RESULTS/dense-correctness"
python "$EVIDENCE/reproduce_task16.py" \
  --root "$EVIDENCE" --repo "$REPRO_REPO" --address "$RAY_ADDRESS" \
  --checkpoint "$QWEN3_WEIGHTS" --model dense --candidate C \
  --workload "$EVIDENCE/assets/workload-dense-24k" \
  --phase throughput --max-tokens-per-gpu 23552 --repeats 3 \
  --output "$RESULTS/dense-throughput"
```

上述命令要求先恢复证据包记录的依赖、环境变量、原生源码及其补丁。计时关闭capture，每次从相同模型权重和新优化器开始，25步中丢弃前5步、统计后20步，顺序AC/CA/AC；按总真实token除以总`train_wait+train`计算，另报监督token，不能挑最好单次。数值比较在退出的子进程中完成，避免比较器的CUDA缓存挤占下一次训练显存。

### 已完成的验收与性能结果

Dense完整同状态检查覆盖4,022,468,096个optimizer-owned元素。A/A初始值、梯度、实际FP32更新均逐位相同；A/C的梯度与实际更新最大绝对差均为2.9802322387695312e-8，初始参数完全相同。实际K2/K1及长度成本控制都通过原冻结容差：loss/梯度atol1e-5、rtol1e-3；实际更新atol1e-7、rtol1e-3。验收逐元素判断，不以最大绝对差一项代替完整容差检查。

| 独立对照           | 顺序 | A监督token/s | C监督token/s | 提升         |
| ------------------ | ---- | ------------ | ------------ | ------------ |
| 1                  | AC   | 6480.7733    | 7538.3291    | 16.3184%     |
| 2                  | CA   | 6452.2739    | 7451.5835    | 15.4877%     |
| 3                  | AC   | 6447.9458    | 7450.2280    | 15.5442%     |
| 三次吞吐各取中位数 | —    | 6452.2739    | 7451.5835    | **15.4877%** |

每次计时窗口都含1,385,448输入token及1,365,216监督token，输入和监督吞吐的相对提升一致。实际四rank迭代器日志核实每卡执行微批数24→20，四卡总空微批12→0；不是从CPU估算推导的执行次数。

同配置另做一次长度成本控制：允许变样本数、保留容量约束和原生binpack，仅把分配成本改为L，监督吞吐7340.8823 token/s，执行次数同为每卡20且空微批0。C三次中位数相对它约+1.5080%；这是一次控制运行，不能宣称重复验证了1.5%的独立收益。整体15.49%包含变样本数和容量利用收益，不能全部归因于二次FLOPs项。

MoE的A/A/B/C/D四个完整比较都通过，覆盖6,919,161,856元素；独立25步A/D一对监督吞吐17764.3757→18044.8398 token/s，+1.5788%。它没有达到10%，也不是三次重复中位数。

冲刺20%的额外结果全部保留：共同GBS12/预算19328的一对仅+10.9987%；共同GBS16/预算20736的一对仅+2.2840%，均低于预先固定的18%继续门槛，未做新完整数值验收。原23552配置的四rank真实时间线显示仍有计算不均与同步等待；根据一次时间线拟合，另冻结注意力执行成本权重2作C单次筛查，约7527.4886监督token/s，相对旧A三次中位数+16.6641%。它没有配对重复和新精度证明，不能替代最终15.4877%的成绩；拟合及权重不是理论FLOPs修正，也未加入默认配置或正式代码。

### 证据包与环境恢复

完整源码、冻结数据、完整数值报告、原始日志、真实微批记录、失败结果、校验清单及有限复现程序归档为`task16-reproduction-and-evidence-20261111.tar.gz`；不含模型权重或全模型原始参数导出。可选的`task16-compiled-operators-cu130-py312.tar.gz`保存本机已测试的FA2/Apex wheel及TE算子与Python层，减少相同平台重编译时间；它是实验复现材料，不是PR核心代码。

实际环境为Linux x86_64、Python3.12.3、Torch2.12.1+cu130、Ray2.56.1、FA2.7.4.post1、TE2.14.1、MCore0.19.0、Bridge0.7.0及带固定补丁的TransferQueue0.1.10.dev0。以包内`RUNTIME_VERSIONS.json`、`FINAL_DISTRIBUTION_PINS.txt`和实际源码为准；旧安装输入记录中包含后来被替换的依赖，不能直接用旧`runtime-requirements.txt`覆盖最终版本。仅相同包版本号不足以替代源码和算子摘要。

先准备匹配的Python/Torch/CUDA运行环境并恢复包内记录的依赖。FA/Apex可安装随可选包保存的实际wheel；TE需同时恢复其中`installed-python/transformer_engine`的Python层和`wheel_lib`算子，不能只复制一个动态库。库文件只适用于记录的ABI与架构；源码包同时保留实际TE及其依赖源码、构建日志、FA版本和算子SHA256，便于按匹配环境重建。不要替换驱动或把其他CUDA大版本的wheel混入本轮结果。

`install_reproduction_sources.py`已经在两个全新目录恢复并核对Dense最终源码及MoE容量改动前源码的摘要，均与实际运行记录一致。`verify_final_evidence.py`和`verify_complete_campaigns.py`用于复核本次租赁机路径下的完整证据；新机器复现训练使用上面的通用CLI，而非直接执行带历史绝对路径的实验控制器。历史实际控制器及原复现程序也保留，便于审计。

计时采用训练循环的`train_wait+train`，不是包含下载、环境安装与模型初始化的整个应用墙钟时间。前5步预热后，计时窗口内出现的编译或等待照常计入，未人工扣除。Dense定向profiler只采实际第11步K2/K1，诊断耗时未计入吞吐验收；分析从Chrome trace筛选真实kernel，排除GPU用户标注与父子区域的重复计时。NCCL活动可能包含等待其他卡，不能直接当成纯网络传输时间。

## 1. 四组实验与候选布局

| 模式 | 微批构造                        | DP 分配     | 用途               |
| ---- | ------------------------------- | ----------- | ------------------ |
| A    | 原有动态估数、KK 分组及容量回退 | 原有长度 KK | 保留已有优化的基线 |
| B    | 带容量约束的 FLOPs 打包         | 原有长度 KK | 单看微批收益       |
| C    | 原有微批路径                    | FLOPs KK    | 单看 DP 收益       |
| D    | FLOPs 打包                      | FLOPs KK    | 组合效果           |

两开关独立，默认关闭。候选脚本用普通同步 SFT、THD、BF16、TP2/PP2/CP1；八卡对应普通 DP2。MoE 另设 EP2/ETP1，不能再把 EP 乘进普通 DP 卡数。MTP、dropout、辅助/z loss 关闭，不改 loss mask 或有效 batch。

最新主分支还提供 `sft_packing_strategy=binpack` 和独立的 `sft_wholepack` 策略。
此候选实验固定 `balanced`，保留其最新容量保护。DP-only 可叠加 binpack；FLOPs 微批构造与 binpack、任一 FLOPs 开关与 wholepack 均不允许同开。
若实际基线已使用 binpack，需另保留该基线的性能对照，不能仅与更弱配置比较后宣称收益。

八卡上 FP32 Adam 状态很大，所以四组都开启 `--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer`，读取优化器实际 FP32 CPU 主权重。CPU RAM、内存带宽与 PCIe 可能成为瓶颈，offload 也可能压低优化收益。改精度、offload、batch、重计算、布局或预算时，所有对照必须一起改并重新冻结。

### 四卡小模型候选方案

八卡不可用时，使用 `scripts/training/sft/run-four-gpu-flops-task16.sh`，仍分别测 A/B/C/D：

| TASK16_MODEL | 模型                     | TP  | PP  | 普通 DP | EP  | ETP |
| ------------ | ------------------------ | --- | --- | ------- | --- | --- |
| dense        | Qwen/Qwen3.5-4B          | 2   | 1   | 2       | 1   | 1   |
| moe          | allenai/OLMoE-1B-7B-0924 | 1   | 2   | 2       | 2   | 1   |

两种模型顺序使用同一台四卡机器，CP1。EP 与普通 DP 重叠，不能把 EP 再乘进卡数。
这是待验证的替代模型方案：Dense 覆盖 TP，MoE 覆盖 PP/EP，两者均有 DP2。
不能据此声称覆盖了全部模型与并行组合，也不能预先保证达到吞吐门槛。

```bash
python -m examples.flops_balancing.download_assets --profile four --output "$ASSETS"
```

此 profile 固定 Qwen3.5-4B 版本 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、
OLMoE 版本 `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，仍不下载权重。
OLMoE 是基础模型；下载工具明确安装本地问答模板 `chat_template.jinja`。
模板给问题加两个换行，只对答案和 EOS 计 loss；冻结和训练使用同一目录，manifest 记录模板摘要。

按下节命令为每个模型**分别**冻结数据，加 `--max-length 4096`，同时替换模型目录、版本及输出目录。
四卡启动器拒绝旧的 8k manifest。训练固定位置上限 4096，8192 的微批 token 预算可以容纳多条文档。
先用 `TASK16_DRY_RUN=1` 及三个路径变量检查命令；正式启动仍通过 `scripts/entrypoint/ray-job.sh`，必须设置实际 Ray 地址。

四卡默认不 offload 优化器，显式关闭 precision-aware optimizer，保留 FP32 梯度、主参数和 Adam 状态。
OLMoE 保留 Bridge 原生模型构造，TP1 不开 sequence parallel；两组 MoE 对照都关闭辅助/z loss。
OLMoE 静态参数、梯度和优化器状态估算约 **29.66 GiB/卡**，不含激活、通信、临时张量和加载峰值。
建议候选硬件为 4×A800 80GB、约 256 GiB 主机内存；最终峰值必须以上机测量为准。
首次先跑一步正确性，再做短窗口筛选，最后对有效候选跑至少三组独立吞吐窗口。

## 2. 固定资产和真实数据

| 资产                                                                                     | 固定版本                                   |
| ---------------------------------------------------------------------------------------- | ------------------------------------------ |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)                                   | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)                           | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| [OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) | `5ea611a9fde449e7df8e72aefe698996a9882390` |
| 项目 Docker 固定 Bridge                                                                  | `6862ee170a2569607e8ebc5dde91f119779093ca` |
| Bridge 切换 dev 后的 Core                                                                | `4a4de865795f123eda422ebf0a4c13d2418cf3cb` |
| 项目固定 TransferQueue                                                                   | `9784ad0d8a5f0db93ff8326379ab2a6d06e640a0` |

历史四卡实测镜像的 TransferQueue 应用了项目 Docker 中的 `docker/patch/transferqueue/9784ad0-assignment-transform.patch`。
当前 Task 16 的可变条数分配已在 Relax 采样器中实现，不再依赖此补丁；复现实测时仍需记录镜像 digest 与实际安装源码摘要。
此前旧 Core 上的单卡优化器检查不代替这一版本组合的原生验证。

`download_assets` 下载固定数据、配置和 tokenizer，不下载模型权重。基础 Parquet 有 19252 行。规则：种子 1605 打乱物理行号，依序保留前 2048 条满足长度区间的原始行；不截断、不重复、不按长度排序。使用生产 `SFTStreamingDataset` 生成长度和 mask。

每个模型分别保存版本、manifest 和长度分布。长度分布与适用条件应和最终性能结果一起报告。

准备过程沿用已有 Qwen 模板 patch 和 per-message mask fallback，保留相关 warning。manifest 记录源行号、逐条长度、mask sum、文件/配置摘要及渲染摘要。GPU 数值运行另记录训练端实际收到的 token/mask 摘要。

复现准备，不需要 GPU：

```bash
cd /path/to/Relax
export ASSETS=/path/to/task16-assets
python -m examples.flops_balancing.download_assets --output "$ASSETS"
python -m examples.flops_balancing.prepare_workload \
  --source "$ASSETS/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet" \
  --model "$ASSETS/Qwen3.5-27B" --output "$ASSETS/workload-dense" \
  --source-revision 5ea611a9fde449e7df8e72aefe698996a9882390 \
  --model-revision fc05daec18b0a78c049392ed2e771dde82bdf654 --samples 2048
```

MoE 替换模型/输出目录与模型版本。长序列另用新目录，增加 `--max-length 32768`。已有输出目录会拒绝覆盖。

## 3. 资源边界与上机检查

候选是单机 8×A800 80GB，需核对实际互联，建议主机 RAM 约 1TB；先验证基线可加载并走完一步。单卡小模型通过不代表完整模型放得下。

全量数值导出包含 FP32 初始值、梯度、更新量，约 `12 × 参数量` 字节。27B 单次约 324GB，35B 单次约 420GB，尚不含副本与文件开销。需预留基线和当前对照的磁盘空间。比较器逐参数读，不把全模型向量一次加载进 RAM。

完整权重下载到相同固定版本目录：

```bash
hf download Qwen/Qwen3.5-27B \
  --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir "$ASSETS/Qwen3.5-27B"
hf download Qwen/Qwen3.5-35B-A3B \
  --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --local-dir "$ASSETS/Qwen3.5-35B-A3B"
python -m examples.flops_balancing.preflight --require-gpus 8 \
  --require-host-memory-gib 900 --require-disk-free-gib 1200 \
  --output "$ASSETS/task16-preflight.json"
```

优先用项目固定 Docker/依赖及补丁。CPU venv 或单独加入 PYTHONPATH 的 Core 源码不是完整训练镜像。A800 候选用标准 MoE alltoall，没有启用参考 Hopper 路径的 DeepEP。preflight 只读设备、拓扑与版本，不装驱动。

上机检查同时读取容器内存限制，不能只看物理服务器的 RAM；输出文件放在实际资产/数值导出磁盘上，确保空间检查针对正确的挂载点。运行记录保存白名单环境变量和实际 TransferQueue/Megatron 源码摘要，性能汇总会拒绝这些条件变化的对照。

## 4. 离线预览与原生数值对照

```bash
TASK16_DRY_RUN=1 TASK16_MODEL=dense MODE=D \
  HF_CHECKPOINT="$ASSETS/Qwen3.5-27B" WORKLOAD_DIR="$ASSETS/workload-dense" \
  RUN_DIR=/tmp/task16-dense-D-preview \
  bash scripts/training/sft/run-qwen35-flops-task16.sh
```

预览校验数据/tokenizer 摘要，记录完整参数、代码与依赖，不启动 Ray。正式运行还检查所有权重 shard 存在。每次使用新的 RUN_DIR，从同一 HF 权重和新优化器开始，不续训、不保存 checkpoint、不评估。

在首次 GPU 对照前固定候选容差并留档：

| 量                         | atol   | rtol   |
| -------------------------- | ------ | ------ |
| loss                       | `1e-5` | `1e-3` |
| 归约、反缩放后且裁剪前梯度 | `1e-5` | `1e-3` |
| 实际 FP32 Δθ               | `1e-7` | `1e-3` |

逐分量判定 `abs(y-x) <= atol + rtol*abs(x)`，另报相对 L2。非有限值、漏参数、初始值/输入/mask/归一化变化均失败。不能看过优化结果后放宽阈值。

先跑 A1、A2 检查基线自身波动，再 B/C/D；先用 `NUM_ROLLOUT=1` 完成一次有效 batch 更新。以下地址和目录需填目标机器的实际值：

若同配置 A/A 已出现梯度或更新差异，先调查算子和归约的非确定性，不能放宽阈值后继续。可在两组新运行前统一设置 `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NCCL_ALGO=Ring`；入口会将显式设置传入 Ray worker，并记录到 run.json。它们不保证整个模型确定性，尤其 packed Qwen3.5 的 FLA 路径不支持原生全局 `--deterministic-mode`。环境变更后必须重新建立基线，后续数值及性能对照保持同一设置。

隔离整层重计算的影响时，可对一整组新 A/A/B/C/D 统一设置 `TASK16_RECOMPUTE=none`；默认值仍为 `full`。该选项只省略原有三项整层重计算参数，不改变 loss、精度或比较阈值。必须重新检查实际配置和显存峰值，不能将两种设置的运行混作性能对照。

```bash
export RAY_ADDRESS=http://actual-head:8265
export HF_CHECKPOINT="$ASSETS/Qwen3.5-27B"
export WORKLOAD_DIR="$ASSETS/workload-dense"
export TASK16_MODEL=dense
export RUN_DIR=/path/to/results/dense-A1
MODE=A TASK16_CAPTURE=1 NUM_ROLLOUT=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen35-flops-task16.sh
python -m examples.flops_balancing.collect_run \
  --address "$RAY_ADDRESS" --job-id actual-job-id --run-dir "$RUN_DIR"
```

训练结束后执行 collect_run。`submit.log` 只是提交状态，`training.log` 才是原始训练日志。capture 环境变量被显式传入 Ray actor，仅导出 rollout=0、step=0。它读取原本一次 prepare_grads 之后的梯度，在原本一次 step 后算 Δθ，不重复 prepare/unscale/step。offload 用 HybridDeviceOptimizer 实际 FP32 主权重映射，不支持的精度感知/FSDP 实现会报错。

```bash
python -m examples.flops_balancing.compare_step_captures \
  /path/to/results/dense-A1/capture /path/to/results/dense-D/capture \
  --loss-atol 1e-5 --loss-rtol 1e-3 \
  --grad-atol 1e-5 --grad-rtol 1e-3 \
  --update-atol 1e-7 --update-rtol 1e-3 \
  --output /path/to/results/dense-A1-vs-D.json
```

A2/B/C/D 分别与 A1 比较。MoE 改 `TASK16_MODEL=moe` 和相应 HF/数据目录重做。原生验证覆盖实际启用的 TP2/PP2/DP2，以及 MoE EP2/ETP1；单卡或 CPU 元数据替身测试不能算这些组合已验证。预打包另受原有 PP=CP=VPP=1 等限制，正式使用时还需单独实测。

## 5. 训练循环吞吐

关闭导出：`TASK16_CAPTURE=0`，确保 runtime env 没有 `RELAX_FLOPS_CAPTURE_DIR`。带同步/磁盘导出的运行不能参与性能比较。

保持 `NUM_ROLLOUT=30`、global batch=64；预先约定丢弃前 5 个 rollout，测接下来的 20 个。每个模型 A 与候选各至少 3 次独立运行；先短跑 B/C/D 筛查，再补有希望方案的重复，A 与候选交错跑，观察 GPU/主机负载和显存。Dense/MoE 都要保留对照，即使某个无收益。

```bash
python -m examples.flops_balancing.summarize_runs \
  /path/to/results/dense-A1-perf /path/to/results/dense-A2-perf /path/to/results/dense-A3-perf \
  /path/to/results/dense-D1-perf /path/to/results/dense-D2-perf /path/to/results/dense-D3-perf \
  --warmup 5 --steps 20 --output /path/to/results/dense-throughput.json
```

主指标 `sum(真实非 padding 输入 token) / sum(train_wait + train)` 包含供数等待、调度、通信和优化器更新，另报监督 token 吞吐。dummy 不计入真实 rollout token。口径取现有生产计时器，固定排除启动和前5步预热，不运行checkpoint；测量窗口内若仍发生编译或等待，不手工扣除，需和原始日志一起说明。先算各次总 token/总时间，再比较中位数及 min/max/std，不能简单平均逐步 token/s。

汇总器检查固定参数、数据、代码、原生源码、依赖和设备记录；拒绝诊断/预览/失败任务、重复目录、缺步和 token 数不一致。10%/20% 标记只来自真实日志。达到阈值仍需结合基线波动、微批数、等待和显存解释收益来源。
