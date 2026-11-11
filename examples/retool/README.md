# ReTool

在单机八卡 Linux 环境使用 [Relax 运行镜像](../../README.md#quick-start)。Python 执行服务需要 root，并依赖 Landlock 或支持相应命名空间的 Apptainer。

```bash
RETOOL_OUTPUT_ROOT=/path/to/new-run bash scripts/training/text/run-retool.sh
```

入口下载官方 Qwen2.5-7B-Instruct、ReTool-SFT 和 DAPO-Math-17k，准备训练与评测数据，依次运行 SFT、GRPO 和两组 K8 评测。指定已有可写父目录下的新输出路径。下载需要代理时设置 `RETOOL_DOWNLOAD_PROXY`。运行时临时目录默认位于 `/tmp`，可通过 `RETOOL_RUNTIME_PARENT` 指定较短的本地路径。容器之外运行时，可用 `MEGATRON_PATH` 指向兼容的 Megatron-LM checkout。

Python 服务通过 SandboxFusion 兼容的 `/run_code` 接口提供 30 秒、1 GiB 的执行预算。用 `RETOOL_SANDBOX_BACKEND=landlock` 或 `apptainer` 选择后端，默认 `auto` 优先 Landlock。

## 固定训练协议

Qwen2.5-7B-Instruct，官方模型revision为a09a35458c702b33eeacc393d103063234e8bc28。作者SFT数据2000条、两epoch、batch80、50更新；原生Megatron，不使用veRL。GRPO采用完整17903题池、有放回抽样、200更新，每步64题×16独立轨迹，TP2、BF16、lr1e-6常数，Adam(.9,.999)、eps1e-8、WD.01、clip1、policy clip两侧.2、KL/entropy0。

保持原ReTool prompt、parser和评分：答案±1减去每次实际工具失败.1、无下限。prompt2048、response8192（含真实工具反馈）、context10240、最多16个assistant turn；temperature1、top-p1、top-k-1。生成到代码边界后，仅执行本次新代码，真实反馈mask0，不执行历史程序或环境重放。SGLang与Megatron使用相同合法词表范围。

指标明确使用TensorBoard，ClearML和W&B关闭。事件文件位于各阶段输出目录的tensorboard_log；训练rollout准确率不是独立测试集成绩。

## 自动训练后评测

训练成功并保存最终完整CP及HF后，入口自动调用本分支的evaluation模块和evaluate.sh。使用原生SGLang的evaluation-only模式，创建rollout而不创建optimizer；官方原模型和最终模型逐个加载，不能以SFT50权重代替官方原模型。

四个数据集全部题目：

| 数据集         | 题数 | 每模型轨迹 |
| -------------- | ---: | ---------: |
| GSM8K test     | 1319 |      10552 |
| MATH500 test   |  500 |       4000 |
| AIME2025 train |   30 |        240 |
| AIME2026 test  |   30 |        240 |

仅允许自主工具的一个模式，两模型合计30064轨迹。每题8个独立非贪心样本，保留同题配对seed，原prompt/预算/parser/acc规则不变。下载记录实际dataset revision；正式运行不截题，只有显式开发smoke从每套取一题验证接线。

- pass@8：完整8条至少一次正确的题数÷题数。
- mean pass@1：正确轨迹÷全部轨迹。
- 工具调用率：至少提交一次工具调用的轨迹÷全部轨迹。
- 另存工具提交数、实际反馈状态、原始tokens/logprobs/mask、终局和原生score；工具执行成功不等于数学正确。
- 官方初始模型对最终模型的差异包含SFT与RL总体变化，不能单独归因于RL。

完整原生样本在evaluation/{official,trained}/{benchmark}/case\*.json，原生dump在各arm的raw/eval_0.pt。各arm的evaluation_complete.json验证完整唯一K8分组后才接受；comparison.json汇总两模型四面板。缺失、失败或未正常退出不得宣称完成。
