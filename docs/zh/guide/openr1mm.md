# OpenR1MM 奖励与评测

## 概述

`openr1mm_accuracy_format` 将答案准确率奖励和格式奖励独立相加，两项各为 `[0, 1]`，总奖励为 `[0, 2]`。实现位于 `relax/engine/rewards/openr1mm_accuracy_format.py`，保留实验所用的 XML／预填 think 评分行为；原有 `openr1mm` reward 及其原生通道支持保持不变。

准确率比较回答与参考答案的最终答案，不使用中间推理。显式选择题标签先校验选项，再用 `math_verify` 比较数学等价性。格式要求一个非空 `<think>...</think>` 后跟一个非空 `<answer>...</answer>`，支持 prompt 预填的 `<think>` 和可选的末尾 `<|im_end|>`。准确率和格式独立计分：格式正确但答案错误可得 1 分，答案正确但格式不符合要求也可得 1 分。这不是通用语义判分器，数学提取和文本归一化都有局限。

评测对 MathVista-testmini 和 MMMU validation 单独计算本地 0/1 accuracy，不调用外部模型、不加格式分、不经过训练的 dynamic sampling filter。[上游 Open-R1-Multimodal 评测](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal/blob/main/local_scripts/lmms_eval_qwen2vl.sh) 使用不同的 lmms-eval/GPT-4o 流程，并包含 MMMU-Pro，不能与这里的本地分数直接横比。

## 准备评测数据

在仓库根目录、已有 Relax 环境中执行，使用 `huggingface_hub`、`pyarrow` 和 Pillow。图片和转换后的数据应放在训练 worker 可访问的共享存储，`DATA_DIR` 指向数据集根目录。

如尚未转换 OpenR1MM 训练 parquet，先执行以下命令；重叠检查要求训练数据的 `image` 列为编码后图片字节的列表：

```bash
python scripts/tools/process_openr1.py \
  --input-dir "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet" \
  --output-dir "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
```

仅下载所需评测 split，再针对实际使用的训练 parquet 转换并检查重叠：

```bash
python examples/openr1mm/download_eval.py --output "$DATA_DIR/openr1mm-eval"
python examples/openr1mm/prepare_eval.py \
  --source "$DATA_DIR/openr1mm-eval/source" \
  --train "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet" \
  --output "$DATA_DIR/openr1mm-eval"
```

下载脚本默认固定 MathVista revision 为 `2b6ad69445fbb5695c9b165475e8decdbeb97747`，MMMU 为 `876ce5cb130f7f7e290ce4d9984357737d4db5cf`。需要时可用 `--mathvista-revision`／`--mmmu-revision` 覆盖；`sources.json` 记录解析后的 revision。下载脚本拒绝复用 revision 未知或不一致的非空源目录；切换 revision 时请使用新的 `--output` 目录，同版本中断下载可以续传。转换产物为：

- `mathvista_testmini.parquet`、`mmmu_validation.parquet`：完整转换的 split。
- `mathvista_testmini_disjoint.parquet`、`mmmu_validation_disjoint.parquet`：排除与训练图片完全重叠的样本。
- `overlap-report.json`：完整／去重后数量和被排除的源样本 ID。

每行包含 `prompt`、`image`、JSON 编码的 `label` 和源 `id`。MMMU 题干和选项中的图片引用按出现顺序展开。重叠依据是解码后的 RGB 像素和**图片尺寸**完全相同，图片编码格式可以不同；任一引用图片重叠就排除整道题。这不排除缩放图、近似图或语义重叠问题。

一次针对 verified 8k 训练集的准备结果为 MathVista 保留 997/1000 题、MMMU 保留 900/900 题，具体数量取决于训练文件。processor/tokenizer 的 prompt 长度过滤还可能进一步减少题数（该次实验实际评测了 992 道 MathVista）。请检查实际评测数量，不要只看 parquet 行数。

## 训练与评测配置

在模型对应的训练配置中使用 `--rm-type openr1mm_accuracy_format`。添加评测时，将以下参数追加到训练命令，保留现有并行、优化器和数据集配置：

```bash
--rm-type openr1mm_accuracy_format \
--custom-rm-path examples.openr1mm.eval_reward.reward \
--eval-config examples/openr1mm/eval.yaml \
--eval-interval 20 \
--eval-temperature 0 \
--n-samples-per-eval-prompt 1 \
--eval-max-prompt-len 8192 \
--eval-max-response-len 8192 \
--eval-max-context-len 16384 \
--sglang-context-length 16384
```

不设置 `--skip-eval-before-train`，即可在第一次更新前评测。YAML 选择两个去重后的 parquet，每题贪心生成一个回答。推理引擎上下文上限必须容纳评测输入输出；提高该上限不会提高训练 rollout 的长度限制。

`eval.yaml` 通过 `${oc.env:DATA_DIR}` 解析路径。请 export `DATA_DIR`，并确保它进入 driver 和 worker 的 Ray runtime `env_vars`。若 job 的工作目录不是仓库根目录，请为 `--eval-config` 使用绝对路径。仓库代码和转换后的文件都必须对 worker 可见。训练时保留模型对应的 chat template 和多模态映射（`--input-key prompt --label-key label --multimodal-keys '{"image":"image"}' --apply-chat-template`）。

custom reward 将 YAML 中带 `eval_benchmark` 标记的评测样本交给本地 accuracy；普通训练样本仍调用 `get_openr1mm_accuracy_format_reward`。不要向训练样本添加这个保留 metadata。训练仍可通过 `--dynamic-sampling-filter-path relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std` 开启 DAPO 过滤，它与评测相互独立。

## 判分与故障排除

评测评分器先隔离最终答案，接受选项字母或完整选项文本，也接受 `B. 选项内容`，但**字母和选项内容必须同时匹配**。字母与内容冲突、多选项罗列都不应得分。这修复了模型从 `B` 改为输出 `B. $7` 时出现的虚假骤降。开放题使用归一化文本精确匹配、候选别名、按 MathVista 精度比较数值和列表比较；通常不做单位、自然语言或符号等价归一化。

分数下降时，应先比较相同的 prompt/label、实际题数、截断和最终答案形式，再判断模型是否退化。历史回答也应使用相同版本的评分器重算，只更新最新评测点会使曲线不可比。训练 reward 包含格式分，因此可能超过 1；这里的 eval 指标不会超过 1。

CPU 回归测试命令：

```bash
python -m pytest tests/engine/rewards/test_openr1mm_accuracy_format.py \
  tests/examples/test_openr1mm_eval_reward.py tests/examples/test_openr1mm_prepare_eval.py
```

## 下一步

- [自定义训练](./customize-training.md)
- [数据集设计](./dataset-design.md)
