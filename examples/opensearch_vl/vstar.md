# V\*Bench：本地裁剪工具评测

[汇总](README.md) · [评测方案](evaluation-setup.md)

本流程比较基座与SFT权重在同一套本地图像工具下的效果。模型自主输出`crop`调用，CPU从原始分辨率图像裁剪，将裁剪图返回模型，直至输出最终选项；不需要搜索或图片托管服务。它测的是视觉细节与工具交互，不与完整搜索Agent的SimpleVQA分数混排。

## 固定协议

- 数据为官方`craigwu/vstar_bench`的191题，revision为`d9ae62c903da0c98336e85c5ee89cd863b04b4da`，只下载题目及对应图片，约281 MB。
- 使用官方`test_questions.jsonl`的选项和标签；原始标注JSON中的正确答案位置不用于构造模型输入。参考标签只参与评分。
- 两组共用提示、温度0、32K输入输出总上下文、最多10轮及每张显示图片最多1,024视觉token。保留完整历史；超限、无答案和格式错误均留在分母。
- 工具沿用`crop(image, x, y, width, height)`。坐标为当前显示图片的像素；程序映射到该图对应的原始分辨率区域裁剪，再缩放用于显示。返回的新图使用`img_2`、`img_3`等引用。
- 最终答案要求`<response>A</response>`，按选项字母匹配；也接受单行明确选项如`(A) white`或`A. white`，不从分析段中猜选项，不调用模型裁判。每题只生成一条轨迹，另报训练集去重分数。

## 准备与运行

沿用现有Python环境中的Pillow、requests、OpenAI SDK和模型对应的Transformers。先下载并校验数据缓存：

```bash
python3 examples/opensearch_vl/tools/vstar_benchmark.py \
  --data "${DATA_DIR}/benchmark/vstar" --prepare-only
```

用显式的基座或SFT最终HF目录启动模型服务，保留原始工具调用文本，不启用消费该文本的工具解析器。Qwen3-VL-8B-Instruct可使用原生chat template：

```bash
python3 examples/opensearch_vl/tools/serve_sglang.py \
  --model-path "${HF_CHECKPOINT}" --served-model-name benchmark-model \
  --host 127.0.0.1 --port 30000 --tp-size 1 --context-length 32768
```

`AGENT_PROVENANCE_FILE`沿用模型来源验证记录：基座`kind=base`，训练后`kind=trained-final`，均须先完成可信文件hash校验，设置`hashes_verified=true`，且`served_model_name`与实际服务一致。训练集重叠索引沿用SimpleVQA流程生成的问题与原图像素hash集合。

```bash
export AGENT_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_MODEL=benchmark-model
export AGENT_PROVENANCE_FILE=/path/to/verified-model-provenance.json
export BENCHMARK_OVERLAP_INDEX=/path/to/train-overlap-index.json
export BENCHMARK_DATA_ROOT="${DATA_DIR}/benchmark"
export BENCHMARK_WORKERS=16
bash examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh benchmark vstar base
```

换成SFT最终权重，更新来源记录并重启服务，再以`benchmark vstar relax`运行另一组。`BENCHMARK_MAX_CASES=2`只用于冒烟验证，不能填写正式分数。两组协议应逐项一致，模型来源分别记录训练与评测commit。

## 结果

`${SAVE_DIR}/${EXP_NAME}/benchmark/<base|relax>/vstar/summary.json`记录准确率、失败原因、工具调用率、调用成功数及去重分数。逐题`trajectories.jsonl`保留模型原文、调用参数、真实执行状态和生成用量；`cases/<id>/`保留模型实际看到的图片。所有这些结构化产物均留在仓库外。

仅在191题完整、模型来源正确且服务异常已排除后，把正式分数补入[8B结果文档](results/qwen3-vl-8b.md)。有工具访问并不表示每题都会调用工具，必须同时查看调用率；本数据集也不保证搜索轨迹SFT必然提高闭卷视觉能力。
