# 评测方案对比与工具准备

[English](evaluation-setup.en.md) · [评测运行方式](evaluation.md)

SimpleVQA 和 VDR 的数据文件不绑定某个工具服务。准备成本取决于采用直接问答、搜索 Agent，还是复现某篇论文的完整工具环境。当前 `benchmark ... base/relax` 默认运行无工具direct模式；`base/relax`区分初始与SFT权重。显式选择 `BENCHMARK_WORKFLOW=agent` 才启用工具。

## GitHub 中的实现有什么区别

| 实现                                                                                                                                                                               | 实际流程                                                                  | 额外准备                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [SimpleVQA 官方推理脚本](https://github.com/SimpleVQA/SimpleVQA/blob/b94d1f3d05d7f488afcdcf95bfcd69bcb279111e/data_engineering/vqa_request/simplevqa_request_lvlm_api_parallel.py) | 图片和问题直接请求模型，答案另交给裁判                                    | 待评模型与裁判接口；不需要搜索、OCR 或超分服务                                        |
| [lmms-eval SimpleVQA](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/simplevqa)                                                                           | 单次生成短答案，归一化后 exact match                                      | 模型与数据；该实现默认最多生成32 tokens，不是本文的 Agent 协议                        |
| [Vision-DeepResearch 官方评测](https://github.com/Osilly/Vision-DeepResearch/blob/1f13ea7261de0ec8e9961d3ef2c9c33fcd6d27b1/rllm/eval/README.md)                                    | 多轮 Agent，使用搜索、访问网页、Python 和裁剪后图搜；摘要与裁判可共享服务 | 搜索/网页读取权限、图片托管、摘要模型和裁判；该入口没有要求独立OCR或超分服务          |
| [WebWatcher](https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWatcher/README.md)                                                                                  | 先运行工具推理，再运行裁判脚本                                            | 文搜/图搜、Jina、图片存储、摘要模型、裁判；README中的一条eval命令以这些服务就绪为前提 |
| [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL/tree/1ac26e2c35a3e813e022813c88888db457d88f7d/opensearch_vl)                                                            | 本例复用的多轮工具环境，包含OCR、搜索、裁剪和图像增强                     | 当前缺项表对应这一套完整环境                                                          |

SimpleVQA 官方的[裁判](https://github.com/SimpleVQA/SimpleVQA/blob/b94d1f3d05d7f488afcdcf95bfcd69bcb279111e/llm_as_a_vqa_judger.py)和[汇总脚本](https://github.com/SimpleVQA/SimpleVQA/blob/b94d1f3d05d7f488afcdcf95bfcd69bcb279111e/simpleVQA_eval.py)区分正确、错误、未尝试，并计算回答条件下的准确率和F1。lmms-eval的exact match、本文的MaaS语义判分也不是同一个指标，不能只按数据集名字比较数字。

VDR仓库的[结果表](https://github.com/Osilly/Vision-DeepResearch/tree/1f13ea7261de0ec8e9961d3ef2c9c33fcd6d27b1#performance)同时列出Direct Answer和Agent Workflow，说明无工具对照本身是有效实验。其评测示例使用 `simplevqa_300.parquet` 和500题VDR；本例选定的是SimpleVQA全部1013道英文题和500题VDR。题目范围、工具、裁判、温度、上下文和失败口径都应单独记录。

## 如何选择准备范围

若先检查SFT是否损伤基础问答，可以增加一组初始/SFT的无工具直接问答对照：图片和问题 → 固定提示生成答案 → 同一裁判。这组实验无需等待搜索、图片上传、OCR、超分，结果单独报告。当前默认direct模式已实现这组评测，不需要工具服务。

若评估训练得到的工具使用能力，则需要可工作的工具环境。可以采用较少工具的搜索Agent，也可以继续本文已选定的OpenSearch-VL协议。删减工具、替换搜索供应商或更换摘要模型都需要形成独立的实验配置，并在初始与SFT两组共用。

当前选定协议为[评测文档](evaluation.md)中的无工具直接问答。以下准备方式仅用于可选的完整OpenSearch-VL工具环境。

## 现有工具逐项怎么准备

| 能力          | 准备方式                                                                                                                                                                   | 接入前检查                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 文本搜索      | 申请或复用[Serper](https://serper.dev/)账号的搜索权限，配置 `SERPER_API_KEY`；也可使用上游已有 `API_HOST / API_USER / API_KEY` 网关                                        | 实际搜索返回非空 `organic`，包含标题、链接和摘要                                                   |
| 网页读取      | 配置[Jina Reader](https://docs.jina.ai/)与 `JINA_API_KEY`；已有兼容网关也可复用                                                                                            | 已知公开网页能返回正文，而不是验证页、空文本或搜索摘要                                             |
| 反向图搜      | 配置[SerpApi Google Lens](https://serpapi.com/google-lens-api)，`SERPAPI_URL=https://serpapi.com/search.json`，提供 `SERPAPI_API_KEY`                                      | 当前适配器发送GET，参数为 `engine=google_lens / url / api_key`；检查JSON中的业务错误和实际匹配内容 |
| 网页/图搜摘要 | 提供固定的 `QWEN_API_BASE / QWEN_MODEL_NAME`                                                                                                                               | 原始OpenSearch适配器只发送Content-Type，不发送MaaS鉴权头；填入MaaS URL本身不能完成接入             |
| 图片托管      | 复用已有COS配置，提供 `IMAGE_UPLOAD_URL` 对应的本地上传桥                                                                                                                  | multipart字段为 `file`，返回 `{"url":"https://..."}`；图片需能由外部搜索服务获取                   |
| OCR与版面解析 | 复用或独立部署[PP-StructureV3](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html)，配置 `LAYOUT_PARSING_API_URL` | 核对请求和响应字段，真实文本图片应提取出可读文字                                                   |
| 超分          | OpenCV contrib的 `dnn_superres` 加EDSR对应倍率权重，配置 `SR_MODEL_PATH`                                                                                                   | 加载权重并检查实际输出尺寸；上游的succeeded字符串不能证明超分成功                                  |

### Jina密钥并非服务的硬性条件

Jina Reader允许匿名访问，官方[限流表](https://jina.ai/en-US/api-dashboard/rate-limit/)列出的匿名额度为20 RPM，免费key为500 RPM；实际额度以账号和当前服务为准。匿名访问适合小规模验证。当前完整入口把 `JINA_API_KEY` 列为必填，是本例的前置检查策略；若启用匿名方案，需要同时修改检查和全局限流，而不能只删掉这一项。网页读取失败时，上游可能退回搜索snippet，不能把这种降级当成正文提取正常。

### 摘要可以复用MaaS，仍需一个适配层

OpenSearch当前的两种摘要请求都是文本，因此可以尝试复用已有MaaS，而无需额外下载摘要模型。接入层需要补齐MaaS鉴权、核对上游非标准请求字段，并保留原摘要prompt。摘要模型必须固定，不能随待评的初始/SFT模型切换；更换模型也要记录为实验配置变化。

若摘要和裁判使用同一个限流为1 QPS的MaaS域名，应让两类请求经过同一个限流器。当前runner只约束裁判调用；它不会自动限制上游摘要HTTP请求。摘要失败时上游可能直接截取原文，因此接入检查必须确认实际发生了模型调用。

### 图片上传可以复用现有COS

上游的 `cos_upload.py` 需要外部 `upload.py` 插件，公开仓库没有提供当前环境可用的实现。本例已提供HTTP上传接口适配，尚需在Git外实现这个小服务：接收图片 → 使用已有coscmd配置上传 → 返回可访问URL。对象应使用内容hash和运行标识命名，放到 `/wuhuan2/lib/relax/` 下本实验的独立目录；保留到两组评测与缓存复用结束后再清理。

这里的外部可访问URL供搜索供应商取图；模型服务能接收base64图片，并不能替代图搜的取图链路。配置文件、内网地址、密钥及上传桥部署参数均放在Git外。

### OCR应使用匹配的服务协议

在独立环境中按[PaddleX Serving指南](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/serving.html)安装兼容版本与serving插件，再启动基本服务，例如：

```bash
paddlex --install serving
paddlex --serve --pipeline PP-StructureV3 --device cpu --host 127.0.0.1 --port 18080
export LAYOUT_PARSING_API_URL=http://127.0.0.1:18080/layout-parsing
```

这是服务启动示例，当前没有部署完成，也不表示CPU吞吐足够。先检查正确性和耗时，再决定是否使用单卡服务；不改动训练环境的CUDA或依赖。

当前适配器POST JSON，`file`为原图base64，`fileType=1`。如设置token，鉴权格式为 `Authorization: token ...`。它读取 `result.layoutParsingResults[0].prunedResult.parsing_res_list`，再选取 `paragraph_title / text / vision_footnote` 的 `block_content`。普通OCR字符串接口或其他VL解析模型不能直接替换这套响应。

### 超分权重需要真正执行一次

[OpenCV教程](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html)使用 `dnn_superres` 读取对应倍率的模型。[EDSR权重仓库](https://github.com/Saafke/EDSR_Tensorflow/tree/06c7bd65b0305c2955328f8f2721ea86c341f660/models)提供 `.pb` 文件。无需训练EDSR或安装TensorFlow；4倍超分使用 `EDSR_x4.pb`，其他倍率应有配套权重。

下载后保存来源revision和SHA256，在工具环境执行：

```python
import os
import cv2
import numpy as np

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(os.environ["SR_MODEL_PATH"])
sr.setModel("edsr", 4)
image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
result = sr.upsample(image)
assert result.shape == (64, 64, 3)
```

## 能否进一步减少外部服务

[OpenVLHarness](https://github.com/uclanlp/OpenVLHarness/tree/11149b6862f268def37e9e6bb37bd462ac8ed5f1)将文搜、Lens图搜、网页抓取统一接到Serper，并共用辅助摘要模型。这是减少供应商数量的参考方案，不是当前适配器已经支持的配置：本例Lens走SerpApi GET，Reader走Jina，不能只替换URL。

若采用统一供应商，应先实现并验证请求、响应和错误语义的适配，再记录为新的工具配置。省下的是账号和服务管理成本，搜索、图片托管、摘要与裁判这些逻辑步骤仍然存在。

实际准备顺序建议是：确定实验协议 → 开通搜索权限 → 接通COS与图搜 → 接通固定摘要与全局限流 → 验证OCR和真实超分 → 小样本完整Agent运行 → 估算调用量、费用和耗时 → 固定配置运行完整两组。仅填写环境变量不足以证明工具已经就绪。
