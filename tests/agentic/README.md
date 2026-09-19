# Agentic 多协议 canonical request 测试

## 任务目标

任务 8 为 OpenAI Chat Completions、OpenAI Responses 和 Anthropic Messages 建立 canonical request 一致性测试。语义相同的原始请求进入各自协议适配器后，应得到相同的 Agentic Session 上下文。

## 修改范围

1. `test_protocol_canonicalization.py` 通过三种协议的实际请求规范化入口运行测试。
2. `fixtures/chat_completions.json` 使用 Chat Completions 的 `messages` 和 `tools` 请求格式。
3. `fixtures/responses.json` 使用 Responses 的 `input`、`function_call` 和 `function_call_output` 请求格式。
4. `fixtures/anthropic_messages.json` 使用 Anthropic 的 `messages`、`tool_use`、`tool_result` 和 `image.source` 请求格式。
5. `fixtures/canonical.json` 保存手写的固定 canonical 预期结果。测试运行时不会使用待测规范化函数生成该文件。
6. `relax/agentic/session/service.py` 和 `relax/agentic/session/state.py` 补充了测试发现的请求校验与稳定序列化处理。

## 测试设计

1. 每个协议都有独立编写的原始请求 fixture，不会从另一协议的 fixture 转换而来。
2. 每份 fixture 包含普通文本、多轮工具调用和结果，以及混合文本、HTTP 图片 URL 和 base64 图片数据的请求。
3. 每个协议的结果都与 `canonical.json` 做深度相等比较，并且三份结果彼此比较。
4. 比较字段固定为 `messages`、`tools` 和 `chat_template_kwargs`。流式配置、响应 ID 和协议专属响应封装字段不属于比较范围。
5. 测试会重复规范化请求、反转对象键插入顺序，并固定比较状态哈希，检查结果不依赖字典键顺序。
6. 测试会在规范化后清空原始请求对象，确认结果没有引用输入对象中的可变数据。
7. 测试会改变消息、工具、tool call 和内容块的数组顺序，确认数组顺序仍然参与状态哈希。
8. 多个纯文本内容块会归并为一个字符串，使三种协议得到相同的 canonical 内容表示。

## 规范化一致性

1. 文本场景覆盖 system、user 和 assistant 消息，以及协议中的 developer 到 system 映射。
2. 工具场景覆盖 assistant tool call 的 ID、函数名、JSON arguments、工具描述和 JSON Schema parameters。
3. tool result 必须引用此前尚未完成的 tool call ID。缺失或不匹配的关联 ID 会返回带原始请求字段路径的请求错误。
4. 图片场景把三种协议的合法图片输入规范化为 `image_url` 内容块。HTTP URL 仅作为字符串处理，不会访问网络；data URL 会验证图片 MIME 前缀和 base64 编码。
5. `tools` 会保留 function 工具定义，并对 JSON Schema 的对象键排序和复制。
6. `chat_template_kwargs` 只接受 JSON 兼容值，并对嵌套对象键排序。非字符串键、集合和非有限浮点值会被拒绝。
7. 状态哈希使用排序后的紧凑 JSON 和 SHA-256，不再把不支持的对象转换为字符串。

## 异常场景

1. 非法、空字符串、非字符串或缺失的 role。
2. 空字符串、空列表、错误类型或不支持的内容块。
3. 缺失或空的 tool call ID，缺失 function、函数名或 arguments。
4. 缺失、重复或不匹配的 tool result 关联 ID，以及 Anthropic tool result 缺失 content。
5. 缺失工具参数 schema，或不稳定的工具定义 JSON 值。
6. 缺失图片源、错误图片块类型、错误图片 URL、错误 media type 和无效 base64 数据。
7. 空消息序列和规范化后没有受支持消息的请求。
8. 每项拒绝测试检查 `AgenticChatRequestError`、HTTP 400、`param` 和原始协议请求结构中的字段路径。

## 验证命令

1. `python -m pytest tests/agentic/test_protocol_canonicalization.py -q`

   运行三协议 golden、异常、稳定性和输入不可变性测试。结果为 `91 passed`。

2. `PYTHONHASHSEED=1 python -m pytest tests/agentic/test_protocol_canonicalization.py -q`

   使用第一个哈希种子运行同一测试。结果为 `91 passed`。

3. `PYTHONHASHSEED=777 python -m pytest tests/agentic/test_protocol_canonicalization.py -q`

   使用第二个哈希种子运行同一测试。结果为 `91 passed`。

4. `python -m ruff format --check relax/agentic/session/service.py relax/agentic/session/state.py tests/agentic/test_protocol_canonicalization.py`

   检查本次 Python 文件格式。结果为通过。

5. `python -m ruff check relax/agentic/session/service.py relax/agentic/session/state.py tests/agentic/test_protocol_canonicalization.py`

   检查本次 Python 文件静态规则。结果为通过。

6. `git diff --check`

   检查提交前差异中的空白错误。结果为通过。

7. `python .pre-commit-hooks/gitleaks_tracked.py`

   扫描已跟踪文件中的敏感信息。结果为未发现泄漏。

## 验证边界

1. 本测试验证请求进入 Agentic Session 前的 canonical 上下文，不验证协议响应封装。
2. 本测试不启动模型、GPU、Ray 集群或外部服务。它们不属于任务 8 的请求规范化验收范围。
