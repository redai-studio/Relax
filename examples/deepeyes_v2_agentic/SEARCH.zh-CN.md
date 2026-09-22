# DeepEyes V2 文本搜索

[English](./SEARCH.md) · [示例环境准备](./README.md)

## 离线快速开始

以下 Bash 命令从 Relax 仓库根目录运行，Python 环境需要已经安装项目依赖。搜索模块复用 [requirements.txt](../../requirements.txt) 中的 HTTPX、Pydantic 和 PyYAML。

未设置 `DEEPEYES_V2_SEARCH_CONFIG_PATH` 时，文本 `search()` 使用确定性的 `mock` 结果。以下调用无需网络、搜索凭据、模型服务或 sandbox：

```bash
unset DEEPEYES_V2_SEARCH_CONFIG_PATH
python -c 'import json, sys; from examples.deepeyes_v2_agentic.app.search_utils import search; sys.stdout.write(json.dumps(search("DeepEyes V2", size=2), ensure_ascii=False) + "\n")'
```

相同查询和结果数量生成相同的 mock 结果。每条结果包含带有 `[mock]` 标记的标题、包含查询内容的摘要、对应查询的示例 URL，以及 `date: null`；`elapsed_time` 为 `0.0`。

## 配置与结果

`DEEPEYES_V2_SEARCH_CONFIG_PATH` 指定 YAML 文件。项目提供 [mock](./search_config.mock.yaml)、[Search-R1 retriever](./search_config.retriever.yaml) 和 [Brave Search](./search_config.brave.yaml) 模板。显式选择离线配置的命令如下：

```bash
export DEEPEYES_V2_SEARCH_CONFIG_PATH="$PWD/examples/deepeyes_v2_agentic/search_config.mock.yaml"
python examples/deepeyes_v2_agentic/app/search_runtime.py prepare
```

`prepare` 验证文件与所选认证变量，并输出配置的绝对路径。启动入口在启动 agent 或提交 Ray 作业前完成这些准备。显式选择无法读取或内容无效的配置时，验证失败。

同步接口为 `search(query: str, size: int | None = None)`，环境通过 `asyncio.to_thread` 调用。模型工具调用可以提供以下参数：

```json
{"name": "search", "arguments": {"query": "DeepEyes V2", "size": 3}}
```

`query` 必须包含非空白文字。`size` 可以省略或设为 `null`；显式提供时必须是正整数。结果数量依次使用显式 `size`、配置 `topk`、默认值 `5`。服务可以返回更少的结果。外部 API 可通过 `request.max_size` 设置数量上限，超出上限时在 HTTP 访问前拒绝请求。

成功调用返回以下结构：

```json
{
  "elapsed_time": 0.12,
  "data": [
    {
      "title": "Result title",
      "link": "https://example.org/result",
      "snippet": "Content returned by the selected service.",
      "date": null
    }
  ]
}
```

`elapsed_time` 是以秒为单位的非负数。真实后端计算客户端创建、请求、重试等待、解析及资源清理的耗时。`title`、`link`、`snippet` 为字符串，`date` 为字符串或 `null`。缺少来源 URL 的文档使用空 `link`，环境直接展示标题文字。空结果列表属于成功响应，表示为 `data: []`。

预期的配置、认证、传输及响应验证失败返回 `"Error"`。环境记录 `search_failed`，并生成 `done=False` 的观察，允许 agent 继续执行。非法工具参数生成 `invalid_search_args`。选用真实后端时保持上述失败行为；mock 结果通过默认配置或显式 mock 配置启用。

## Search-R1 retriever

从模板创建本地配置：

```bash
mkdir -p log/search-configs
cp examples/deepeyes_v2_agentic/search_config.retriever.yaml log/search-configs/retriever.yaml
```

执行者编辑 `log/search-configs/retriever.yaml`，将 `endpoint` 设为可以访问的服务地址，包括 `/retrieve`；例如，同一主机上的服务可以使用 `http://127.0.0.1:17389/retrieve`。模板使用 `topk: 3`，超时和重试选项见后续说明。随后选择该文件：

```bash
export DEEPEYES_V2_SEARCH_CONFIG_PATH="$PWD/log/search-configs/retriever.yaml"
python examples/deepeyes_v2_agentic/app/search_runtime.py prepare
```

每次搜索通过 `POST` 请求提交一条查询：

```json
{"queries": ["DeepEyes V2"], "topk": 3, "return_scores": true}
```

预期响应符合 [retrieval_server.py](../search_r1/retrieval_server.py) 的结构：

```json
{
  "result": [
    [
      {
        "document": {"contents": "Document title\nDocument text"},
        "score": 1.0
      }
    ]
  ]
}
```

文档对象也可以直接出现在 `result[0]` 中。每份文档必须包含非空字符串 `contents`，其第一行提供标题，其余文字提供摘要；只有一行时，同时用于标题和摘要。显式提供的 `title`、`link` 或 `url`、`date` 元数据会被保留，`link` 优先于 `url`。统一结果省略评分。[Search-R1 示例](../search_r1/README.md) 介绍服务与语料准备。

## 外部 API 与 Brave

Brave 模板通过环境变量读取有效凭据。以下交互命令读取凭据值，命令与 YAML 文件仅保存变量名称：

```bash
read -r -s BRAVE_SEARCH_API_KEY
export BRAVE_SEARCH_API_KEY
export DEEPEYES_V2_SEARCH_CONFIG_PATH="$PWD/examples/deepeyes_v2_agentic/search_config.brave.yaml"
python examples/deepeyes_v2_agentic/app/search_runtime.py prepare
```

模板使用以下映射，以及公共超时与重试默认值：

```yaml
backend: external
endpoint: https://api.search.brave.com/res/v1/web/search
topk: 5
method: GET
headers:
  Accept: application/json
auth:
  header: X-Subscription-Token
  env: BRAVE_SEARCH_API_KEY
  prefix: ""
request:
  location: query
  query_field: q
  size_field: count
  max_size: 20
  static_fields: {}
response:
  items_path: [web, results]
  optional_items_paths:
    - [web]
  snippet_optional: true
  fields:
    title: [title]
    link: [url]
    snippet: [description]
    date: null
```

其他 API 使用同一个 `external` 后端，通过以下字段配置：

- `endpoint`：完整的 HTTP/HTTPS URL。`method` 支持 `GET` 或 `POST`，`headers` 提供固定 header。`auth.env` 指定凭据环境变量，`auth.header` 指定发送 header，`auth.prefix` 可以设为 `"Bearer "`。无需认证的服务可以省略 `auth`。
- `request`：`location` 支持 `query` 或 `json`，其中 `GET` 必须使用 `query`。`query_field` 与 `size_field` 指定请求最外层字段名称，`static_fields` 提供其他兼容 JSON 的字段，不能与这些名称冲突。查询参数支持标量或由标量组成的列表。`max_size` 可以限制请求数量。
- `response.items_path`：使用对象键路径指定结果列表，`[]` 表示响应本身就是列表。`response.fields` 中的每条路径从单条结果对象读取内容。`title`、`link`、`snippet` 必须得到字符串。
- `response.optional_items_paths`：默认为 `[]`。每项必须是 `items_path` 的非空前缀，可以等于完整路径。仅声明路径的节点缺失或为 `null` 时生成空结果列表；其他节点缺失或类型错误会导致验证失败。Brave 模板声明 `[web]`，因此 `web` 缺失或为 `null` 时生成 `data: []`，`web: {}` 和 `web.results: null` 会导致验证失败。
- `response.snippet_optional`：默认为 `false`。启用后，摘要最终字段缺失或为 `null` 时生成 `snippet: ""` 并保留该条结果。中间节点必须存在且为对象，最终值的其他类型会导致验证失败。Brave 模板对 `description` 启用此规则。
- `response.fields.date`：设为 `null` 时固定生成 `date: null`。配置路径时允许键缺失，或者最终值为字符串或 `null`。非法中间对象或其他最终值类型会导致验证失败。

未知字段、非法类型、请求名称冲突，以及重复或冲突的认证 header，都会导致验证失败。搜索认证使用独立于模型和 agent 配置的变量，启动入口拒绝 `OPENAI_API_KEY`、`RELAX_*` 等保留名称。

## 超时、重试与代理

公共默认值如下：

```yaml
topk: 5
timeout_s: 10.0
max_retries: 2
retry_delay_s: 0.5
retry_max_delay_s: 2.0
trust_env: false
```

`topk` 和 `timeout_s` 必须大于零，重试次数与等待时间必须为非负数。`timeout_s` 分别设置 HTTPX 的连接、读取、写入和获取连接超时。读取超时限制等待每个数据块的时间，因此完整响应耗时可以超过该值；重试与等待也会增加搜索总耗时。参见 [HTTPX 超时说明](https://www.python-httpx.org/advanced/timeouts/)。

最多尝试 `max_retries + 1` 次请求。支持重试的 HTTP 状态为 `408`、`429`、`500`、`502`、`503`、`504`；HTTPX 超时、连接错误、读取错误、写入错误和远端协议错误也支持重试。其他 HTTP 失败、非法 JSON 和非法结果结构会结束调用。客户端禁用重定向。等待时间从 `min(retry_delay_s, retry_max_delay_s)` 开始，每次翻倍并受 `retry_max_delay_s` 限制，默认依次等待 `0.5`、`1.0` 秒。当前策略不读取 `Retry-After`。每次搜索在全部尝试结束后关闭 HTTPX 客户端。

使用环境代理时，在所选本地配置中设置 `trust_env: true`，并导出对应的 `HTTP_PROXY`、`HTTPS_PROXY` 或 `ALL_PROXY`。`NO_PROXY` 指定直接访问的主机。此模式下 HTTPX 也会读取 `SSL_CERT_FILE` 和 `SSL_CERT_DIR`；默认的 `trust_env: false` 忽略这些环境设置。参见 [HTTPX 环境变量说明](https://www.python-httpx.org/environment_variables/)。

## Agent 入口与 Ray

标准和 KLX 训练脚本、`scripts/smoke.sh`、`scripts/run_single_session.py` 会准备所选搜索配置。Shell 包装脚本读取可选的 `env.sh`，直接运行 Python 时继承已经导出的变量。`env.sh` 中的赋值可以替换 Shell 提供的值，因此该文件也需要选择预期搜索配置。

相对配置路径基于启动目录解析。Ray 通过 `runtime_env.env_vars` 接收绝对路径及所选 `auth.env` 的值。每个 worker/agent 节点都需要在相同绝对路径读取内容一致的配置，文件分发由部署过程负责。凭据存在于 Ray 作业运行元数据中，有权访问该元数据的账号能够读取凭据。

既有 `RUNTIME_ENV_JSON` 字段得到保留，环境变量按名称合并，示例生成的配置与搜索变量具有最终优先级。搜索 helper 不自动复制代理和证书变量；需要时通过 worker 环境或 `RUNTIME_ENV_JSON.env_vars` 提供。回环服务或代理地址指向 worker 自身的主机。

[示例 README](./README.md) 说明模型、应用环境、sandbox 和训练准备。标准图片轨迹需要兼容的模型服务及准备好的 sandbox。合成的单次会话问题用于图片裁剪，以下离线场景直接验证搜索分支。配置传递具有示例级回归测试，目标集群还需要独立完成部署验证。

## 离线 smoke 与回归测试

离线 smoke 使用受控模型响应与搜索 transport 执行真正的 agent 循环。导入需要示例的 Python 依赖，包括 OpenAI SDK 和 Pillow；运行无需模型或搜索服务、搜索凭据、GPU 或 sandbox 会话。错误场景提供受控的 `503` 响应，并验证 agent 在重试耗尽后继续完成回答。

```bash
mkdir -p log/search
SEARCH_SMOKE_DIR=$(mktemp -d "$PWD/log/search/offline-XXXXXXXX")
python examples/deepeyes_v2_agentic/scripts/smoke_search_offline.py \
  --scenario mock --output-dir "$SEARCH_SMOKE_DIR/mock"
python examples/deepeyes_v2_agentic/scripts/smoke_search_offline.py \
  --scenario retriever-error --output-dir "$SEARCH_SMOKE_DIR/retriever-error"
python examples/deepeyes_v2_agentic/scripts/smoke_search_offline.py \
  --scenario external-error --output-dir "$SEARCH_SMOKE_DIR/external-error"
```

每次运行保存 `input.json`、`output.json`、`model_requests.json` 和 `report.json`，检查观察、后续消息、最终答案及资源清理。错误场景还会保存受控配置。smoke 脚本允许复用已有目录，因此上述命令每次使用新目录。

示例回归测试命令如下：

```bash
mkdir -p log/search
SEARCH_TEST_DIR=$(mktemp -d "$PWD/log/search/tests-XXXXXXXX")
TMPDIR="$SEARCH_TEST_DIR" python -m pytest \
  tests/examples/deepeyes_v2_agentic -q \
  --basetemp "$SEARCH_TEST_DIR/pytest" -o cache_dir="$SEARCH_TEST_DIR/cache"
```

测试覆盖后端适配、配置、重试与错误、本机 HTTP 超时、agent 观察、入口传递及验证工具。运行需要绑定本机回环端口和启动子进程的权限，无需外部搜索凭据、模型服务或 GPU。这些检查验证示例行为，完整训练及模型与 sandbox 验证各有运行前提。

## 真实服务验证

[verify_search_live.py](./scripts/verify_search_live.py) 独立于模型和 sandbox 调用已配置的 `retriever` 或 `external` 服务。执行者需要提供至少两条不同的非空查询，并使其适合对应语料或服务；同时提供非空部署或版本说明，以及尚不存在的输出目录。版本说明的来源记录为 `operator_supplied`。

已配置 Search-R1 服务的验证命令如下：

```bash
mkdir -p log/search
SEARCH_LIVE_DIR=$(mktemp -d "$PWD/log/search/live-XXXXXXXX")
python examples/deepeyes_v2_agentic/scripts/verify_search_live.py \
  --config log/search-configs/retriever.yaml \
  --service-version 'Search-R1 deployment with the configured corpus and index' \
  --query 'capital of France' --query 'capital of Japan' \
  --output-dir "$SEARCH_LIVE_DIR/retriever"
```

导出 `BRAVE_SEARCH_API_KEY` 后，Brave 验证命令如下：

```bash
mkdir -p log/search
SEARCH_LIVE_DIR=$(mktemp -d "$PWD/log/search/live-XXXXXXXX")
python examples/deepeyes_v2_agentic/scripts/verify_search_live.py \
  --config examples/deepeyes_v2_agentic/search_config.brave.yaml \
  --service-version 'Brave Search web API v1' \
  --query 'Python official documentation' --query 'HTTPX official documentation' \
  --output-dir "$SEARCH_LIVE_DIR/brave"
```

退出状态 `0` 要求每条查询结果非空、服务字段与统一结果完全对应，并通过证据文件检查。失败时返回非零状态。`query-NNN.json` 记录请求尝试、原始响应、统一结果及来源检查；`summary.json` 记录服务说明、服务地址来源、搜索选项、配置与实现 SHA-256，以及证据文件名。验证程序重新读取全部文件并核查完整内容；摘要发布或完整性检查失败时移除 `summary.json`。

来源检查遵循配置中的可选字段规则，包括可选摘要对应的空字符串。合法空结果记录为 `empty_results`，因为真实服务验证要求每条查询取得至少一条结果。

URL 认证信息及所选 `auth.env` 的值会自动脱敏。URL 认证包含 HTTPX 生成的完整 Basic Authorization 值、其中的 Base64 凭据及其 URL 编码形式，服务响应回传这些内容时同样进行脱敏。其他敏感 endpoint 查询参数通过可重复的 `--sensitive-query-param NAME` 声明，固定敏感 header 通过可重复的 `--sensitive-header NAME` 声明。对于 endpoint 参数 `access_token` 和固定 header `X-Internal-Key`，附加参数如下：

```text
--sensitive-query-param access_token --sensitive-header X-Internal-Key
```

声明的名称必须存在于 endpoint 或配置的 `headers` 中，header 名称匹配不区分大小写。名称不存在时，在请求或创建输出目录前失败。普通值保持原内容，显式声明为敏感内容时进行脱敏。脱敏覆盖外部文字、原始响应的键和值，同时保留固定报告字段、摘要及文件引用。脱敏后的键出现名称冲突时，验证失败。
