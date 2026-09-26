# DeepEyes V2 text search

[中文](./SEARCH.zh-CN.md) · [Example setup](./README.md)

## Offline quick start

Run these Bash commands from the Relax repository root in a Python environment with the project dependencies installed. Search reuses HTTPX, Pydantic and PyYAML from [requirements.txt](../../requirements.txt).

With `DEEPEYES_V2_SEARCH_CONFIG_PATH` unset, text `search()` uses deterministic `mock` results. This call needs no network, search credentials, model service or sandbox:

```bash
unset DEEPEYES_V2_SEARCH_CONFIG_PATH
python -c 'import json, sys; from examples.deepeyes_v2_agentic.app.search_utils import search; sys.stdout.write(json.dumps(search("DeepEyes V2", size=2), ensure_ascii=False) + "\n")'
```

The same query and result count produce identical mock results. Each has a `[mock]` title, a snippet containing the query, a query-specific example URL and `date: null`; `elapsed_time` is `0.0`.

## Configuration and results

`DEEPEYES_V2_SEARCH_CONFIG_PATH` selects a YAML file. Templates are provided for [mock](./search_config.mock.yaml), [Search-R1 retriever](./search_config.retriever.yaml) and [Brave Search](./search_config.brave.yaml). For an explicit offline selection:

```bash
export DEEPEYES_V2_SEARCH_CONFIG_PATH="$PWD/examples/deepeyes_v2_agentic/search_config.mock.yaml"
python examples/deepeyes_v2_agentic/app/search_runtime.py prepare
```

`prepare` validates the file and selected authentication variable, then prints the absolute configuration path. Entry scripts perform this preparation before launching the agent or submitting a Ray job. An explicitly selected unreadable or invalid configuration fails validation.

The synchronous API is `search(query: str, size: int | None = None)`; the environment calls it through `asyncio.to_thread`. A model tool call can supply:

```json
{"name": "search", "arguments": {"query": "DeepEyes V2", "size": 3}}
```

`query` must contain non-whitespace text. `size` can be omitted or `null`; when supplied, it must be a positive integer. Result count precedence is explicit `size`, configuration `topk`, then the default `5`. Services may return fewer results. An external API's optional `request.max_size` rejects larger requests before HTTP access.

Successful calls use this structure:

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

`elapsed_time` is a nonnegative number of seconds. Real backends measure client creation, requests, retry waits, parsing and cleanup. `title`, `link` and `snippet` are strings; `date` is a string or `null`. A missing document URL becomes an empty `link`, displayed as plain title text. An empty result list is a successful `data: []` response.

Expected configuration, authentication, transport and response-validation failures return `"Error"`. The environment records `search_failed` with `done=False`, allowing the agent to continue. Invalid tool arguments produce `invalid_search_args`. Selecting a real backend keeps these failure semantics; mock results require the default or an explicit mock configuration.

## Search-R1 retriever

Create a local configuration from the template:

```bash
mkdir -p log/search-configs
cp examples/deepeyes_v2_agentic/search_config.retriever.yaml log/search-configs/retriever.yaml
```

Edit `log/search-configs/retriever.yaml`: set `endpoint` to the reachable service URL including `/retrieve`, such as `http://127.0.0.1:17389/retrieve` for a service on the same host. The template uses `topk: 3`; timeout and retry options are described below. Then select the file:

```bash
export DEEPEYES_V2_SEARCH_CONFIG_PATH="$PWD/log/search-configs/retriever.yaml"
python examples/deepeyes_v2_agentic/app/search_runtime.py prepare
```

Each search sends one query in a `POST` request:

```json
{"queries": ["DeepEyes V2"], "topk": 3, "return_scores": true}
```

The expected response follows [retrieval_server.py](../search_r1/retrieval_server.py):

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

Documents may also appear directly in `result[0]`. Each requires a nonempty string `contents`; its first line supplies the title and remaining text supplies the snippet. A single line supplies both. Explicit `title`, `link` or `url`, and `date` metadata are preserved; `link` takes precedence over `url`. Scores are omitted from normalized results. The [Search-R1 example](../search_r1/README.md) describes service and corpus preparation.

## External APIs and Brave

The Brave template reads a valid credential from the environment. The following interactive command reads the value without storing it in the command or YAML file:

```bash
read -r -s BRAVE_SEARCH_API_KEY
export BRAVE_SEARCH_API_KEY
export DEEPEYES_V2_SEARCH_CONFIG_PATH="$PWD/examples/deepeyes_v2_agentic/search_config.brave.yaml"
python examples/deepeyes_v2_agentic/app/search_runtime.py prepare
```

The template uses this mapping, together with the common timeout and retry defaults:

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

Other APIs use the same `external` backend with these fields:

- `endpoint`: a complete HTTP/HTTPS URL. `method` accepts `GET` or `POST`; `headers` supplies fixed headers. `auth.env` names the credential environment variable, `auth.header` selects its header, and `auth.prefix` can be `"Bearer "`. Services without authentication can omit `auth`.
- `request`: `location` accepts `query` or `json`; `GET` requires `query`. `query_field` and `size_field` name top-level request fields. `static_fields` adds JSON-compatible values without conflicting with those names. Query parameters accept scalars or lists of scalars. `max_size` optionally limits the requested count.
- `response.items_path`: an object-key path to the result list; `[]` selects a response that is itself a list. Each path in `response.fields` reads from one result object. `title`, `link` and `snippet` must resolve to strings.
- `response.optional_items_paths`: defaults to `[]`. Each entry must be a nonempty prefix of `items_path`, including the complete path. Only a missing or null node at a declared path produces an empty result list; other missing nodes and invalid types fail validation. The Brave template declares `[web]`, so missing/null `web` produces `data: []`, while `web: {}` and `web.results: null` fail validation.
- `response.snippet_optional`: defaults to `false`. When enabled, a missing/null final snippet field produces `snippet: ""` and preserves the result. Intermediate nodes must exist and be objects; other final types fail validation. The Brave template enables this for `description`.
- `response.fields.date`: `null` always produces `date: null`. A configured path permits a missing key or a string/null final value. An invalid intermediate object or another final type fails validation.

Unknown fields, invalid types, conflicting request names and duplicate or conflicting authentication headers fail validation. Search authentication uses a separate variable from model and agent configuration: entry scripts reject reserved names such as `OPENAI_API_KEY` and `RELAX_*`.

## Timeouts, retries and proxies

Common defaults are:

```yaml
topk: 5
timeout_s: 10.0
max_retries: 2
retry_delay_s: 0.5
retry_max_delay_s: 2.0
trust_env: false
```

`topk` and `timeout_s` must be positive; retry counts and delays must be nonnegative. `timeout_s` sets each HTTPX connect, read, write and connection-pool timeout. Reads limit the wait for each data chunk, so total response time can exceed this value. Retries and waits also contribute to total search time. See [HTTPX timeout behavior](https://www.python-httpx.org/advanced/timeouts/).

At most `max_retries + 1` requests are attempted. Retryable statuses are `408`, `429`, `500`, `502`, `503` and `504`; HTTPX timeouts, connection/read/write errors and remote protocol errors are also retried. Other HTTP failures, malformed JSON and invalid result structures end the call. Redirects are disabled. The wait starts at `min(retry_delay_s, retry_max_delay_s)` and doubles up to `retry_max_delay_s`; default waits are `0.5` and `1.0` seconds. `Retry-After` is not consumed. Each search closes its HTTPX client after all attempts.

For an environment proxy, set `trust_env: true` in the selected local configuration and export the appropriate `HTTP_PROXY`, `HTTPS_PROXY` or `ALL_PROXY`. `NO_PROXY` excludes hosts from proxy use. HTTPX also reads `SSL_CERT_FILE` and `SSL_CERT_DIR` in this mode. The default `trust_env: false` ignores these settings. See [HTTPX environment variables](https://www.python-httpx.org/environment_variables/).

## Agent entry points and Ray

The standard and KLX training scripts, `scripts/smoke.sh` and `scripts/run_single_session.py` prepare the selected search configuration. Shell wrappers source the optional `env.sh`; direct Python invocations inherit exported variables. Assignments in `env.sh` can replace shell-provided values, so that file must select the intended search configuration too.

Relative configuration paths resolve against the launch directory. Ray receives the absolute path and selected `auth.env` value through `runtime_env.env_vars`. Each worker/agent node needs the same configuration at that absolute path; file distribution is a deployment responsibility. Credentials are present in Ray job runtime metadata and are readable by accounts with access to it.

Existing `RUNTIME_ENV_JSON` fields are retained, with environment variables merged by name; generated example/search values take precedence. The search helper does not automatically copy proxy and certificate variables. When needed, provide them in worker environments or `RUNTIME_ENV_JSON.env_vars`. A loopback service/proxy address refers to the worker's own host.

The [example README](./README.md) covers model, application environment, sandbox and training setup. Standard image trajectories require a compatible model endpoint and the prepared sandbox. The synthetic single-session question exercises image cropping; the offline scenarios below directly exercise search. Runtime propagation has example-level regression coverage; a target cluster needs its own deployment validation.

## Offline smoke and regression tests

Offline smoke runs the actual agent loop with controlled model responses and search transport. Imports require the example's Python dependencies, including OpenAI SDK and Pillow. It needs no running model/search service, search credential, GPU or sandbox session. Error scenarios return controlled `503` responses and verify that the agent completes its answer after retry exhaustion.

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

Each run writes `input.json`, `output.json`, `model_requests.json` and `report.json`, checking observations, follow-up messages, the final answer and cleanup. Error scenarios also save their controlled configuration. These commands use new directories because the smoke script permits reuse of an existing directory.

Run the example regression suite with:

```bash
mkdir -p log/search
SEARCH_TEST_DIR=$(mktemp -d "$PWD/log/search/tests-XXXXXXXX")
TMPDIR="$SEARCH_TEST_DIR" python -m pytest \
  tests/examples/deepeyes_v2_agentic -q \
  --basetemp "$SEARCH_TEST_DIR/pytest" -o cache_dir="$SEARCH_TEST_DIR/cache"
```

Tests cover backend adaptation, configuration, retries and errors, local HTTP timeouts, agent observations, entry-point propagation and verification tools. They require loopback socket and subprocess permissions, with no external search credentials, model service or GPU. These checks exercise the example; full training and model/sandbox validation have separate prerequisites.

## Live-service verification

[verify_search_live.py](./scripts/verify_search_live.py) calls the configured `retriever` or `external` service independently of the model and sandbox. Supply at least two different nonempty queries that match the corpus or service, a nonempty deployment/version description, and an output directory that does not exist. The version description is recorded as `operator_supplied`.

For the configured Search-R1 service:

```bash
mkdir -p log/search
SEARCH_LIVE_DIR=$(mktemp -d "$PWD/log/search/live-XXXXXXXX")
python examples/deepeyes_v2_agentic/scripts/verify_search_live.py \
  --config log/search-configs/retriever.yaml \
  --service-version 'Search-R1 deployment with the configured corpus and index' \
  --query 'capital of France' --query 'capital of Japan' \
  --output-dir "$SEARCH_LIVE_DIR/retriever"
```

For Brave, after exporting `BRAVE_SEARCH_API_KEY`:

```bash
mkdir -p log/search
SEARCH_LIVE_DIR=$(mktemp -d "$PWD/log/search/live-XXXXXXXX")
python examples/deepeyes_v2_agentic/scripts/verify_search_live.py \
  --config examples/deepeyes_v2_agentic/search_config.brave.yaml \
  --service-version 'Brave Search web API v1' \
  --query 'Python official documentation' --query 'HTTPX official documentation' \
  --output-dir "$SEARCH_LIVE_DIR/brave"
```

Exit status `0` requires nonempty results, exact correspondence between service fields and normalized results, and successful artifact checks for every query. Failure returns a nonzero status. `query-NNN.json` records attempts, raw responses, normalized results and source checks; `summary.json` records the service description, endpoint origin, search options, configuration/implementation SHA-256 and evidence filenames. The verifier rereads every artifact to check its complete content; summary publication or integrity failure removes `summary.json`.

Source checks apply the configured optional-field rules, including empty strings for optional snippets. A valid empty search response is recorded as `empty_results` because live verification requires at least one result per query.

URL authentication and the configured `auth.env` value are automatically redacted. URL authentication includes the complete HTTPX Basic Authorization value, its Base64 credentials and their URL-encoded forms when echoed in responses. Declare additional sensitive endpoint query parameters with repeated `--sensitive-query-param NAME` arguments and fixed sensitive headers with repeated `--sensitive-header NAME` arguments. For an endpoint parameter `access_token` and a fixed `X-Internal-Key` header, append:

```text
--sensitive-query-param access_token --sensitive-header X-Internal-Key
```

These names must exist in the endpoint or configured `headers`; header matching is case-insensitive. Unknown names fail before requests or output-directory creation. Ordinary values remain intact unless declared sensitive. Redaction covers external text and raw-response keys/values while preserving fixed report fields, hashes and file references. A collision between redacted keys fails verification.
