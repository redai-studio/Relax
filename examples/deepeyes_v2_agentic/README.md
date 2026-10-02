# DeepEyes V2 — agentic example

DeepEyes V2 ([paper](https://arxiv.org/abs/2511.05271),
[upstream](https://github.com/Visual-Agent/DeepEyesV2)) on Relax's agentic
stack. Multimodal RL with three action channels:

- `<code>...</code>` — Python in a Jupyter sandbox (`image_1` pre-loaded)
- `<tool_call>{"name": "search" | "image_search", ...}</tool_call>` — text/image search
- `<answer>...</answer>` — terminate

Reward = `0.8 * acc + 0.2 * format`, accuracy via LLM-judge, routed on
`extra_info.data_source` ∈ {perception, reason, search, vstar-test}.

## One env var: `DATA_DIR`

`scripts/prepare.sh` keeps the high-concurrency app environment on node-local
storage and lays out shared data assets under `DATA_DIR`:

```
${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/
└── .venv/                           (host-side agent environment)

${DATA_DIR}/
├── sif/
│   └── deepeyes_v2_kernel.sif       (~115 MiB, built locally)
└── data/
    ├── raw/                          (~10 GiB, raw HF download — kept for re-conversion)
    │   └── {perception_all_1..5,reason,search,vstar_test}.parquet
    ├── {perception_all_1..5,reason,search,vstar_test}.parquet   (data_source-injected)
    └── smoke.parquet                 (4 synthetic rows, all 4 data_sources)
```

## Configure once: `env.sh`

Machine-local paths / endpoints / proxies / mirrors live in `env.sh`
(gitignored). Every script in this example auto-sources it.

```bash
cp examples/deepeyes_v2_agentic/env.sh.example examples/deepeyes_v2_agentic/env.sh
# edit env.sh — at minimum set DATA_DIR; see comments in the file for the
# optional knobs (OPENAI_BASE_URL, HF_HTTP_PROXY, BOOTSTRAP_FROM_IMAGE, …)
```

## Prep (one-time)

```bash
bash examples/deepeyes_v2_agentic/scripts/prepare.sh
```

What it does (all idempotent — skips done work on re-run):

1. **App environment** — creates a reusable node-local `uv` virtual environment
   with system site packages and installs the host-side `jupyter_client` used
   to control the sandbox kernel. Its default location is
   `/tmp/deepeyes-v2-app-env`; prepare the same path on every training node.
2. **SIF** — `apptainer build` from `apptainer_env/deepeyes_v2_kernel.def`, falls
   back to `--fakeroot`, verifies kernel deps inside the SIF.
3. **Train parquets** — downloads `honglyhly/DeepEyesV2_RL` (8 files, ~10 GiB)
   from the configured Hugging Face endpoint (default `https://huggingface.co`),
   then runs `convert_tool/rl_data_convert.py` to inject
   `extra_info.data_source`.
4. **Smoke parquet** — 4 synthetic rows covering all `data_source` values.

Skip individual steps with `SKIP_SIF=1 SKIP_TRAIN=1 SKIP_SMOKE=1`.

## Smoke (single sample, no Ray)

One trajectory through the full agent app (`app/agent.py` + sandbox +
tools) against an OpenAI-compatible chat endpoint. Run this FIRST to
verify the agent loop / sandbox / message wiring before the cluster
launch. Needs `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `env.sh`:

```bash
bash examples/deepeyes_v2_agentic/scripts/smoke.sh
# or a different row from smoke.parquet:
SMOKE_ROW=2 bash examples/deepeyes_v2_agentic/scripts/smoke.sh
```

Pretty-prints the output JSON + a summary. Healthy run:
`stop_reason=env_done`, `branch_counts.code≥1`, `final_answer` non-null,
`last_error=null`. After: `apptainer instance list` must be empty (else
`env.close()` didn't run on some exit path — file a bug).

For ad-hoc debug with a synthetic input (no parquet needed):

```bash
source examples/deepeyes_v2_agentic/env.sh
${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/.venv/bin/python \
    examples/deepeyes_v2_agentic/scripts/run_single_session.py
```

## Train (cluster)

`DATA_DIR` comes from `env.sh`. Also export `MODEL_DIR` + `SAVE_DIR`:

```bash
export MODEL_DIR=...          # contains Qwen3.6-35B-A3B/ and Qwen2.5-1.5B-Instruct/
export SAVE_DIR=...

bash examples/deepeyes_v2_agentic/run_deepeyes_v2_agentic.sh
```

The launcher auto-resolves `APPTAINER_IMAGE_PATH = ${DATA_DIR}/sif/deepeyes_v2_kernel.sif`
and propagates it to every Ray worker via `--runtime-env-json`. Set
`APPTAINER_IMAGE_PATH` explicitly to override (e.g. shared NFS path for
multi-node).

## Image-search cache (optional, only for the `search` split)

The `<tool_call>image_search</tool_call>` branch hits a precomputed
MMSearch-R1 cache, not live Google. If you skip this, the search backend
returns a benign `Error`, the env surfaces it as `search_failed`, and training
keeps moving — fine for any split that isn't `search`.

To enable it, get the MMSearch raw cache (separate dataset, not bundled) and:

```bash
python examples/deepeyes_v2_agentic/convert_tool/cache_convert.py \
    --input_json_path  ${MMSEARCH_CACHE_JSON} \
    --output_json_path ${DEEPEYES_V2_SEARCH_CACHE_PATHS} \
    --data_path        ${MMSEARCH_IMAGE_ROOT}
```

then `export DEEPEYES_V2_SEARCH_CACHE_PATHS=...` before launching.

## Text search backends

The `<tool_call>search</tool_call>` branch dispatches through a pluggable
backend (`app/search_backends.py`), selected with
`DEEPEYES_V2_SEARCH_BACKEND` (set it in `env.sh`):

| Backend             | Value       | Behaviour                                                                                                                                                                                                                                                    |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Mock (default)      | `mock`      | Deterministic offline snippets — no network, no keys, same query always yields the same results.                                                                                                                                                             |
| Search-R1 retriever | `retriever` | POSTs `{"queries": [...], "topk": k, "return_scores": false}` to any Search-R1 compatible service (e.g. `examples/search_r1/retrieval_server.py`) and maps `result[0]` onto the unified schema (`contents` becomes the snippet fallback).                    |
| External API        | `external`  | POSTs to a generic JSON web-search API. Defaults match the Serper.dev `POST /search` preset (`X-API-KEY` header, `{"q": …, "num": …}` body, results under `organic`); endpoint, auth header, request fields and response field mapping are all configurable. |

All backends return `{"elapsed_time": float, "data": [{"title", "link", "snippet", "date"|None}, ...]}`; on timeout, connection failure, non-2xx
status or malformed payloads the tool returns `"Error"` (after configurable
retries) and the env surfaces it as a clean in-trajectory tool error — the
agent process keeps running. Knobs (all optional, defaults in parentheses):
`DEEPEYES_V2_SEARCH_TOP_K` (5), `DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS` (30 —
this bounds a single HTTP request, so one search call can take up to roughly
`timeout × attempts` plus retry delays, ≈93 s at the defaults; lower
`DEEPEYES_V2_SEARCH_MAX_RETRIES` when rollout latency matters),
`DEEPEYES_V2_SEARCH_MAX_RETRIES` (3), `DEEPEYES_V2_SEARCH_RETRY_DELAY_SECONDS`
(1.0); per-backend: `DEEPEYES_V2_RETRIEVER_URL` (required for `retriever`),
`DEEPEYES_V2_EXTERNAL_SEARCH_ENDPOINT` (required for `external`),
`_API_KEY` / `_AUTH_HEADER` (`X-API-KEY`) / `_QUERY_FIELD` (`q`) /
`_TOPK_FIELD` (`num`) / `_RESULTS_FIELD` (`organic`) / `_FIELD_MAP` (JSON,
e.g. `'{"date": "published_date"}'`). API keys live only in the gitignored
`env.sh` — never in committed code. See `env.sh.example` for a template.
All of these variables are forwarded into the Ray workers by
`run_deepeyes_v2_agentic.sh` / `run_deepeyes_v2_agentic_klx.sh`.

### Verifying against a real service

The unit tests run against a local stub; to exercise a real backend, set the
exports above in `env.sh` and call `search()` once from the repo root:

```bash
# Search-R1 retriever: start the vendored server first (index setup in
# examples/search_r1/README.md, entry point examples/search_r1/run_retriever.sh):
bash examples/search_r1/run_retriever.sh
export DEEPEYES_V2_SEARCH_BACKEND=retriever
export DEEPEYES_V2_RETRIEVER_URL=http://127.0.0.1:17389/retrieve

# Or an external API (Serper.dev shown; the key stays in the gitignored env.sh):
export DEEPEYES_V2_SEARCH_BACKEND=external
export DEEPEYES_V2_EXTERNAL_SEARCH_ENDPOINT=https://google.serper.dev/search
export DEEPEYES_V2_EXTERNAL_SEARCH_API_KEY=...

python -c "import sys, json; sys.path.insert(0, 'examples/deepeyes_v2_agentic'); from app.search_utils import search; print(json.dumps(search('relax'), ensure_ascii=False)[:500])"
```

## Layout

| Path                             | Role                                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `env.sh.example`                 | Template for the gitignored `env.sh` — set DATA_DIR + optional knobs                     |
| `scripts/prepare.sh`             | Single prep entry point — app environment + SIF + train data + smoke parquet             |
| `scripts/prepare_app_env.sh`     | Reusable host-side agent environment                                                     |
| `scripts/build_smoke_parquet.py` | Synthetic 4-row parquet generator (called by prepare.sh; also runnable standalone)       |
| `scripts/run_single_session.py`  | Single-trajectory harness (parquet row or synthetic input)                               |
| `app/agent.py`                   | Per-session agent driver                                                                 |
| `app/env_deepeyes_v2.py`         | Tool handlers (exec_code / exec_tool / close)                                            |
| `app/prompt.py`                  | Observation templates + sandbox init code                                                |
| `app/search_utils.py`            | Text + image-search entry points (`search` dispatches to `app/search_backends.py`)       |
| `app/search_backends.py`         | Pluggable text-search backends: mock (default) / Search-R1 retriever / external API      |
| `app/sandboxes/`                 | Jupyter sandbox abstraction + apptainer backend                                          |
| `reward_deepeyes_v2.py`          | Post-trajectory scorer (data_source-routed, LLM-judge)                                   |
| `convert_tool/`                  | `rl_data_convert.py` (data_source injection) + `cache_convert.py` (search cache rewrite) |
| `apptainer_env/`                 | Apptainer image def + sandbox YAML config                                                |
| `run_deepeyes_v2_agentic.sh`     | Full GRPO launch (Qwen3.6-35B-A3B, colocate)                                             |
| `scripts/smoke.sh`               | Single-sample smoke wrapper (one trajectory through the full app, no Ray)                |
| `run_agent_app.sh`               | Per-session wrapper invoked by Relax for each rollout                                    |
| `sglang_judge_service.sh`        | Stands up the LLM-judge SGLang server                                                    |

## Pitfalls — read before debugging

Adapting DeepEyes V2 on Relax's agentic stack has a set of recurring
footguns (SGLang mamba IMA, "step 1 keeps looping" caused by
`--use-fault-tolerance` silently masking errors,
reward tool-bonus divergence from upstream, …). Before opening py-spy,
read [`PITFALLS.md`](./PITFALLS.md) — the top entry (don't enable
`--use-fault-tolerance` during adaptation) alone will save hours.

## How it differs from `examples/deepeyes_agentic/` (V1)

Three action branches vs V1's single `image_zoom_in_tool`; stateful Jupyter
sandbox per trajectory; reward routed on data_source. Same agentic stack +
OpenAI-SDK driver pattern.

## Phase 1 scope notes

- Only the **apptainer** backend ships in Phase 1; `nexsandbox_backend.py` is
  intentionally absent (Phase 1.5 plan in `docs/superpowers/plans/`).
- Sandbox abstraction is example-local at `app/sandboxes/`; will move to
  `relax/runtime/sandbox/` in Phase 1.5 once smoke validates the design.
- Cold-start SFT out of scope (Phase 2). First runs will see low reward
  without a community V2 SFT checkpoint.
