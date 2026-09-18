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

## Web-search backend (text `search` tool)

The `<tool_call>search</tool_call>` tool dispatches to a pluggable backend
(`app/search_backends.py`), selected via `DEEPEYES_V2_SEARCH_BACKEND`:

| Backend   | Value       | Notes                                                                        |
| --------- | ----------- | ---------------------------------------------------------------------------- |
| Mock      | `mock`      | Default. Deterministic canned snippets — fully offline, no keys, no service. |
| Retriever | `retriever` | Search-R1 compatible `POST /retrieve` service (see `examples/search_r1`).    |
| External  | `external`  | Any external search API via a JSON config (built-in defaults target Serper). |

All backends return the unified shape
`{"elapsed_time", "data": [{"title", "link", "snippet", "date" | null}]}`.
Timeouts, server errors and malformed responses are retried
(`DEEPEYES_V2_SEARCH_MAX_RETRIES`, default 3) and then degrade to the env's
`Error` → `search_failed` convention — the agent process never crashes.

### Environment variables

| Variable                              | Default                           | Scope                        |
| ------------------------------------- | --------------------------------- | ---------------------------- |
| `DEEPEYES_V2_SEARCH_BACKEND`          | `mock`                            | backend selection            |
| `DEEPEYES_V2_SEARCH_TIMEOUT_SECONDS`  | `10`                              | HTTP timeout (real backends) |
| `DEEPEYES_V2_SEARCH_MAX_RETRIES`      | `3`                               | retry attempts               |
| `DEEPEYES_V2_SEARCH_RETRIEVER_URL`    | `http://127.0.0.1:17389/retrieve` | retriever endpoint           |
| `DEEPEYES_V2_SEARCH_RETRIEVER_TOPK`   | the tool's `size` (5)             | retriever top-k              |
| `DEEPEYES_V2_SEARCH_EXTERNAL_CONFIG`  | built-in Serper mapping           | external JSON config file    |
| `DEEPEYES_V2_SEARCH_EXTERNAL_API_KEY` | none                              | external API key (env only)  |

The launcher propagates all of them to every Ray worker.

### Example: Search-R1 retriever

```bash
export DEEPEYES_V2_SEARCH_BACKEND=retriever
export DEEPEYES_V2_SEARCH_RETRIEVER_URL=http://127.0.0.1:17389/retrieve
export DEEPEYES_V2_SEARCH_RETRIEVER_TOPK=5
```

### Example: Serper (external)

```bash
export DEEPEYES_V2_SEARCH_BACKEND=external
export DEEPEYES_V2_SEARCH_EXTERNAL_API_KEY="$SERPER_API_KEY"
```

To adapt a different API, point `DEEPEYES_V2_SEARCH_EXTERNAL_CONFIG` at a JSON
file overriding any subset of the defaults — endpoint, HTTP method, auth
header/scheme, request field mapping and response field mapping:

```json
{
  "endpoint": "https://google.serper.dev/search",
  "method": "POST",
  "auth_header": "X-API-KEY",
  "auth_scheme": "",
  "request_map": { "query": "q", "size": "num" },
  "response_map": { "results": "organic", "title": "title", "link": "link", "snippet": "snippet", "date": "date" }
}
```

`response_map.results` is a dot path to the result list (e.g. `"data.items"`);
the other entries rename each per-result field. Keep the API key in the env
var — never in the config file or the repo.

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
| `app/search_utils.py`            | `search()` dispatch (retry + `Error` convention) + image-search cache                    |
| `app/search_backends.py`         | Pluggable text-search backends (mock / retriever / external)                             |
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
