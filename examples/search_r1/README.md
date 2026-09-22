# Search-R1

This example runs the Search-R1 `<search>`, `<information>`, and `<answer>` protocol through Relax managed agent sessions. It supports two modes:

- `vanilla`: one Search-R1 trajectory produces one training row.
- `multiagent`: the main trajectory delegates each search request to an independent searcher context. Training exports
  `main + searcher_*`; evaluation exports `main`.

Each mode uses the same mode-specific loop for training and evaluation. Both loops share the process entrypoint,
protocol, retriever client, and scoring code. The behavior phase performs generation, retrieval, delegation, and
termination without reading gold answers. After the behavior phase finishes, the same agent process scores the
completed main response and builds the Relax output.

See [Multi-Agent Training](./MULTIAGENT.md) for the explicit export contract, identity semantics, role-aware
advantages, and operational risks.

## Components

```text
examples/search_r1/
├── app/
│   ├── agent.py
│   ├── multiagent.py
│   ├── protocol.py
│   ├── retriever.py
│   ├── search_r1_config.yaml
│   ├── scoring.py
│   └── vanilla.py
├── scripts/
│   ├── prepare_data.py
│   ├── prepare_online_assets.sh
│   └── prepare_retriever_env.sh
├── advantage_search_r1.py
├── eval_search_r1.py
├── MULTIAGENT.md
├── retrieval_server.py
├── run_agent_app.sh
├── run_retriever.sh
└── run_search_r1.sh
```

`app/vanilla.py` contains the complete vanilla behavior and output path. `app/agent.py` handles process entry, client
setup, and explicit mode dispatch.

## Data contract

The upstream parquet files keep aliases in `golden_answers`, while Relax managed agents receive the configured
metadata column. Prepare a compact copy containing `prompt` and `extra_info`; the latter receives an `answers` field:

```bash
export SEARCH_R1_DATA_ROOT=/path/to/search_r1
python examples/search_r1/scripts/prepare_data.py \
  --input-dir "${SEARCH_R1_DATA_ROOT}/qa/nq_hotpotqa_train" \
  --output-dir "${SEARCH_R1_DATA_ROOT}/qa/nq_hotpotqa_train_relax"
```

The training entrypoint reads:

```text
--input-key prompt
--metadata-key extra_info
```

Evaluation reads the prepared copy of the single official `test.parquet`. Its source labels remain in
`extra_info.data_source`, and the eval logging hook reports both the aggregate Search-R1 score and
`eval/<data_source>` metrics from the same rollout results.

It does not register `--custom-prompt-path` or `--custom-rm-path`. Each prepared session receives:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "metadata": {
    "index": 0,
    "split": "train",
    "data_source": "nq",
    "answers": ["gold alias"]
  }
}
```

`prepare_online_assets.sh` downloads immutable revisions of the official QA parquet files, Wiki18 corpus, E5 Flat
index, and E5 model, then prepares the Relax parquet files automatically:

```bash
export SEARCH_R1_DATA_ROOT=/path/to/search_r1
bash examples/search_r1/scripts/prepare_online_assets.sh
```

The script uses these upstream artifacts:

- `PeterJinGo/nq_hotpotqa_train@b7d80abfee334a7a91cb377544f09180d58b34f6`
- `PeterJinGo/wiki-18-corpus@69c1c00ffe7c5554c68d8548355cb22e46aabc51`
- `PeterJinGo/wiki-18-e5-index@a4d31160a035f30764604f4827cd8f1d0315eb86`
- `intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd`

The official `wiki-18.jsonl.gz` artifact is a gzip-compressed tar archive. The script extracts
`data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl` as `wiki-18.jsonl`. The E5 Flat index is the
concatenation of the official `part_aa` and `part_ab` files in that order. The QA conversion preserves each official
`prompt` and `extra_info` value and copies `golden_answers` into `extra_info.answers`. The E5 download contains the
config, safetensors weights, and tokenizer files consumed by `AutoModel` and `AutoTokenizer`.

Expected layout:

```text
${SEARCH_R1_DATA_ROOT}/
├── qa/
│   ├── nq_hotpotqa_train/{train,test}.parquet
│   └── nq_hotpotqa_train_relax/{train,test}.parquet
├── retrieval/wiki18_e5_flat/
│   ├── wiki-18.jsonl.gz
│   ├── wiki-18.jsonl
│   ├── part_aa
│   ├── part_ab
│   └── e5_Flat.index
└── models/e5-base-v2/
```

## Agent and reward flow

The managed agent executes these phases in one process:

1. Run the vanilla or multiagent behavior loop using messages, the Relax Chat API, and `/retrieve`.
2. Score the completed main continuation against `metadata.answers`.
3. Return reward and metadata to Relax.

The vanilla loop follows the upstream environment transition contract. Each of four regular turns parses the first
complete `<search>` or `<answer>` action. A search appends the upstream `Doc N(Title: ...)` observation, while an
invalid action appends the upstream corrective observation and consumes one turn before generation resumes. A still
active trajectory receives one final generation after the regular turns; a search emitted by that final generation
ends the trajectory without another retrieval. Each generation is capped at 1000 tokens. Oversized observations are
limited to a configurable 3000-character prefix in the lightweight agent process.

The score follows the original Search-R1 EM reward:

```text
score = 1                          if exact match succeeds with a valid protocol
      = 1 - structure_weight       if exact match succeeds with an invalid protocol
      = structure_weight           if the protocol is valid but the answer is incorrect
      = final_weight               if an invalid protocol contains a final answer
      = 0                          otherwise
```

`structure_weight` and `final_weight` in `app/search_r1_config.yaml` default to `0.0`. Exact match uses the Search-R1
normalization rules: lowercase text, remove ASCII punctuation and English articles, and collapse whitespace.

Vanilla returns one implicit output with an agent-computed reward:

```json
{
  "metadata": {"role": "main", "score/main": 1.0},
  "reward": 1.0
}
```

If the final committed response must be selected explicitly after a context-length failure, vanilla emits the same
information as one named `main` record.

In multiagent mode, every main search query still retrieves the same raw Top-k documents as the vanilla loop. An
independent searcher concurrently investigates the delegated query. A successfully completed searcher finding is
appended to the main observation after the direct retrieval evidence; a failed searcher cannot remove that evidence.
Training exports `main + searcher_*` as explicit JSONL records for the main context and every searcher started by that
session. The custom advantage hook derives each searcher's credit from the main outcome without copying that outcome
into searcher metadata. Evaluation exports the main record. The exact export schema, identity semantics, credit
assignment, and risks are documented in [Multi-Agent Training](./MULTIAGENT.md).

## Retriever

Prepare a uv environment on the retriever GPU host. It inherits the Relax host environment and adds GPU FAISS to
its own venv:

```bash
export SEARCH_R1_DATA_ROOT=/path/to/search_r1
export SEARCH_R1_RETRIEVER_ENV_ROOT=/path/to/node-local/search-r1-retriever
bash examples/search_r1/scripts/prepare_retriever_env.sh
```

The service exposes `POST /retrieve` and accepts:

```json
{"queries":["query"],"topk":3}
```

Set the training host endpoint to the retriever host, for example:

```bash
export SEARCH_R1_RETRIEVER_URL=http://retriever-host:17389/retrieve
```

## Training

For a local single-node run, initialize Ray before launching the retriever. Keep the following commands in the same
shell so `run_search_r1.sh` reuses the initialized Ray environment:

```bash
cd /path/to/Relax
source scripts/entrypoint/local.sh

export MODEL_DIR=/path/to/models
export SEARCH_R1_DATA_ROOT=/path/to/search_r1
export SEARCH_R1_RETRIEVER_ENV_ROOT=/path/to/node-local/search-r1-retriever
export SEARCH_R1_RETRIEVER_URL=http://127.0.0.1:17389/retrieve

tmux new-session -d -s search-r1-retriever \
  "cd '$PWD' && SEARCH_R1_DATA_ROOT='$SEARCH_R1_DATA_ROOT' \
  SEARCH_R1_RETRIEVER_ENV_ROOT='$SEARCH_R1_RETRIEVER_ENV_ROOT' \
  bash examples/search_r1/run_retriever.sh"

curl -X POST "${SEARCH_R1_RETRIEVER_URL}" \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Who wrote Hamlet?"],"topk":3}'

# Choose one training mode:
# Vanilla exports one Search-R1 trajectory per session.
SEARCH_R1_MODE=vanilla bash examples/search_r1/run_search_r1.sh

# Multi-agent exports the main trajectory and its adopted searcher trajectories.
SEARCH_R1_MODE=multiagent bash examples/search_r1/run_search_r1.sh
```

On a managed Ray cluster, initialize or connect to the cluster first, then start `run_retriever.sh`, verify the
`POST /retrieve` endpoint, and submit either training mode.

The shared entrypoint uses Qwen2.5-7B on eight GPUs with eight samples per prompt, logical global batch size 128,
dynamic batching, per-token loss, TIS, and ClearML. Project and experiment names include `vanilla` or `multiagent`.
The managed agent and its Chat Completions client use a timeout of `9999` seconds so prelaunch, queueing, and long
rollouts do not terminate a healthy session. Retrieval has its own `retriever_timeout_s` setting and defaults to 30
seconds.

A full Search-R1 evaluation contains 51,713 samples and may take longer than the actor's default 1800-second wait.
Before enabling full evaluation, change the default value of `max_wait_seconds` in
`relax/backends/megatron/actor.py::_wait_for_previous_eval()` from `1800` to `7200`. This two-hour limit covers the
29–57 minute evaluation times observed on a single eight-GPU node and leaves time for the multi-gigabyte eval dump to
finish writing.

Agent behavior is configured in `app/search_r1_config.yaml`:

| Key                         | Default | Meaning                                                  |
| --------------------------- | ------: | -------------------------------------------------------- |
| `retriever_topk`            |     `3` | Documents returned per query                             |
| `retriever_timeout_s`       |    `30` | Retrieval timeout in seconds                             |
| `max_observation_chars`     |  `3000` | Character limit for one environment observation          |
| `max_search_turns`          |     `4` | Main or vanilla search budget                            |
| `searcher_max_search_turns` |     `1` | Search budget for each delegated searcher                |
| `structure_weight`          |   `0.0` | Reward for a valid protocol with an incorrect answer     |
| `final_weight`              |   `0.0` | Reward for an invalid protocol containing a final answer |

The launcher accepts generic run controls through `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`,
`GLOBAL_BATCH_SIZE`, `EVAL_INTERVAL`, `PROJECT_NAME`, `EXP_NAME`, and `DUMP_DETAILS_DIR`. `SEARCH_R1_MODE`,
`SEARCH_R1_DATA_ROOT`, and `SEARCH_R1_RETRIEVER_URL` select the application mode and required assets.

## Attribution

The prompt, action loop, answer normalization, EM reward, and retrieval protocol are adapted from
[PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) under Apache-2.0. The source files are
`search_r1/llm_agent/generation.py`, `verl/utils/reward_score/qa_em.py`, and
`search_r1/search/retrieval_server.py` at commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`.
