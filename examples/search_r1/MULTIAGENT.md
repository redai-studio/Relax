# Search-R1 Multi-Agent Training

This guide describes the multi-agent mode implemented by `examples/search_r1`. It covers the managed-session flow,
explicit JSONL export, logical identity, role-aware advantage, configuration, and known risks.

## Overview

Search-R1 multi-agent mode keeps the original main `<search>/<information>/<answer>` trajectory and adds an independent
searcher context for each valid main search action. Direct retrieval and the searcher run concurrently. The direct Top-k
documents always enter the main observation; a successful searcher finding is appended to that observation.

Training exports the main context and every searcher context started by that session. Evaluation exports the main
context alone, so task EM remains comparable with vanilla Search-R1. Training and evaluation execute the same
multi-agent loop and agent-side scoring implementation.

## Architecture

```text
┌────────────────────┐
│ Relax managed input│
└─────────┬──────────┘
          ▼
┌────────────────────┐       ┌────────────────────┐
│ Main Search-R1 loop│──────>│ Direct retriever   │
└─────────┬──────────┘       └─────────┬──────────┘
          │ search query               │ Top-k documents
          ├───────────────────┐        │
          ▼                   │        │
┌─────────────────────┐        │        │
│ Independent searcher│       │        │
└─────────┬───────────┘        │        │
          │ finding           │        │
          └──────────────┬────┴────────┘
                         ▼
              ┌────────────────────┐
              │ Main observation   │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Agent-side scoring │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Explicit JSONL     │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Role-aware GRPO    │
              └────────────────────┘
```

The implementation is split across:

| Component                | Responsibility                                              |
| ------------------------ | ----------------------------------------------------------- |
| `app/agent.py`           | Session input, client setup, and mode dispatch              |
| `app/multiagent.py`      | Main and searcher loops with explicit export                |
| `app/protocol.py`        | Search-R1 action parsing and Chat Completions requests      |
| `app/retriever.py`       | Session-local asynchronous retrieval client                 |
| `app/scoring.py`         | Search-R1 scoring and main metadata                         |
| `app/vanilla.py`         | Standalone vanilla loop and output                          |
| `advantage_search_r1.py` | Prompt-group normalization and role-aware credit assignment |
| `run_search_r1.sh`       | Shared vanilla and multi-agent training configuration       |

## Explicit Export Contract

The managed command writes newline-delimited JSON records to `RELAX_OUTPUT_JSON`, with one context per JSONL record.
Each record supports these fields:

| Field                  | Required | Meaning                                                    |
| ---------------------- | -------- | ---------------------------------------------------------- |
| `name`                 | Yes      | Non-empty role identifier unique within one session export |
| `messages`             | Yes      | Complete history ending at a committed assistant response  |
| `tools`                | No       | Tool schema used to render that history                    |
| `chat_template_kwargs` | No       | Template arguments used to render that history             |
| `metadata`             | No       | Sample metadata consumed by metrics and custom advantage   |
| `reward`               | No       | Numeric, object, or null raw reward                        |

Search-R1 training emits records shaped like:

```jsonl
{"name":"main","messages":[...],"metadata":{"role":"main","score/main":1.0,"reward/shaped":1.0},"reward":1.0}
{"name":"searcher_0","messages":[...],"metadata":{"role":"searcher"}}
```

Evaluation emits the `main` record and omits searcher records. Vanilla mode normally uses an implicit object containing
`metadata` and `reward`; it switches to one explicit `main` record when a committed response must be selected after a
context-length error.

Relax matches each explicit record to a committed SessionForest state using `messages`, `tools`, and
`chat_template_kwargs`. Exported history must preserve the exact message structure used during live generation.

## Logical Identity and Batching

One managed session is one logical training identity. In multi-agent mode that identity can produce a variable number
of physical rows:

```text
session A -> main, searcher_0, searcher_1
session B -> main
session C -> main, searcher_0
```

Rows from the same session share `Sample.index`; prompt samples generated from the same input share
`Sample.group_index`. These fields have different roles:

- `group_index` identifies the GRPO prompt cohort used by group reward and advantage.
- `index` keeps every row from one managed session together through transfer and training.

Variable fanout requires `--use-dynamic-batch-size`. The Search-R1 entrypoint enables dynamic batching and
`--calculate-per-token-loss`. Its `--global-batch-size` counts logical identities rather than exported physical rows.
The recipe uses the default policy-gradient reducer.

Vanilla mode exports one row per identity. It therefore exercises the M=1 degradation path while sharing the same
resident agentic pipeline and role-aware advantage implementation.

## Reward and Advantage

The agent process computes the final main score after behavior finishes. Gold aliases are read during this scoring
phase and are never passed to model generation or retrieval decisions.

The main record contains:

- `score/main`: exact-match outcome;
- `reward/shaped`: exact match plus the configured structure/final-format shaping;
- protocol, search, and termination metrics;
- `reward`: exact match during eval and shaped reward during training.

Each searcher record contains its role, search count, and termination reason. It has no top-level reward and the recipe
does not calculate an independent gold answer for a delegated searcher.

`advantage_search_r1.py` performs the training credit assignment:

1. Read the main `reward/shaped` value from every session slot in a prompt group.
2. Drop the group when all main values are identical.
3. Z-normalize main values with sample standard deviation.
4. Give each main row scale `1.0`.
5. Split a scalar advantage scale of `0.5` equally across every searcher row exported by that main session.

The custom advantage hook returns `name -> scalar advantage`, allowing Relax to route each value back to its exported
row. Searcher credit is derived directly from the main row in the same session slot, so searcher metadata does not
duplicate the main outcome. With per-token loss, the effective aggregate searcher gradient remains weighted by each
row's valid token count; `0.5` describes the sum of scalar advantage coefficients rather than a fixed aggregate
gradient ratio. A dropped group is replenished by the resident rollout pipeline before the rollout quota closes.

## Quick Start

Prepare data and start the retriever as described in [README](./README.md), then select the multi-agent mode:

```bash
export MODEL_DIR=/path/to/models
export SEARCH_R1_DATA_ROOT=/path/to/search_r1
export SEARCH_R1_RETRIEVER_URL=http://retriever-host:17389/retrieve

SEARCH_R1_MODE=multiagent bash examples/search_r1/run_search_r1.sh
```

The same entrypoint runs the M=1 comparison:

```bash
SEARCH_R1_MODE=vanilla bash examples/search_r1/run_search_r1.sh
```

## Configuration

`SEARCH_R1_MODE` selects `vanilla` or `multiagent`. Agent behavior is configured in `app/search_r1_config.yaml`:

| Key                         | Default | Effect                                                           |
| --------------------------- | ------: | ---------------------------------------------------------------- |
| `max_search_turns`          |     `4` | Main search budget                                               |
| `searcher_max_search_turns` |     `1` | Search budget for each delegated searcher                        |
| `max_observation_chars`     |  `3000` | Character cap applied to each direct retrieval block             |
| `structure_weight`          |   `0.0` | Reward assigned to a valid but incorrect action sequence         |
| `final_weight`              |   `0.0` | Reward assigned to an invalid sequence containing a final answer |

## Metrics

Main rows expose numeric `score/main`, `reward/shaped`, protocol validity, search count, and termination flags for
metric aggregation. Their string termination reason is retained in metadata and dumps. Searcher rows expose a numeric
search count and retain their string termination reason in metadata and dumps.

Evaluation uses the main records from one complete test parquet. The aggregate metric is `eval/search_r1`; source
metrics are `eval/<data_source>`.

## Risks and Boundaries

| Risk                         | Consequence                                                                                          | Current control or interpretation                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Variable searcher fanout     | Physical row count changes across identities                                                         | Dynamic batching and identity-aware transfer are required                                                    |
| Main-outcome searcher credit | A useful searcher can receive negative credit and an irrelevant searcher can receive positive credit | Searcher advantage follows the propagated main outcome rather than independent search accuracy               |
| Searcher observation growth  | A successful finding is appended after the direct retrieval block is truncated                       | The combined tool message may exceed `max_observation_chars`; Relax still enforces the session context limit |
| Excessive searching          | A previously correct trajectory can exhaust its search budget before answering                       | Track `termination/search_budget_exhausted` together with EM                                                 |
| No-contrast filtering        | Training metrics omit prompt groups with uniform outcomes                                            | Use deterministic eval metrics to assess task quality                                                        |
| Eval exports main only       | Eval does not directly score searcher rows                                                           | Use searcher training diagnostics and main task EM for separate questions                                    |

This recipe documents GRPO with the default policy-gradient reducer. It does not establish PPO, SFT, or custom
policy-gradient reducer support for M>1 exports.

## Troubleshooting

### Explicit export does not match a committed Forest state

Check that exported `messages`, `tools`, and `chat_template_kwargs` exactly match the live request history. Every record
must end at a committed assistant response.

### Multi-agent rows are split or batch accounting is incorrect

Confirm that `--use-dynamic-batch-size` remains enabled and that all records from one session retain the same
`Sample.index` in rollout dumps.

### Training repeatedly replenishes groups

Inspect main rewards within each prompt group. With no-contrast dropping enabled, all-correct and all-wrong groups are
rejected and replaced.

### Context-length failures increase in multi-agent mode

Compare direct retrieval size, adopted searcher finding length, search turns, and final training token length. The
character cap applies before a successful searcher finding is appended.

## Next Steps

- [Search-R1 README](./README.md) — data, retriever, training, and attribution
