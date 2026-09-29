# Agentic exports and training scores

## Export, score, and batching

| Contexts exported per Session | Export | Training score | Dynamic batching |
| --- | --- | --- | --- |
| One, linear leaf on the reward path | Implicit or explicit | Agent output `reward` or `--custom-rm-path` | Not required by context count |
| One context with custom advantage | Explicit object or one JSONL record | `--agentic-custom-advantage-path` | Not required by context count |
| One selected from several leaves | Explicit object or one JSONL record | Agent output `reward`, `--custom-rm-path`, or `--agentic-custom-advantage-path` | Not required by context count |
| More than one | Explicit JSONL records | `--agentic-custom-advantage-path` | `--use-dynamic-batch-size` and `--max-tokens-per-gpu` |

Multiple resident Sessions do not trigger the multi-context rule. One Session exporting several training contexts does.
Custom advantage uses named explicit export records, including when one Session exports one context.

## Logical identity and physical rows

All contexts exported by one Session retain the same logical `Sample.index`; each export becomes a physical training row. Transfer debt remains Group-based. Under sample-mean loss, training uses the shared identity to aggregate the Session's effective-token denominator.

Do not treat context fanout as extra GRPO siblings or increase logical batch denominators by the number of exports.
Per-token loss gives longer or multi-context Sessions more weight; keep this as an explicit algorithm/backend choice.

## Custom advantage

`advantage_func` receives one mapping per sampled Session: `{export_name: export_metadata}`.

It returns `None` or a list with the same Session order and export names. In each result mapping, an export's advantage is a
scalar shared by all assistant turns or a list containing one value per assistant turn.

| Function output | Meaning |
| --- | --- |
| `{name: scalar}` | Use the same score for every assistant turn |
| `{name: list[float]}` | Use item `k` for assistant turn `k`; list length must equal `agentic_trace.turn_count` |
| `None` | Drop and replenish the complete Group |

Relax expands each list item over that turn's tokens; observation tokens keep zero. Put every input used by the function
in export metadata and normalize it inside the function. Exported `reward` is not included in the function input. Eval
does not call the function.

Avoid `--custom-rm-path` for multi-context training. Without `--group-rm`, custom advantage skips the ordinary RM path.
With `--group-rm`, Group RM may still write `sample.reward` for metrics, filtering, or dumps while custom advantage
supplies the training score. Route advantage-estimator compatibility to the algorithm expert.

## Outcome metrics

Built-in passrate/pass@k consumes physical rows and treats the selected primary reward value as success only when it
equals `1`. For multi-context passrate, use explicit export and attach reward to exactly one representative export per
logical Session, normally the main-agent context. Set its primary reward to `1` for success or `0` otherwise, and leave
reward unset on sibling exports. For a reward object, `--reward-key` selects that primary value. A Group RM that writes
reward to every exported row is incompatible with this built-in aggregation unless a custom logger restores
logical-Session grouping.

## Eval policy

Eval does not run Agentic custom advantage. Default to one representative exported context per Eval Session. Built-in
Eval aggregation is physical-row based; multi-context Eval requires an explicitly reviewed custom aggregation.

## Metadata boundary

Export/output `metadata` feeds custom advantage, metrics, and dumps. Dataset `train_metadata` is the metadata transferred into the training batch. Do not use one as a substitute for the other.

Route Group RM internals, OPD, TIS/OPSM, generic reward normalization, loss reducers, and TransferQueue sampler algorithms to their dedicated review. Check only whether their assumptions remain valid under Agentic context fanout.

Source anchors:

- `relax/agentic/pipeline/reward.py`
- `relax/agentic/session/state.py::SessionForest.build_sample`
- `relax/agentic/pipeline/transfer.py`
- `relax/utils/utils.py::convert_samples_to_train_data`
- `relax/utils/utils.py::post_process_rewards`
