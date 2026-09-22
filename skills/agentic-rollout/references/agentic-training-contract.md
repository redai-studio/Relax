# Agentic training contract

Keep this reference limited to semantics introduced by Agentic export fanout.

## Export, credit, and batching

| Contexts exported per Session | Export | Training credit | Dynamic batching |
| --- | --- | --- | --- |
| One, linear leaf | Implicit or explicit | Write `reward`, or configure a reward producer | Not required by context count |
| One selected from several leaves | Explicit object or one JSONL record | Write `reward`, or configure a reward producer | Not required by context count |
| More than one | Explicit JSONL records | Require `--agentic-custom-advantage-path`; ordinarily avoid `--custom-rm-path` | Require `--use-dynamic-batch-size` and `--max-tokens-per-gpu` |

Multiple resident Sessions do not trigger the multi-context rule. One Session exporting several training contexts does.

## Logical identity and physical rows

All contexts exported by one Session retain the same logical `Sample.index`; each export becomes a physical training row. Transfer debt remains Group-based. Under sample-mean loss, training uses the shared identity to aggregate the Session's effective-token denominator.

Do not treat context fanout as extra GRPO siblings or increase logical batch denominators by the number of exports.
Per-token loss gives longer or multi-context Sessions more weight; keep this as an explicit algorithm/backend choice.

## Custom advantage

The hook receives original Session order, with each Session represented as `{export_name: merged_export_metadata}`. Every non-`None` result must return a numeric value for every exported name.

A top-level `None` intentionally drops the whole prompt Group. No context from that Group reaches Transfer; outstanding demand remains and is replenished. It is not a missing value for one context.

Ordinarily avoid `--custom-rm-path` for multi-context credit ownership. Without `--group-rm`, custom advantage makes
the ordinary RM path skip it. With `--group-rm`, a deliberately designed Group RM may still write `sample.reward` for
metrics, filtering, or dumps while custom advantage supplies training credit. Route advantage-estimator compatibility
to the algorithm expert.

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
