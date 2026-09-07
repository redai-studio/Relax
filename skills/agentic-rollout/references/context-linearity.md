# Context linearity

Audit the exact model-visible sequence for each logical agent context after protocol projection:

```text
Chat messages | Responses input Items | Messages system + messages
-> canonical messages + function tools + chat_template_kwargs
-> canonical assistant response
-> protocol response
```

Chat, Responses, and Messages requests must replay complete history. Responses `previous_response_id` does not
contribute lineage state. Preserve the interface payload required by the client and compare the projected canonical
state for SessionForest identity.

## Linear transition

After a response is committed, the next request should extend that lineage:

```text
previous request messages
+ previous assistant response
+ new user observation or tool result
```

Relax finds the longest committed message prefix with the same normalized tools and template arguments. The unmatched suffix becomes the next observation. State identity is therefore wider than `messages` alone.

## Classify every transition

| Transition | Meaning | Review action |
| --- | --- | --- |
| Append | Continues the latest committed response | Expected linear path |
| Fork | Continues an earlier committed state | Decide whether the branch is intentional |
| Concurrent sibling | Multiple requests share one parent | Check parallel-agent design versus shared-history races |
| Reset | No committed prefix matches | Check history loss or tools/template drift |

## Common accidental forks

- mutating or replacing a shared messages list;
- retrying from a stale snapshot;
- dropping an assistant response or tool result;
- deleting historical `reasoning_content` or thinking before the next request;
- compaction that replaces history without an explicit context boundary;
- changing tool order, schema defaults, descriptions, or generated metadata;
- omitting tools on one request after previously sending them;
- changing reasoning or template kwargs between otherwise continuous turns.

A tools or `chat_template_kwargs` change alters the state hash. Identical messages with different tools/template arguments form a different SessionForest subtree.

## Decide with the user

Root fallback and forks are valid SessionForest behavior. Determine whether each nonlinear transition is intentional,
then present the relevant choice instead of automatically rewriting the agent.

For deleted thinking or `reasoning_content`:

- preserve it to keep one linear model-visible history, accepting possible differences from the agent's normal inference
  behavior; or
- preserve the agent's deletion behavior, record the exact `messages` and any used `tools` or
  `chat_template_kwargs` for every resulting context, and use explicit export plus the matching multi-context training
  contract.

For changing tools:

- different agents or intentional branches may legitimately have different tool sets; or
- when one otherwise-linear agent dynamically enables tools, discuss the prefix-cache tradeoff. Prefer a stable full
  tool set plus a per-turn user message describing which tools are currently available when the user accepts that
  design.

Implicit export is reserved for an audited strictly linear history. Any retained nonlinear topology uses explicit
export. Explicit records use canonical Chat-shaped messages and tools even when requests used Responses or Messages.
Multiple exported contexts require custom advantage and dynamic batching.

## Audit output

```text
Context ID:
Transition: append | fork | concurrent sibling | reset
Matched committed parent:
Canonical messages prefix preserved: yes/no
Tools fingerprint stable: yes/no
Template kwargs stable: yes/no
Intentional: yes/no/unknown
Action: keep branch | linearize | investigate
```

Source anchors:

- `relax/agentic/session/service.py::_match_parent_state`
- `relax/agentic/session/service.py::_append_observation`
- `relax/agentic/session/state.py::_messages_tools_template_state_hash`
- `docs/en/guide/agentic-rollout.md#state-identity-and-prefix-matching`
