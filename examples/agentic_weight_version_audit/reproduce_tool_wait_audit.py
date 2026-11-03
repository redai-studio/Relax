#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reproducible policy-provenance audit for Agentic async weight lag (No.15).

Scenario
--------
An agent emits turn-0 tokens under published weight version 2, then blocks on a
tool call. While waiting, the trainer publishes versions 3 and 4. Turn-1 then
decodes under version 4 (and mid-turn flips to 5). Training finally consumes the
trajectory when the published version is 7.

Run::

    python examples/agentic_weight_version_audit/reproduce_tool_wait_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from relax.agentic.session.state import SessionForest, check_messages
from relax.utils.weight_version import (
    WeightVersionSpan,
    build_weight_version_diagnostic,
    compute_policy_provenance_metrics,
)


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(t) if isinstance(t, int) and 32 <= t < 127 else "?" for t in token_ids)


def build_tool_wait_sample():
    forest = SessionForest.create_empty(session_id="sess-tool-wait", group_index=0, index=42)
    user = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": "search then answer"}]),
        train_token_delta=[10, 11],
        rollout_token_delta=[10, 11],
    )
    turn0 = forest.append_resp(
        parent_state_hash=user.state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "tool?", "tool_calls": []}],
        token_delta=[97, 98, 99],
        logprob_delta=[-0.1, -0.2, -0.3],
        weight_version_delta=[WeightVersionSpan("2", 0, 3)],
        status="completed",
        export_metadata_patch={"request_id": "req-turn0"},
    )
    tool = forest.append_obs(
        parent_state_hash=turn0.state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=[{"role": "tool", "tool_call_id": "call-1", "content": "result"}],
        train_token_delta=[20, 21],
        rollout_token_delta=[20, 21],
    )
    # Trainer published v3 and v4 during the tool wait; decode starts at v4 and
    # crosses to v5 mid-generation (partial / long decode).
    leaf = forest.append_resp(
        parent_state_hash=tool.state_hash,
        rollout_id=4,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "final"}],
        token_delta=[100, 101, 102, 103],
        logprob_delta=[-0.1, -0.1, -0.1, -0.1],
        weight_version_delta=[
            WeightVersionSpan("4", 0, 2),
            WeightVersionSpan("5", 2, 4),
        ],
        status="completed",
        export_metadata_patch={"request_id": "req-turn1"},
    )
    return forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_FakeTokenizer())


def main() -> None:
    sample = build_tool_wait_sample()
    current_version = 7
    metrics = compute_policy_provenance_metrics([sample], current_version=current_version)
    report = build_weight_version_diagnostic(sample, current_version=current_version)

    print("=== selected-batch policy provenance ===")
    pprint(metrics)
    print("\n=== per-sample Session / turn / segment diagnostic ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        "\nInterpretation:\n"
        f"- generation_version_span={report['generation_version_span']} "
        "(newest stamped version - oldest during generation)\n"
        f"- post_generation_staleness={report['post_generation_staleness']} "
        "(consume version - newest stamped; wait after generation finished)\n"
        f"- staleness={report['staleness']} "
        "(consume version - oldest stamped; full consume lag)\n"
        "- cancelled/replaced IR prefixes never appear because only append_resp commits spans"
    )


if __name__ == "__main__":
    main()
