# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Golden tests for cross-protocol canonicalization into the Agentic Session.

Three ingress protocols (Chat Completions, Responses, Anthropic Messages) must
collapse semantically equivalent requests onto the same ``messages`` /
``tools`` / ``chat_template_kwargs`` triple, because that triple is what the
Session state hash is computed from.
"""

from __future__ import annotations

import pytest

from relax.agentic.session.state import _messages_tools_template_state_hash
from tests.agentic.helpers import (
    CANONICAL_FIELDS,
    PROTOCOLS,
    canonical_case_ids,
    canonical_expectations,
    canonical_of,
    normalize,
    protocol_payloads,
)


# Cases the task brief explicitly requires to stay covered; dropping one must
# fail here rather than silently shrink the suite.
REQUIRED_CASE_IDS = frozenset(
    {
        "text_only",
        "non_ascii_text",
        "system_merge_two_chunks",
        "assistant_tool_call_single",
        "assistant_parallel_tool_calls",
        "tool_result_empty_output",
        "assistant_reasoning_only",
        "reasoning_with_text",
        "image_http_url",
        "image_base64_data_url",
        "image_with_multiple_text_blocks",
        "tools_with_description",
        "tools_without_description",
        "enable_thinking_true",
        "enable_thinking_false",
        "full_multiturn_conversation",
    }
)

CASE_IDS = canonical_case_ids()


def test_canonical_fixture_case_sets_align() -> None:
    per_protocol = {protocol: set(protocol_payloads(protocol)) for protocol in PROTOCOLS}

    assert per_protocol == {protocol: set(CASE_IDS) for protocol in PROTOCOLS}
    assert REQUIRED_CASE_IDS <= set(CASE_IDS), sorted(REQUIRED_CASE_IDS - set(CASE_IDS))


def test_canonical_field_names_are_stable() -> None:
    for protocol in PROTOCOLS:
        for case_id in CASE_IDS:
            result = normalize(protocol, protocol_payloads(protocol)[case_id])
            assert set(CANONICAL_FIELDS) <= set(result), f"{protocol}/{case_id} lost a canonical field"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_canonical_fields_agree_across_protocols(case_id: str) -> None:
    canonical = {
        protocol: canonical_of(normalize(protocol, protocol_payloads(protocol)[case_id])) for protocol in PROTOCOLS
    }
    reference = canonical["chat_completions"]

    assert canonical["responses"] == reference, f"Responses diverged on case {case_id}"
    assert canonical["anthropic_messages"] == reference, f"Anthropic Messages diverged on case {case_id}"


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_canonical_fields_match_golden(protocol: str, case_id: str) -> None:
    result = normalize(protocol, protocol_payloads(protocol)[case_id])

    assert canonical_of(result) == canonical_expectations()[case_id]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_canonical_state_hash_agrees_across_protocols(case_id: str) -> None:
    hashes = set()
    for protocol in PROTOCOLS:
        canonical = canonical_of(normalize(protocol, protocol_payloads(protocol)[case_id]))
        hashes.add(
            _messages_tools_template_state_hash(
                canonical["messages"],
                canonical["tools"],
                canonical["chat_template_kwargs"],
            )
        )

    assert len(hashes) == 1, f"case {case_id} reached the Session with {len(hashes)} distinct state hashes"
