# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.agentic.rollout import _collect_agentic_metadata_metrics
from relax.utils.types import Sample


def test_session_metrics_do_not_weight_longer_trajectories_more():
    samples = []
    for session_id, outcome, turns in (("short", 1, 1), ("long", 0, 3)):
        for index in range(turns):
            samples.append(
                Sample(
                    session_id=session_id,
                    metadata={
                        "agent_turns": turns,
                        "outcome": outcome,
                        "session_wallclock_s": 10 * turns,
                        "parse_failed_steps": 0,
                        "turn_index": index,
                        "export_index": index,
                        "parse_attempt": 0,
                        "accepted_for_execution": True,
                        "export_scalar": index,
                        "_session_metrics": {
                            "osworld/success_rate": outcome,
                            "osworld/environment_steps": turns,
                        },
                    },
                )
            )

    metrics = _collect_agentic_metadata_metrics(samples)

    assert metrics["osworld/success_rate/mean"] == 0.5
    assert metrics["osworld/environment_steps/mean"] == 2
    assert metrics["export_scalar/mean"] == 0.75
    assert metrics["agentic/exported_contexts"] == 4
    assert metrics["agentic/sessions"] == 2
    assert not any(key.startswith("outcome/") for key in metrics)
    assert not any(key.startswith("agent_turns/") for key in metrics)
    assert not any(key.startswith("turn_index/") for key in metrics)
    assert not any(key.startswith("parse_attempt/") for key in metrics)


def test_legacy_metadata_metrics_remain_export_weighted():
    samples = [Sample(metadata={"outcome": value, "_private": 99}) for value in (1, 0, 0)]

    metrics = _collect_agentic_metadata_metrics(samples)

    assert metrics["outcome/mean"] == 1 / 3
    assert "agentic/sessions" not in metrics
    assert not any(key.startswith("_private/") for key in metrics)


def test_session_metrics_use_metadata_id_and_ignore_unidentifiable_sessions():
    samples = [
        Sample(metadata={"session_id": "a", "_session_metrics": {"osworld/success_rate": 1}}),
        Sample(metadata={"session_id": "a", "_session_metrics": {"osworld/success_rate": 1}}),
        Sample(metadata={"_session_metrics": {"osworld/success_rate": 0}}),
    ]

    metrics = _collect_agentic_metadata_metrics(samples)

    assert metrics["osworld/success_rate/mean"] == 1
    assert metrics["agentic/sessions"] == 1
