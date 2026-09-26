"""Speculative decoding metric aggregation for rollout samples."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from relax.utils.types import Sample


def aggregate_speculative_metrics(samples: Iterable[Sample]) -> dict[str, float | int | bool]:
    """Aggregate speculative counters with Agentic generation-node de-duplication.

    New Agentic exports carry one record per committed request in sample
    metadata.  Records are keyed by session and request id, so shared lineage
    nodes are counted once while independent requests remain distinct.  Older
    samples do not carry identities; they use a clearly marked additive
    fallback for backwards compatibility.
    """
    samples = list(samples)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    saw_generation_records = False
    for sample in samples:
        raw_records = sample.metadata.get("spec_generation_nodes") if isinstance(sample.metadata, dict) else None
        if not isinstance(raw_records, list):
            continue
        saw_generation_records = True
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            session_id = record.get("session_id")
            request_id = record.get("request_id")
            if not isinstance(session_id, str) or not session_id or not isinstance(request_id, str) or not request_id:
                continue
            records.setdefault((session_id, request_id), record)

    if saw_generation_records and records:
        accepted = sum(int(record.get("accepted", 0) or 0) for record in records.values())
        proposed = sum(int(record.get("proposed", 0) or 0) for record in records.values())
        verify = sum(int(record.get("verify", 0) or 0) for record in records.values())
        completion = sum(int(record.get("completion", 0) or 0) for record in records.values())
        acceptance_coverage = sum(bool(record.get("acceptance_present")) for record in records.values())
        length_coverage = sum(bool(record.get("length_present")) for record in records.values())
        metrics: dict[str, float | int | bool] = {
            "spec_acceptance_coverage": acceptance_coverage / len(records) if records else 0.0,
            "spec_length_coverage": length_coverage / len(records) if records else 0.0,
            "spec_generation_count": len(records),
            "spec_metrics_legacy": False,
        }
        if acceptance_coverage:
            metrics["spec_accept_rate"] = accepted / proposed if proposed > 0 else 0.0
        if length_coverage:
            metrics["spec_accept_length"] = completion / verify if verify > 0 else 0.0
        return metrics

    if not samples:
        return {}

    accepted = sum(sample.spec_info.spec_accept_token_num for sample in samples)
    proposed = sum(sample.spec_info.spec_draft_token_num for sample in samples)
    verify = sum(sample.spec_info.spec_verify_ct for sample in samples)
    completion = sum(sample.spec_info.completion_token_num for sample in samples)
    metrics = {
        "spec_acceptance_coverage": 0.0,
        "spec_length_coverage": 0.0,
        "spec_generation_count": 0,
        "spec_metrics_legacy": True,
    }
    if proposed > 0:
        metrics["spec_accept_rate"] = accepted / proposed
    if verify > 0:
        metrics["spec_accept_length"] = completion / verify
    return metrics
