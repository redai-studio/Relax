# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sidecar provenance for committed Agentic tokens and training consumers."""

from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4


AUDIT_FIELD = "weight_version_audit"


def attempt_weight_version(meta: dict[str, Any]) -> str | None:
    """Legacy scalar is valid under Relax's abort-before-update contract.

    Do not assign the last version to a mixed attempt from another pause mode.
    """
    version = meta.get("weight_version")
    spans = meta.get("weight_versions")
    if spans is not None and (
        not isinstance(spans, list)
        or not spans
        or any(not isinstance(span, dict) or span.get("version") != version for span in spans)
    ):
        return None
    return str(version) if version is not None else None


class PublicationLedger:
    """Owned by one async Coordinator; methods never await or share globals."""

    def __init__(self) -> None:
        self.epoch: str | None = None
        self.sequence: int | None = None
        self.versions: dict[str, int] = {}
        self.valid = True

    def record(self, epoch: str, sequence: int, version: str) -> None:
        if self.epoch is not None and epoch != self.epoch:
            # One policy publisher per deployment. Do not silently compare
            # counters from another writer/restarted training process.
            self.valid = False
            raise ValueError("Agentic publication publisher changed within a deployment")
        if version in self.versions:
            if self.versions[version] != sequence:
                self.valid = False
                raise ValueError("A published weight version cannot be reused")
            return
        if self.sequence is not None and sequence <= self.sequence:
            self.valid = False
            raise ValueError("Out-of-order Agentic publication")
        if sequence != (self.sequence or 0) + 1:
            # An unobserved publication could have preceded a request's end
            # snapshot. Never reconstruct a false waiting age after a gap.
            self.valid = False
        self.epoch = epoch
        self.sequence = sequence
        self.versions[version] = sequence

    def snapshot(self, version: str | None = None) -> dict[str, Any]:
        if not self.valid:
            return {"epoch": None, "sequence": None, "source_sequence": None}
        return {
            "epoch": self.epoch,
            "sequence": self.sequence,
            "source_sequence": self.versions.get(version) if version is not None else None,
        }


def record_publication(owner: Any) -> None:
    """Best-effort audit after successful installation, before generation
    resumes.

    Call on the publication rank only. The writer's successful-publication
    counter preserves gaps even if a diagnostic RPC is lost. Adapter-only mode
    is excluded because its base-model response tag is not its policy.
    """
    if not getattr(getattr(owner, "args", None), "use_agentic_rollout", False):
        return
    if getattr(owner, "_lora_adapter_mode", False) or getattr(owner, "lora_adapter_mode", False):
        return
    from ray import serve

    from relax.agentic import AGENTIC_CHAT_API_SERVICE_NAME
    from relax.utils.logging_utils import get_logger

    version = str(owner.weight_version)
    previous = getattr(owner, "_agentic_publication", None)
    if previous is not None and previous[2] == version:
        epoch, sequence, _ = previous
    else:
        epoch = previous[0] if previous is not None else uuid4().hex
        sequence = previous[1] + 1 if previous is not None else 1
        owner._agentic_publication = (epoch, sequence, version)
    if not getattr(owner, "_agentic_audit_valid", True):
        return
    try:
        # DCS invokes its synchronous backend inside an event loop. This
        # bounded wait must finish before generation resumes; Serve's router
        # runs on a separate loop and the remote ledger never calls back here.
        serve.get_app_handle(AGENTIC_CHAT_API_SERVICE_NAME).record_publication.remote(epoch, sequence, version).result(
            timeout_s=5, _skip_asyncio_check=True
        )
    except Exception:
        owner._agentic_audit_valid = False
        get_logger(__name__).warning("Agentic publication audit unavailable for version %s", version, exc_info=True)


def consumption_snapshot(publisher: Any) -> dict[str, Any]:
    """Read the writer's successful counter, never a possibly stale replica."""
    publication = getattr(publisher, "_agentic_publication", None)
    if publication is None or not getattr(publisher, "_agentic_audit_valid", True):
        return {"audit_valid": False}
    epoch, sequence, _ = publication
    return {"epoch": epoch, "sequence": sequence, "audit_valid": True}


def prepare_audit(sample: Any) -> dict[str, Any] | None:
    """Freeze final CPU mask counts before the training tensors move to GPU."""
    trace = (sample.metadata or {}).get("agentic_trace", {})
    spans = trace.get("weight_version_spans")
    if spans is None:
        # Legacy/custom samples still contribute to the missing-token rate.
        spans = [{"start": 0, "end": sample.response_length, "weight_version": None, "publication": {}}]
    spans = copy.deepcopy(spans)
    for span in spans:
        span["effective_tokens"] = sum(sample.loss_mask[span["start"] : span["end"]])
    return {
        "session_id": sample.session_id,
        "sample_index": sample.index,
        "leaf_state_hash": trace.get("leaf_state_hash"),
        "spans": spans,
        "generation_end": copy.deepcopy(trace.get("generation_end_publication", {})),
    }


def diagnose(audit: dict[str, Any], consumption: dict[str, Any]) -> dict[str, Any]:
    """Use only comparable published sequences; unknown values stay null."""
    report = copy.deepcopy(audit)
    current = consumption.get("sequence")
    epoch = consumption.get("epoch")
    known_tokens = total_tokens = weighted_age = 0
    versions = []
    generation_epochs = set()
    ages = []
    end = report.get("generation_end", {})
    end_seq = end.get("sequence")
    for span in report["spans"]:
        count = span["effective_tokens"]
        total_tokens += count
        publication = span.get("publication", {})
        if consumption.get("audit_valid") is False:
            publication = {}
            span["publication"] = {}
        sequence = publication.get("source_sequence")
        if count and sequence is not None and publication.get("epoch") is not None:
            versions.append(sequence)
            generation_epochs.add(publication["epoch"])
        comparable = epoch is not None and publication.get("epoch") == epoch
        age = current - sequence if comparable and current is not None and sequence is not None else None
        span["consumption_age"] = age if age is not None and age >= 0 else None
        generation_age = (
            end_seq - sequence
            if end.get("epoch") is not None
            and end.get("epoch") == publication.get("epoch")
            and end_seq is not None
            and sequence is not None
            else None
        )
        span["generation_age"] = generation_age if generation_age is not None and generation_age >= 0 else None
        if count and span["consumption_age"] is not None:
            known_tokens += count
            weighted_age += count * age
            ages.append(age)
    tail = (
        end_seq - max(versions)
        if versions and generation_epochs == {end.get("epoch")} and end_seq is not None
        else None
    )
    wait = (
        current - end_seq
        if epoch is not None and end.get("epoch") == epoch and current is not None and end_seq is not None
        else None
    )
    report.update(
        consumption=copy.deepcopy(consumption),
        effective_tokens=total_tokens,
        known_age_tokens=known_tokens,
        weighted_age_sum=weighted_age,
        version_missing_tokens=sum(
            s["effective_tokens"] for s in report["spans"] if s.get("publication", {}).get("source_sequence") is None
        ),
        generation_version_span=max(versions) - min(versions) if versions and len(generation_epochs) == 1 else None,
        generation_tail_lag=tail if tail is not None and tail >= 0 else None,
        consumption_age_max=max(ages) if ages else None,
        consumption_age_token_mean=weighted_age / known_tokens if known_tokens else None,
        age_missing_token_rate=(total_tokens - known_tokens) / total_tokens if total_tokens else None,
        post_generation_wait_lag=wait if wait is not None and wait >= 0 else None,
    )
    report["version_missing_token_rate"] = report["version_missing_tokens"] / total_tokens if total_tokens else None
    return report


def audit_consumed_rows(rollout_data: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Enrich already-consumed rows for the existing training dump/log path."""
    from relax.utils.logging_utils import get_logger

    rows = rollout_data.get(AUDIT_FIELD)
    if rows is None:
        return
    reports = [diagnose(row, snapshot) if row is not None else None for row in rows]
    rollout_data[AUDIT_FIELD] = reports
    # Counts/sums permit correct cross-DP offline aggregation, unlike means.
    valid = [report for report in reports if report is not None]
    get_logger(__name__).info(
        "Agentic weight audit: audited_effective_tokens=%s known_age_tokens=%s weighted_age_sum=%s "
        "missing_version_tokens=%s unaudited_rows=%s",
        sum(row["effective_tokens"] for row in valid),
        sum(row["known_age_tokens"] for row in valid),
        sum(row["weighted_age_sum"] for row in valid),
        sum(row["version_missing_tokens"] for row in valid),
        sum(row is None for row in reports),
    )
