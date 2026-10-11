# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build explicit fleet receipts for tests that are not testing transport."""

from relax.agentic.session.lora_version import VersionState


def fleet_receipts(entry, state="PREPARED"):
    return {
        engine: {
            "lora_name": entry.lora_name,
            "version_id": entry.version_id,
            "digest": entry.digest,
            "attempt_id": entry.current_attempt_id,
            "engine_incarnation": f"{engine}-boot1",
            "state": state,
        }
        for engine in ("engine0", "engine1")
    }


def commit_ready(registry, version_id, attempt_id):
    entry = registry.versions[version_id]
    if entry.state is VersionState.LOADING:
        if not entry.driven:
            registry.claim_publication(version_id, attempt_id)
        registry.record_prepared(version_id, attempt_id, fleet_receipts(entry))
        registry.record_ready(version_id, attempt_id, fleet_receipts(entry, "READY_LOCAL"))
    return registry.mark_published(version_id, attempt_id)
