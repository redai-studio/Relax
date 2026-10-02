# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-engine commit, partial failure, cancellation, late ACK and
idempotence."""

from uuid import uuid4

from . import scenarios, support


CHECKS = (
    "partial_readiness",
    "failure_late_ack",
    "publication_idempotence",
    "content_conflict",
    "session_logprob_comparison",
)


async def run(ctx):
    ctx.native_a = await support.native_bindings(ctx)
    old = [await support.start_case(ctx, "old", 0)]
    retry = await scenarios.failed_publication(ctx)
    retry = await scenarios.partial_publication(ctx, old, retry)
    request_id = uuid4().hex
    payload = {"version_id": ctx.versions["B"], "request_id": request_id, "retry_of": retry}
    accepted = await support.request(ctx, "POST", "/lora/publications", payload)
    duplicate = await support.request(ctx, "POST", "/lora/publications", payload)
    assert duplicate["operation_id"] == accepted["operation_id"]
    published = await support.wait_state(ctx, accepted["operation_id"], "PUBLISHED")
    replay = await support.request(ctx, "POST", "/lora/publications", payload)
    assert replay["operation_id"] == accepted["operation_id"]
    current = await support.request(ctx, "GET", "/lora/versions")
    assert current["default"]["version_id"] == ctx.versions["B"]
    ctx.report["checks"]["publication_idempotence"] = {"status": "PASS", "published": published, "replay": replay}
    newer = await support.start_case(ctx, "new", 0)
    ctx.report["checks"]["session_logprob_comparison"] = {"status": "RUNNING"}
    try:
        for cases, version in ((old, ctx.versions["A"]), ([newer], ctx.versions["B"])):
            for stream, directory in cases:
                (directory / "continue").touch()
                await support.score_export(ctx, await ctx.runtime.collect_group(stream), version)
    except Exception as error:
        ctx.report["checks"]["session_logprob_comparison"] = {"status": "FAIL", "error": str(error)}
        raise
    ctx.report["checks"]["session_logprob_comparison"] = {"status": "PASS"}
    before = await support.request(ctx, "GET", "/lora/versions")
    response = await ctx.client.post(
        ctx.base + "/lora/publications",
        json={"version_id": ctx.versions["B"], "digest": "f" * 64, "request_id": uuid4().hex},
    )
    assert response.status_code in (400, 409), response.text
    after = await support.request(ctx, "GET", "/lora/versions")
    assert before["default"] == after["default"] and before["occupied"] == after["occupied"]
    ctx.report["checks"]["content_conflict"] = {"status": "PASS", "rejection": response.json()}


def test_publication(auto_acceptance):
    auto_acceptance("publication")
