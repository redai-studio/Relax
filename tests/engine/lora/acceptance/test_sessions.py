# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Agentic old/new sessions and abort/resume, compared to independent
weights."""

import asyncio
from uuid import uuid4

from . import support


CHECKS = ("old_session_logprobs", "new_session_logprobs", "abort_resume", "missing_adapter_rejected")


async def run(ctx):
    bound = await support.native_bind(ctx)
    identity = support.attempt_payload(ctx, bound, uuid4().hex)
    rid = identity.pop("rid")
    identity["native_lora_id"] = "missing-" + uuid4().hex
    response = await ctx.client.post(
        bound["engine"]["endpoint"] + "/generate",
        json={
            "rid": rid,
            "input_ids": ctx.tokenizer.encode(ctx.prompts[0]),
            "lora_path": identity["native_lora_id"],
            "lora_binding": identity,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        },
    )
    assert response.status_code in (400, 409, 503), response.text
    assert "ADAPTER_NOT_READY" in response.text, response.text
    ctx.report["checks"]["missing_adapter_rejected"] = {"status": "PASS", "response": response.json()}
    support.configure_test(ctx, capture_attempt=True)
    await ctx.train_runtime.resume_generation(rollout_id=0)
    resumed, directory = await support.start_case(ctx, "resume", 0, domain=ctx.train_runtime, wait_first=False)
    interrupted = await support.wait_file(ctx, ctx.root / "train.attempt.json")
    await ctx.train_runtime.pause_generation()
    support.configure_test(ctx)
    assert not (directory / "turn-0.json").exists(), (
        "abort profile ended too early; do not count it as resume coverage"
    )
    old = await asyncio.gather(*(support.start_case(ctx, "old", i) for i in range(len(ctx.prompts))))
    await support.publish(ctx, ctx.versions["B"])
    new = await asyncio.gather(*(support.start_case(ctx, "new", i) for i in range(len(ctx.prompts))))
    for label, cases, version in (("old", old, "A"), ("new", new, "B")):
        for _, case_dir in cases:
            (case_dir / "continue").touch()
        exports = await asyncio.gather(*(ctx.runtime.collect_group(stream) for stream, _ in cases))
        for exported in exports:
            await support.score_export(ctx, exported, ctx.versions[version])
        ctx.report["checks"][label + "_session_logprobs"] = {"status": "PASS", "sessions": len(cases)}
    (directory / "continue").touch()
    await ctx.train_runtime.resume_generation(rollout_id=1)
    exported = await ctx.train_runtime.collect_group(resumed)
    await support.score_export(ctx, exported, ctx.versions["A"])
    attempts = [attempt for sample in exported.samples for attempt in sample.metadata.get("lora_attempts", [])]
    assert any(
        item["rid"].rsplit(":", 1)[0] == interrupted["rid"].rsplit(":", 1)[0]
        and int(item["rid"].rsplit(":", 1)[1]) > 0
        for item in attempts
    ), "no actual resumed backend attempt"
    ctx.report["checks"]["abort_resume"] = {"status": "PASS", "interrupted": interrupted, "attempts": attempts}


def test_sessions(auto_acceptance):
    auto_acceptance("sessions")
