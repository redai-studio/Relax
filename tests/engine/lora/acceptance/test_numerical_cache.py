# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fixed-token references, both cache directions, wrong-KV controls and real
graph replay."""

from . import scenarios, support
from .assertions import compare


CHECKS = (
    "baseline_repeatability",
    "fixture_separation",
    "kv_sensitivity",
    "cache_a_to_b",
    "cache_b_to_a",
    "wrong_kv_negative_control",
    "mixed_batch",
    "cuda_graph",
)


async def run(ctx):
    separated = 0
    baselines = []
    for prompt in ctx.prompts:
        ids = ctx.tokenizer.encode(prompt, add_special_tokens=True)
        pair = []
        for name in ("A", "B"):
            first = await support.cold_diagnostic(ctx, ctx.versions[name], ids)
            second = await support.cold_diagnostic(ctx, ctx.versions[name], ids)
            compare(second, first, atol=ctx.atol, rtol=ctx.rtol)
            pair.append(first)
        separated += sum(
            abs(pair[0][token] - pair[1][token]) > 10 * (ctx.atol + ctx.rtol * abs(pair[1][token]))
            for token in pair[0]
        )
        baselines.append({"input_ids": ids, "A": pair[0], "B": pair[1]})
    assert separated >= 8, "fixtures lack numerical separation"
    ctx.report["checks"]["baseline_repeatability"] = {"status": "PASS", "baselines": baselines}
    ctx.report["checks"]["fixture_separation"] = {"status": "PASS", "positions": separated}
    await scenarios.calibration(ctx)
    a = await support.native_bindings(ctx)
    await support.publish(ctx, ctx.versions["B"])
    b = await support.native_bindings(ctx)
    await scenarios.cache_direction(ctx, a, b, "cache_a_to_b")
    await scenarios.cache_direction(ctx, b, a, "cache_b_to_a")
    await scenarios.wrong_kv_controls(ctx, a, b)
    await scenarios.mixed_batches(ctx, a, b, ctx.versions["A"], ctx.versions["B"], "mixed_batch")


def test_numerical_cache(auto_acceptance):
    auto_acceptance("numerical_cache")
