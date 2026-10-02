# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Ordinary/managed steady state and actual publication under continuous
traffic."""

from . import performance, support


CHECKS = ("steady_state_overhead", "continuous_traffic")


async def run(ctx):
    if ctx.config.get("overhead_profile"):
        from .overhead import run

        await run(ctx)
        return
    a = await support.native_bindings(ctx)
    ordinary = await performance.ordinary_baseline(ctx, a)
    assert ordinary is not None, "ordinary LoRA baseline must not be omitted"
    steady_a = await performance.traffic_window(ctx, a * 2)
    performance.compare_windows(steady_a, ordinary, ctx.config)
    during = await performance.traffic_window(ctx, a * 2, ctx.versions["B"])
    performance.compare_windows(during, steady_a, ctx.config)
    b = await support.native_bindings(ctx)
    steady_ab = await performance.traffic_window(ctx, a + b)
    ctx.report["checks"]["steady_state_overhead"] = {"status": "PASS", "ordinary": ordinary, "managed": steady_a}
    ctx.report["checks"]["continuous_traffic"] = {
        "status": "PASS",
        "steady_a": steady_a,
        "publication_a": during,
        "steady_ab": steady_ab,
        "comparison": "publication uses retained A sessions; compared with the same A-only offered load",
    }


def test_performance(auto_acceptance):
    auto_acceptance("performance")
