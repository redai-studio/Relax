# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual optimizer step -> existing checkpoint exporter -> seal -> publish ->
real generation."""

import json
from pathlib import Path

from relax.engine.lora.snapshot import read_snapshot

from . import support


CHECKS = ("real_export",)


async def run(ctx):
    exported = json.loads(Path(ctx.config["export_evidence"]).read_text())
    assert exported["state"] == "SEALED" and exported["exported_step"] == 1
    snapshot = read_snapshot(exported["snapshot"]["path"])
    assert snapshot.digest == exported["snapshot"]["digest"]
    published = await support.publish(ctx, snapshot.version_id)
    assert published["digest"] == snapshot.digest
    bound = await support.native_bind(ctx)
    result = await support.diagnostic(ctx, bound, ctx.tokenizer.encode(ctx.prompts[0], add_special_tokens=True))
    identity = result["meta_info"]["lora_adapter"]
    assert identity["source_train_step"] == 1 and identity["adapter_digest"] == snapshot.digest
    ctx.report["checks"]["real_export"] = {
        "status": "PASS",
        "export": exported,
        "publication": published,
        "generation": result,
        "scope": "single-device optimizer and existing DCP checkpoint export; not Megatron online training acceptance",
    }


def test_export(auto_acceptance):
    auto_acceptance("export")
