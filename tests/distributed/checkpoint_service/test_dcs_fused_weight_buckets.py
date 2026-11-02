# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DCS bucket failures release the update lock without hiding the original
error."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture()
def dcs_update_bucket():
    path = Path(__file__).resolve().parents[3] / "relax/distributed/checkpoint_service/backends/device_direct.py"
    tree = ast.parse(path.read_text())
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeviceDirectBackend")
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_bucket_weights_from_distributed"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("release_fails", [False, True])
def test_http_failure_releases_dcs_lock_preserves_error_and_does_not_advance_progress(
    dcs_update_bucket, release_fails
):
    calls = []
    original_error = RuntimeError("HTTP 400: partial weights")

    def get(value):
        if value == "http":
            raise original_error
        if value == "release":
            calls.append("release")
            if release_fails:
                raise RuntimeError("lock actor stopped")
        return value

    namespace = dcs_update_bucket
    namespace.update(
        ray=SimpleNamespace(get=get),
        dist=SimpleNamespace(broadcast=lambda *args, **kwargs: SimpleNamespace(wait=lambda: None)),
        logger=logging.getLogger(__name__),
    )
    state = SimpleNamespace(
        _group_name="slime-pp_7",
        _model_update_groups=object(),
        weight_version=1,
        lock=SimpleNamespace(
            acquire=SimpleNamespace(remote=lambda: True), release=SimpleNamespace(remote=lambda: "release")
        ),
        _batch_request=lambda *args: "http",
    )
    progress = SimpleNamespace(update=lambda count: calls.append("progress"))
    with pytest.raises(RuntimeError) as error:
        namespace["_update_bucket_weights_from_distributed"](state, [("weight", torch.ones(1))], progress)
    assert error.value is original_error
    assert calls == ["release"]
