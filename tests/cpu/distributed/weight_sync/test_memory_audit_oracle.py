# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""An oracle mismatch must preserve evidence and fail the acceptance run."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from relax.backends.sglang.disk_weight_dump import dump_model
from task_delta_weight_sync.memory_audit_oracle import check_version


@pytest.mark.parametrize("failure", [None, "bit", "coverage"])
def test_oracle_checks_all_delivered_models_and_rejects_one_bit(tmp_path, failure):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([1.0, -0.0], dtype=torch.bfloat16), requires_grad=False)
    export = tmp_path / "hf"
    export.mkdir()
    shard = export / "model.safetensors"
    save_file(model.state_dict(), shard)
    delivered = []
    for index in range(8):
        directory = tmp_path / "delivered" / str(index)
        dump_model(model, directory, 0, packed=True)
        delivered.append(dict(engine_rank=index, actor_id=str(index), directory=str(directory / "rank-0")))
    if failure == "bit":
        changed = model.weight.detach().clone().view(torch.int16)
        changed[0] ^= 1
        save_file({"weight": changed.view(torch.bfloat16)}, Path(delivered[-1]["directory"]) / "model.safetensors")
    if failure == "coverage":
        path = Path(delivered[-1]["directory"]) / "coverage.json"
        coverage = json.loads(path.read_text())
        coverage["parameter_aliases"] = [["missing_alias", "weight"]]
        path.write_text(json.dumps(coverage))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="CAPTURED_NOT_COMPARED",
                version=1,
                update="sha",
                stream="stream",
                schema="schema",
                independent_hf=str(export),
                hf_files={
                    shard.name: dict(bytes=shard.stat().st_size, sha256=hashlib.sha256(shard.read_bytes()).hexdigest())
                },
                captured=delivered,
            )
        )
    )
    events = []

    def request(endpoint, payload, **kwargs):
        assert endpoint == "update_weights_from_disk" and payload["model_path"] == str(export)
        events.append("ordinary-full-load")
        return dict(success=True)

    def snapshot(engine, destination, version):
        assert events == ["ordinary-full-load"]
        assert version == "sha"
        dump_model(model, Path(destination) / "engine-0", 0, packed=True)
        return dict(directory=str(Path(destination) / "engine-0/rank-0"))

    engine = SimpleNamespace(
        pause_generation=lambda **kw: None,
        _make_request=request,
        continue_generation=lambda **kw: events.append("resume"),
        disk_weight_generation_probe=lambda timeout: events.append("probe"),
    )
    output = tmp_path / "oracle"
    with patch("task_delta_weight_sync.memory_audit.snapshot_engine", side_effect=snapshot):
        if failure:
            with pytest.raises(RuntimeError, match="GPU raw-bit comparison failed"):
                check_version(SimpleNamespace(engine=engine), str(receipt), str(output))
        else:
            assert check_version(SimpleNamespace(engine=engine), str(receipt), str(output))["equal"]
    result = json.loads((output / "version-000001/result.json").read_text())
    assert len(result["pairs"]) == 8
    assert result["equal"] is (failure is None)
    assert result["pairs"][-1]["diff"]["mismatch_bits"] == int(failure == "bit")
    assert events == (["ordinary-full-load"] if failure else ["ordinary-full-load", "resume", "probe"])


@pytest.mark.parametrize("write_fails", [False, True])
def test_oracle_cleans_every_engine_even_if_kill_and_result_write_fail(tmp_path, write_fails):
    import ray

    from relax.distributed.weight_sync import store
    from task_delta_weight_sync import gpu_replay, memory_audit_oracle

    root = tmp_path / "audit"
    (root / "receipts").mkdir(parents=True)
    for version in range(11):
        (root / "receipts" / f"{version:06d}.json").write_text(
            json.dumps(dict(version=version, stream="same", schema="same", independent_hf="unused"))
        )
    closed = []
    original_write = store.atomic_write

    def write(path, data):
        if write_fails:
            raise OSError("injected final result write failure")
        return original_write(path, data)

    class Remote:
        def __init__(self, function):
            self.function = function

        def remote(self, *args, **kwargs):
            return lambda: self.function(*args, **kwargs)

    def close(rank):
        closed.append(rank)
        if rank == 0:
            raise OSError("injected shutdown failure")

    def create(rank, *args):
        return SimpleNamespace(
            pause=Remote(lambda: None),
            shutdown=Remote(lambda: close(rank)),
            __ray_call__=Remote(lambda *args: dict(equal=True)),
        )

    factory = SimpleNamespace(options=lambda **kw: SimpleNamespace(remote=create))

    def get(refs, **kw):
        return [ref() for ref in refs] if isinstance(refs, list) else refs()

    with (
        patch(
            "sys.argv",
            ["oracle", "--audit-root", str(root), "--output", str(tmp_path / "out"), "--node-ip", "test-node"],
        ),
        patch.object(ray, "init"),
        patch.object(
            ray,
            "nodes",
            return_value=[dict(Alive=True, NodeManagerAddress="test-node", NodeID="ab" * 28, Resources=dict(GPU=8))],
        ),
        patch.object(ray, "get", side_effect=get),
        patch.object(ray, "kill", side_effect=OSError("kill failed")),
        patch.object(gpu_replay, "CheckedEngine", factory),
        patch.object(store, "atomic_write", side_effect=write),
    ):
        with pytest.raises(
            (OSError if write_fails else RuntimeError),
            match=("injected final result write failure" if write_fails else "Reference engine cleanup failed"),
        ):
            memory_audit_oracle.main()
    assert closed == list(range(8))

    if not write_fails:
        result = json.loads((tmp_path / "out/result.json").read_text())
        assert result["status"] == "FAILED_CLEANUP" and len(result["cleanup_errors"]) == 2
