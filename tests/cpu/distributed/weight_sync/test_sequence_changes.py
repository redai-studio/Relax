# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def test_sequence_counts_shared_parameters_once_and_keeps_unchanged_steps(tmp_path):
    """A changed tied parameter counts once; an unchanged next step remains
    visible."""
    script = Path(__file__).resolve().parents[4] / "task_delta_weight_sync/count_sequence_changes.py"
    spec = importlib.util.spec_from_file_location("sequence_changes", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    (run / "versions").mkdir(parents=True)
    for version in range(3):
        bits = torch.tensor([0x3F800000 + (version > 0), 0x3F800000], dtype=torch.int32)
        values = {
            "embedding.weight": bits.view(torch.float32),
            "lm_head.weight": bits.view(torch.float32).clone(),
            "other.weight": torch.zeros(1),
        }
        export = run / "exports" / f"update-{version:06d}"
        export.mkdir(parents=True)
        save_file(values, export / "model.safetensors")
        gpu = run / "gpu" / f"version-{version}" / "delivered" / "rank-0"
        gpu.mkdir(parents=True)
        weight_map = {}
        for index, (name, tensor) in enumerate(values.items()):
            filename = f"tensor-{index}.safetensors"
            save_file({name: tensor}, gpu / filename)
            weight_map[name] = filename
        (gpu / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (gpu / "coverage.json").write_text(
            json.dumps(
                {
                    "persistent_buffer_count": 0,
                    "parameter_aliases": [["embedding.weight", "lm_head.weight"]],
                }
            )
        )
        diff = run / f"readback-{version}.json"
        diff.write_text(json.dumps({"equal": True, "right": str(gpu)}))
        receipt = {
            "status": "ACTIVE",
            "metrics": {"installation": {"members": [{"engine": 0, "readback": [{"diff": str(diff)}]}]}},
        }
        (run / "versions" / f"{version:06d}.json").write_text(json.dumps(receipt))
    rows = module.count_changes(run, 2, tmp_path / "counts")
    assert len(rows) == 2
    assert rows[0]["named_hf"]["mismatch_elements"] == rows[0]["named_gpu"]["mismatch_elements"] == 2
    assert (rows[0]["unique_parameters"], rows[0]["changed_parameters"], rows[0]["changed_bits"]) == (3, 1, 1)
    assert (rows[1]["changed_parameters"], rows[1]["changed_bytes"], rows[1]["changed_bits"]) == (0, 0, 0)
    assert rows[1]["named_hf"]["equal"] is True
    assert json.loads((tmp_path / "counts/summary.json").read_text())[1]["version"] == 2
