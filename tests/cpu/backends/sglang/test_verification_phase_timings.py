import json

import pytest
import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.sglang_loader import _verification_timings, verify_qwen3_dense


def test_phase_telemetry_keeps_results_and_unwinds_after_failure(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
            }
        )
    )
    name = "model.norm.weight"
    expected = torch.arange(32, dtype=torch.float32)
    save_file({name: expected}, tmp_path / "model.safetensors")
    physical = {(0, name): expected.clone()}
    baseline = verify_qwen3_dense(tmp_path, None, physical_tensors=physical)
    timings = {}
    assert verify_qwen3_dense(tmp_path, None, physical_tensors=physical, timings=timings) == baseline
    assert all(value >= 0 for value in timings.values())
    assert {
        "checkpoint_tensor_mapping",
        "reference_contiguous_cpu",
        "reference_transfer",
        "byte_compare_sync",
        "layout_and_dispatch",
    } <= timings.keys()
    assert _verification_timings.get() is None
    physical[0, name][-1] += 1
    with pytest.raises(ValueError, match="Physical weight mismatch"):
        verify_qwen3_dense(tmp_path, None, physical_tensors=physical, timings={})
    assert _verification_timings.get() is None
