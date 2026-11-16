# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU export round-trip through real Diffusers serialization and LoRA
loading."""

import json
from unittest.mock import Mock

import pytest
import torch

from relax.backends.fsdp import checkpoint
from relax.models.qwen_image.sft_adapter import QwenImageSFTAdapter


MODULE = "transformer.transformer_blocks.0.attn.to_q"


@pytest.fixture
def export_case(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "COMMITTED").touch()
    contract = {
        "family": "qwen_image",
        "generation_task": "t2i",
        "trainable_mode": "lora",
        "lora": {"rank": 2, "alpha": 2},
    }
    (directory / "adapter_contract.json").write_text(json.dumps(contract))
    state = {
        f"{MODULE}.lora_A.default.weight": torch.arange(16, dtype=torch.float32).reshape(2, 8) / 97,
        f"{MODULE}.lora_B.default.weight": (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 103).T,
    }
    loader = Mock(return_value=state)
    monkeypatch.setattr(checkpoint, "_load_dcp_model_state", loader)
    return directory, tmp_path / "export", contract, state, loader


def test_export_uses_real_diffusers_format_and_loads_fp32_lora_into_tiny_qwen(export_case):
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
    from safetensors.torch import load_file

    directory, output, contract, state, loader = export_case
    QwenImageSFTAdapter.export_checkpoint(directory, output, contract)
    loader.assert_called_once_with(str(directory / "fsdp"))
    assert json.loads((output / "relax_adapter_contract.json").read_text()) == contract
    serialized = load_file(output / "pytorch_lora_weights.safetensors")
    expected = {name.replace(".default.", "."): tensor for name, tensor in state.items()}
    assert set(serialized) == set(expected)
    for name, tensor in serialized.items():
        assert tensor.dtype == torch.float32 and tensor.is_contiguous()
        assert torch.equal(tensor, expected[name])

    # Exercise the actual consumer too: a finite safetensors dump with wrong
    # names is insufficient evidence that Diffusers can apply the trained LoRA.
    transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=8,
        out_channels=2,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        joint_attention_dim=4,
        axes_dims_rope=(2, 2, 4),
    )
    loaded = QwenImagePipeline.lora_state_dict(str(output))
    QwenImagePipeline.load_lora_into_transformer(loaded, transformer, adapter_name="roundtrip")
    projection = transformer.transformer_blocks[0].attn.to_q
    matrix_a, matrix_b = state[f"{MODULE}.lora_A.default.weight"], state[f"{MODULE}.lora_B.default.weight"]
    assert torch.equal(projection.lora_A["roundtrip"].weight, matrix_a)
    assert torch.equal(projection.lora_B["roundtrip"].weight, matrix_b)
    assert projection.lora_A["roundtrip"].weight.dtype == projection.lora_B["roundtrip"].weight.dtype == torch.float32
    assert projection.scaling["roundtrip"] == 1.0
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 7
    # Serialized parameters are contiguous. Match that storage layout in the
    # oracle so a transposed-weight GEMM kernel does not add unrelated rounding.
    expected_result = projection.base_layer(inputs) + torch.nn.functional.linear(
        torch.nn.functional.linear(inputs, matrix_a.contiguous()), matrix_b.contiguous()
    )
    torch.testing.assert_close(projection(inputs), expected_result, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["uncommitted", "contract", "existing_output"])
def test_export_rejects_uncommitted_or_mismatched_or_existing_destination(export_case, fault):
    directory, output, contract, _state, loader = export_case
    if fault == "uncommitted":
        (directory / "COMMITTED").unlink()
    elif fault == "contract":
        contract = {**contract, "generation_task": "i2i"}
    else:
        output.mkdir()
        (output / "owned.txt").write_text("preserve")
    with pytest.raises(ValueError):
        QwenImageSFTAdapter.export_checkpoint(directory, output, contract)
    loader.assert_not_called()
    if fault == "existing_output":
        assert [path.name for path in output.iterdir()] == ["owned.txt"]
        assert (output / "owned.txt").read_text() == "preserve"
    else:
        assert not output.exists()


@pytest.mark.parametrize("rank", [0, -1, True, False, 2.5, "2", None])
def test_export_rejects_invalid_rank_before_loading_checkpoint(export_case, rank):
    directory, output, contract, _state, loader = export_case
    contract["lora"] = {"rank": rank, "alpha": rank}
    (directory / "adapter_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="DiT LoRA"):
        QwenImageSFTAdapter.export_checkpoint(directory, output, contract)
    loader.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("alpha", [True, "2", 0, 3, float("inf")])
def test_export_rejects_unsupported_alpha_contract(export_case, alpha):
    directory, output, contract, _state, loader = export_case
    contract["lora"]["alpha"] = alpha
    (directory / "adapter_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="DiT LoRA"):
        QwenImageSFTAdapter.export_checkpoint(directory, output, contract)
    loader.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("missing", ["lora_A", "lora_B"])
def test_export_rejects_incomplete_lora_pair(export_case, missing):
    directory, output, contract, state, _loader = export_case
    state.pop(f"{MODULE}.{missing}.default.weight")
    with pytest.raises(ValueError, match="Incomplete.*A/B pair"):
        QwenImageSFTAdapter.export_checkpoint(directory, output, contract)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["empty", "rank", "shape", "name", "scope", "integer", "nan", "infinite"])
def test_export_rejects_unusable_checkpoint_tensors_before_creating_output(export_case, fault):
    directory, output, contract, state, _loader = export_case
    name = f"{MODULE}.lora_A.default.weight"
    if fault == "empty":
        state.clear()
    elif fault == "rank":
        state[name] = torch.zeros(3, 8)
    elif fault == "shape":
        state[name] = state[name].flatten()
    elif fault == "name":
        state[name.replace("lora_A", "base_layer")] = state.pop(name)
    elif fault == "scope":
        state[name.replace("transformer.", "text_encoder.", 1)] = state.pop(name)
    elif fault == "integer":
        state[name] = state[name].to(torch.int64)
    else:
        state[name][0, 0] = float("nan" if fault == "nan" else "inf")
    with pytest.raises(ValueError):
        QwenImageSFTAdapter.export_checkpoint(directory, output, contract)
    assert not output.exists()


def test_export_requires_separate_non_distributed_process(export_case, monkeypatch):
    directory, output, contract, _state, loader = export_case
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="separate CPU process"):
        QwenImageSFTAdapter.export_checkpoint(directory, output, contract)
    loader.assert_not_called()
    assert not output.exists()
