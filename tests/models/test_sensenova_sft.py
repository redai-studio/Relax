# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model math and conditioning gradient regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.models.sensenova.adapter import flow_matching_loss


@pytest.mark.parametrize(
    "device",
    [
        None,
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required for cross-device position tests"
            ),
        ),
    ],
)
def test_abs_positions_honors_requested_device(device):
    pytest.importorskip("torchvision")
    from relax.models.sensenova.processing import build_abs_positions_from_grid_hw

    grid = torch.tensor([[2, 2], [1, 3]], device="cpu")
    x, y = build_abs_positions_from_grid_hw(grid, device=device)
    assert x.device.type == (device or "cpu")
    assert y.device == x.device
    assert x.cpu().tolist() == [0, 1, 0, 1, 0, 1, 2]
    assert y.cpu().tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert grid.device.type == "cpu"


@pytest.mark.parametrize(
    "size,budget,expected",
    [
        ((64, 64), 1, (32, 32)),
        ((64, 64), 1024, (32, 32)),
        ((1024, 32), 1024, (160, 32)),
        ((32, 1024), 1024, (32, 160)),
        ((64, 64), 4096, (64, 64)),
        ((128, 64), 8192, (128, 64)),
    ],
)
def test_fixed_tier_image_keeps_at_least_one_patch(tmp_path, size, budget, expected):
    pytest.importorskip("torchvision")
    from PIL import Image

    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    path = tmp_path / "image.png"
    Image.new("RGB", size).save(path)
    image = SenseNovaSFTAdapter.load_image(path, max_pixels=1048576, fixed_pixels=budget)
    assert image.size == expected
    assert image.mode == "RGB"


def test_image_without_fixed_tier_still_rejects_subpatch_size(tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image

    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    with pytest.raises(ValueError, match="smaller than patch size"):
        SenseNovaSFTAdapter.load_image(path, max_pixels=1048576)


@pytest.mark.parametrize("seed", [None, 0, 19])
def test_fixed_tier_replay_preserves_training_rng(monkeypatch, seed):
    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    adapter = SenseNovaSFTAdapter()
    monkeypatch.setattr(adapter, "load_image", lambda path, max_pixels, fixed_pixels=None: fixed_pixels)
    row = {"prompt": "edit", "image": "target.png", "edit_image": ["source.png", "glyph.png"]}
    config = {"max_pixels": 1048576, "fixed_pixels_tiers": [1440**2, 2048**2], "seed": seed}
    rng = torch.get_rng_state().clone()
    samples = [adapter.prepare_sample(row, Path("/data"), config, index=i) for i in range(12)]
    restored = SenseNovaSFTAdapter()
    monkeypatch.setattr(restored, "load_image", lambda path, max_pixels, fixed_pixels=None: fixed_pixels)
    replay = [restored.prepare_sample(row, Path("/data"), config, index=i) for i in range(12)]
    assert samples == replay
    for sample in samples:
        assert sample["image"] in config["fixed_pixels_tiers"]
        assert sample["edit_image"] == [sample["image"], sample["image"]]
    assert torch.equal(rng, torch.get_rng_state())


def test_flow_matching_loss_and_gradient() -> None:
    weight = torch.nn.Parameter(torch.tensor(0.25))
    clean = torch.tensor([1.0, -2.0])
    noise = torch.tensor([-1.0, 3.0])
    sigma = 0.5
    scheduler = SimpleNamespace(
        timesteps=torch.tensor([500.0]),
        add_noise=lambda x, e, t: x * (1 - sigma) + e * sigma,
        training_target=lambda x, e, t: e - x,
        training_weight=lambda t: torch.tensor(1.7),
    )
    pipe = SimpleNamespace(
        scheduler=scheduler,
        device="cpu",
        torch_dtype=torch.float32,
        dit=None,
        model_fn=lambda dit, latents, **kwargs: latents * weight,
    )
    inputs = {"input_latents": clean}
    loss = flow_matching_loss(pipe, inputs, timestep_id=torch.tensor([0]), noise=noise)
    loss.backward()
    x = clean * (1 - sigma) + noise * sigma
    residual = x * 0.25 - (noise - clean)
    torch.testing.assert_close(loss, residual.square().mean() * 1.7)
    torch.testing.assert_close(weight.grad, 2 * (residual * x).mean() * 1.7)
    assert "latents" not in inputs


def test_full_sft_keeps_conditioning_gradient(monkeypatch) -> None:
    pytest.importorskip("torchvision")
    from relax.models.sensenova import adapter as sft

    prefix_weight = torch.nn.Parameter(torch.tensor(2.0))

    def unit_runner(unit, pipe, shared, positive, negative):
        return {**shared, "prefix": prefix_weight * 3}, positive, negative

    pipe = torch.nn.Module()
    pipe.device = "cpu"
    pipe.units = [object()]
    pipe.unit_runner = unit_runner
    data = {"image": SimpleNamespace(height=32, width=32), "edit_image": [object()], "prompt": "edit"}
    monkeypatch.setattr(sft, "flow_matching_loss", lambda pipe, inputs: inputs["prefix"])
    sft.SenseNovaSFTModel(pipe, training_mode="full")(data).backward()
    assert prefix_weight.grad == 3
    assert not sft.SenseNovaSFTModel(pipe, training_mode="lora")(data).requires_grad


def test_sensenova_uses_decoder_layers_as_fsdp_boundaries() -> None:
    pytest.importorskip("torchvision")
    from relax.backends.fsdp.runtime import resolve_block_classes, resolve_fine_grained_modules
    from relax.models.sensenova.adapter import SenseNovaSFTModel
    from relax.models.sensenova.model import SenseNovaU1DecoderLayer, SenseNovaU1ForCausalLM

    pipe = torch.nn.Module()
    pipe.language_model = SenseNovaU1ForCausalLM(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        pad_token_id=0,
    )
    model = SenseNovaSFTModel(pipe, training_mode="full")
    assert model._no_split_modules == ("SenseNovaU1DecoderLayer",)
    assert resolve_block_classes(model) == [SenseNovaU1DecoderLayer]
    fine_grained = resolve_fine_grained_modules(model)
    assert len(fine_grained) == 12
    assert sum(isinstance(module, torch.nn.Embedding) for module in fine_grained) == 1
    assert sum(isinstance(module, torch.nn.Linear) for module in fine_grained) == 11

    lora_pipe = torch.nn.Module()
    lora_pipe.layer = SenseNovaU1DecoderLayer(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
    )
    assert resolve_fine_grained_modules(SenseNovaSFTModel(lora_pipe, training_mode="lora")) == []


def test_sensenova_fsdp_wraps_fine_grained_modules_before_decoder(monkeypatch) -> None:
    pytest.importorskip("torchvision")
    from relax.backends.fsdp.runtime import fsdp2_wrap
    from relax.models.sensenova.adapter import SenseNovaSFTModel
    from relax.models.sensenova.model import SenseNovaU1DecoderLayer

    pipe = torch.nn.Module()
    pipe.layer = SenseNovaU1DecoderLayer(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
    )
    model = SenseNovaSFTModel(pipe, training_mode="full")
    wrapped = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: wrapped.append(module))

    fsdp2_wrap(model)

    assert len(wrapped[:-2]) == 10
    assert all(isinstance(module, torch.nn.Linear) for module in wrapped[:-2])
    assert isinstance(wrapped[-2], SenseNovaU1DecoderLayer)
    assert wrapped[-1] is model


def test_prefix_cache_tensors_are_decoder_outputs() -> None:
    pytest.importorskip("torchvision")
    from relax.models.sensenova.adapter import configure_attention
    from relax.models.sensenova.model import SenseNovaU1Model

    configure_attention("torch")
    model = SenseNovaU1Model(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        pad_token_id=0,
    )
    indexes = torch.zeros(3, 2, dtype=torch.long)
    prefix, cache = model(
        inputs_embeds=torch.randn(1, 2, 32),
        image_gen_indicators=torch.zeros(1, 2, dtype=torch.bool),
        indexes=indexes,
        use_cache=True,
        update_cache=True,
        use_gradient_checkpointing=True,
    )
    assert prefix.shape == (1, 2, 32)
    assert cache.get_seq_length() == 2
    assert all(layer.keys.grad_fn is not None for layer in cache.layers)

    generated, returned_cache = model(
        inputs_embeds=torch.randn(1, 2, 32),
        image_gen_indicators=torch.ones(1, 2, dtype=torch.bool),
        indexes=indexes,
        past_key_values=cache,
        use_cache=True,
        update_cache=False,
    )
    generated.sum().backward()
    assert returned_cache is cache
    assert cache.get_seq_length() == 2
    assert model.layers[-1].self_attn.k_proj.weight.grad is not None


def test_export_uses_dit_relative_keys_and_unit_scaling(tmp_path, monkeypatch) -> None:
    from safetensors.torch import load_file

    from relax.models.sensenova import export

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "COMMITTED").touch()
    contract = checkpoint_dir / "adapter_contract.json"
    contract.write_text(json.dumps({"lora": {"rank": 2, "alpha": 2}}))
    state = {
        "pipe.dit.layer.lora_A.default.weight": torch.randn(2, 4),
        "pipe.dit.layer.lora_B.default.weight": torch.randn(3, 2),
    }
    monkeypatch.setattr(export, "_load_dcp_model_state", lambda _: state)
    output = tmp_path / "adapter.safetensors"
    export.export_adapter(checkpoint_dir, output)
    restored = load_file(str(output))
    assert restored.keys() == {name.removeprefix("pipe.dit.") for name in state}
    for name, tensor in state.items():
        assert torch.equal(restored[name.removeprefix("pipe.dit.")], tensor.bfloat16())
    contract.write_text(json.dumps({"lora": {"rank": 2, "alpha": 4}}))
    with pytest.raises(ValueError, match="alpha == rank"):
        export.export_adapter(checkpoint_dir, tmp_path / "invalid.safetensors")


def test_full_export_preserves_keys_shapes_and_bf16_values(tmp_path, monkeypatch) -> None:
    from safetensors.torch import load_file, save_file

    from relax.models.sensenova import export as full_export

    base, saved = tmp_path / "base", tmp_path / "saved"
    base.mkdir()
    saved.mkdir()
    (saved / "COMMITTED").touch()
    (saved / "adapter_contract.json").write_text(json.dumps({"trainable_mode": "full"}))
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7
    save_file({"weight": torch.zeros_like(tensor)}, str(base / "model.safetensors"))
    monkeypatch.setattr(full_export, "_load_dcp_model_state", lambda _: {"pipe.dit.weight": tensor})
    output = full_export.export_full(saved, base, tmp_path / "export")
    restored = load_file(str(output / "model-00001.safetensors"))
    torch.testing.assert_close(restored["weight"], tensor.bfloat16(), rtol=0, atol=0)
    assert json.loads((output / "export.json").read_text())["tensors"] == 1
    save_file({"weight": torch.zeros(3, 2)}, str(base / "model.safetensors"))
    with pytest.raises(ValueError, match="shape mismatch"):
        full_export.export_full(saved, base, tmp_path / "invalid")
