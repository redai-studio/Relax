# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""V4.1 rollout constraints using tiny on-disk checkpoint headers, never a GPU
model."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from .test_sglang_engine import server_args_module as server_args_module


def _checkpoint(path: Path, *, v41: bool = True, quantized: bool = False) -> str:
    path.mkdir()
    config: dict[str, Any] = {
        "model_type": "deepseek_v41" if v41 else "qwen3",
        "dtype": "bfloat16",
        "architectures": ["DeepseekV41ForCausalLM" if v41 else "Qwen3ForCausalLM"],
        "text_config": {"model_type": "deepseek_v41_text", "dtype": "bfloat16", "engram_layer_ids": [0]}
        if v41
        else {},
    }
    if quantized:
        config["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "fp4"}
    (path / "config.json").write_text(json.dumps(config))
    tensors = {
        "embed.weight": {"dtype": "BF16", "shape": [2, 4], "data_offsets": [0, 16]},
        "layers.0.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [2, 32], "data_offsets": [16, 80]},
        "layers.0.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [2, 1], "data_offsets": [80, 82]},
    }
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    shard = "model-00001-of-00001.safetensors"
    (path / shard).write_bytes(struct.pack("<Q", len(header)) + header + bytes(82))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, shard)}))
    return str(path)


def _args(checkpoint: str, **overrides: Any) -> SimpleNamespace:
    values = dict(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        hf_checkpoint=checkpoint,
        seed=17,
        colocate=True,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_load_format="auto",
        sglang_quantization=None,
        sglang_enable_weights_cpu_backup=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _compute(module: SimpleNamespace, args: SimpleNamespace, **overrides: Any) -> dict[str, Any]:
    kwargs, _ = module._compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=1236,
        base_gpu_id=0,
        sglang_overrides=overrides or None,
    )
    return kwargs


@pytest.mark.parametrize("value", [None, "0", "false"])
@pytest.mark.parametrize("colocate", [True, False])
def test_v41_offload_rejects_missing_or_disabled_host_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_args_module: SimpleNamespace,
    value: str | None,
    colocate: bool,
) -> None:
    if value is None:
        monkeypatch.delenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", value)
    args = _args(_checkpoint(tmp_path / "bf16"), colocate=colocate)
    with pytest.raises(ValueError, match="preserve frozen Engram"):
        _compute(server_args_module, args)


@pytest.mark.parametrize("value", ["1", "true"])
@pytest.mark.parametrize("colocate", [True, False])
def test_v41_host_table_preserves_offload_without_forcing_full_weights_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_args_module: SimpleNamespace,
    value: str,
    colocate: bool,
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", value)
    kwargs = _compute(server_args_module, _args(_checkpoint(tmp_path / "bf16"), colocate=colocate))
    assert kwargs["enable_memory_saver"] is True
    assert kwargs["enable_weights_cpu_backup"] is False
    assert kwargs["load_format"] == "auto"


@pytest.mark.parametrize("colocate", [True, False])
def test_v41_explicit_cpu_backup_is_an_alternative_to_host_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace, colocate: bool
) -> None:
    monkeypatch.delenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", raising=False)
    args = _args(_checkpoint(tmp_path / "bf16"), colocate=colocate)
    kwargs = _compute(server_args_module, args, enable_weights_cpu_backup=True)
    assert kwargs["enable_weights_cpu_backup"] is True


def test_v41_backup_override_cannot_bypass_retention_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace
) -> None:
    monkeypatch.delenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", raising=False)
    args = _args(_checkpoint(tmp_path / "bf16"), sglang_enable_weights_cpu_backup=True)
    with pytest.raises(ValueError, match="preserve frozen Engram"):
        _compute(server_args_module, args, enable_weights_cpu_backup=False)


@pytest.mark.parametrize("colocate", [True, False])
def test_v41_without_memory_saving_does_not_require_host_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_args_module: SimpleNamespace,
    colocate: bool,
) -> None:
    monkeypatch.delenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", raising=False)
    args = _args(_checkpoint(tmp_path / "bf16"), colocate=colocate)
    kwargs = _compute(server_args_module, args, enable_memory_saver=False)
    assert kwargs["enable_memory_saver"] is False
    assert kwargs["enable_weights_cpu_backup"] is False


def test_v41_dummy_override_is_rejected_even_with_safe_host_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    args = _args(_checkpoint(tmp_path / "bf16"))
    with pytest.raises(ValueError, match="dummy loading"):
        _compute(server_args_module, args, load_format="dummy")


def test_v41_quantization_override_is_rejected_for_bf16_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    args = _args(_checkpoint(tmp_path / "bf16"))
    with pytest.raises(ValueError, match="quantization override"):
        _compute(server_args_module, args, quantization="fp8")


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_v41_dtype_override_cannot_change_bf16_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace, dtype: str
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    args = _args(_checkpoint(tmp_path / "bf16"))
    with pytest.raises(ValueError, match="requires dtype"):
        _compute(server_args_module, args, dtype=dtype)


def test_v41_fp16_flag_is_rejected_by_final_rollout_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    args = _args(_checkpoint(tmp_path / "bf16"), fp16=True)
    with pytest.raises(ValueError, match="requires dtype"):
        _compute(server_args_module, args)


def test_final_model_path_override_from_ordinary_to_v41_cannot_bypass_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace
) -> None:
    monkeypatch.delenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", raising=False)
    args = _args(_checkpoint(tmp_path / "ordinary", v41=False))
    with pytest.raises(ValueError, match="preserve frozen Engram"):
        _compute(server_args_module, args, model_path=_checkpoint(tmp_path / "bf16"))


@pytest.mark.parametrize("target", ["missing", "ordinary", "quantized"])
def test_known_v41_cannot_override_to_an_unvalidated_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace, target: str
) -> None:
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    args = _args(_checkpoint(tmp_path / "bf16"))
    model_path = str(tmp_path / target)
    if target != "missing":
        model_path = _checkpoint(tmp_path / target, v41=target != "ordinary", quantized=target == "quantized")
    with pytest.raises(ValueError):
        _compute(server_args_module, args, model_path=model_path)


def test_ordinary_model_keeps_dummy_quantization_and_offload_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_args_module: SimpleNamespace
) -> None:
    monkeypatch.delenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", raising=False)
    args = _args(_checkpoint(tmp_path / "ordinary", v41=False))
    kwargs = _compute(server_args_module, args, load_format="dummy", quantization="fp8")
    assert kwargs["load_format"] == "dummy" and kwargs["quantization"] == "fp8"
    assert kwargs["enable_memory_saver"] is True and kwargs["enable_weights_cpu_backup"] is False
