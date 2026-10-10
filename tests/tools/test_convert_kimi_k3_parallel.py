# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/models/kimi-k3/tools"))
import convert_kimi_k3_torch_dist_to_hf as baseline
import convert_kimi_k3_torch_dist_to_hf_parallel as parallel


class ExpertLayout:
    def __init__(self, rank: int = 0):
        self.weight_map = {
            f"language_model.model.layers.1.block_sparse_moe.experts.{expert}.w{projection}.weight{suffix}": "expert.safetensors"
            for expert in range(4)
            for projection in (1, 2, 3)
            for suffix in ("_packed", "_scale")
        }
        self.owners = {"expert.safetensors": 0}
        self.rank = rank

    def owns(self, name: str) -> bool:
        return self.rank == 0

    def spec(self, name: str) -> dict:
        rows, cols = (64, 32) if ".w2." in name else (32, 64)
        return {"dtype": "U8", "shape": [rows, cols // (32 if name.endswith("_scale") else 2)]}


def _weight(rank: int, fc: int) -> torch.Tensor:
    shape = (64, 64) if fc == 1 else (64, 32)
    generator = torch.Generator().manual_seed(911 + 31 * rank + fc)
    weight = torch.randn(shape, generator=generator).to(torch.bfloat16)
    weight[0] = 0
    weight[1, :8] = torch.tensor([0.25, -0.25, 0.75, -0.75, 1.25, 1.75, 2.5, 6.0])
    return weight


def _distributed_worker(rank: int, init_file: str) -> None:
    import torch.distributed as dist
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, RowParallelMapping
    from megatron.bridge.models.conversion.quantization_utils import quantize_mxfp4_e2m1_like_scale

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    singleton = None
    for member in range(2):
        group = dist.new_group([member], backend="gloo")
        if member == rank:
            singleton = group
    groups = SimpleNamespace(pp=singleton, tp=singleton, expt_tp=singleton, ep=dist.group.WORLD)
    module = SimpleNamespace(config=SimpleNamespace(num_moe_experts=4))
    layout = ExpertLayout(rank)
    quantizer = parallel.LocalExpertQuantizer("cpu", quantize_mxfp4_e2m1_like_scale)
    owner_hook = baseline.NativeK3Transform(layout, "cpu", quantize_mxfp4_e2m1_like_scale)
    try:
        for fc in (1, 2):
            for local_expert in (0, 1):
                expert = rank * 2 + local_expert
                prefix = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}"
                megatron_name = f"decoder.layers.1.mlp.experts.linear_fc{fc}.weight{expert}"
                mapping = (
                    GatedMLPMapping(megatron_name, prefix + ".w1.weight", prefix + ".w3.weight")
                    if fc == 1
                    else RowParallelMapping(megatron_name, prefix + ".w2.weight")
                )
                mapping.set_process_groups_from_pg_collection(groups)
                weight = _weight(expert, fc)
                task = SimpleNamespace(weight_dtype=None)
                reference = owner_hook(task, mapping.megatron_to_hf(weight, module), None)
                restore = parallel.install_parallel_mappings(layout, quantizer)
                try:
                    actual = owner_hook(task, mapping.megatron_to_hf(weight, module), None)
                finally:
                    restore()
                assert set(actual) == set(reference)
                for key in actual:
                    assert actual[key].dtype == reference[key].dtype
                    assert torch.equal(actual[key], reference[key]), key
        assert quantizer.calls == 4
    finally:
        dist.destroy_process_group()


def test_two_rank_ep_quantization_matches_gather_then_quantize(tmp_path: Path) -> None:
    pytest.importorskip("megatron.bridge.models.conversion.param_mapping")
    pytest.importorskip("megatron.bridge.models.conversion.quantization_utils")
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


@pytest.mark.parametrize(
    "native_shape,gathered_shape,valid",
    [
        ((32, 2), (32, 2), True),
        ((32, 2), (32, 2, 1), True),
        ((64, 1), (64, 1), True),
        ((1, 2), (1, 2), True),
        ((1, 2), (2, 1), True),
        ((1, 1), (1, 1), True),
        ((1, 1), (1,), True),
        ((32, 2), (2, 32), False),
        ((32, 2), (64, 1), False),
        ((32, 2), (1, 32, 2), False),
        ((32, 2), (32, 3, 1), False),
    ],
)
def test_expert_scale_restores_only_known_bridge_layout(
    native_shape: tuple[int, int], gathered_shape: tuple[int, ...], valid: bool
) -> None:
    name = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
    scale = torch.arange(math.prod(gathered_shape), dtype=torch.uint8).reshape(gathered_shape)
    packed = torch.zeros((native_shape[0], native_shape[1] * 16), dtype=torch.int8)
    layout = ExpertLayout()
    layout.spec = lambda key: {"dtype": "U8", "shape": list(native_shape)}
    mapping = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        is_expert=True,
        hf_param=name,
        megatron_to_hf_quant=lambda *args: {name: packed, name + "_scale_inv": scale},
    )
    if not valid:
        with pytest.raises(ValueError, match="HF scale shape mismatch"):
            parallel.convert_expert(mapping, None, None, None, layout)
        return
    result = parallel.convert_expert(mapping, None, None, None, layout)
    assert result[name + "_packed"] is packed
    assert tuple(result[name + "_scale"].shape) == native_shape
    assert result[name + "_scale"].dtype == torch.uint8
    assert torch.equal(result[name + "_scale"].flatten(), scale.flatten())


def test_invalid_local_geometry_is_rejected() -> None:
    quantizer = parallel.LocalExpertQuantizer("cpu", lambda *_: pytest.fail("must reject before quantizing"))
    with pytest.raises(ValueError, match="geometry"):
        quantizer(torch.empty((2, 33), dtype=torch.bfloat16), (1, 32))
    with pytest.raises(ValueError, match="BF16"):
        quantizer(torch.empty((2, 64), dtype=torch.float32), (1, 32))


def test_dense_and_shared_experts_are_not_selected() -> None:
    for name in (
        "language_model.model.layers.1.mlp.gate_proj.weight",
        "language_model.model.layers.1.block_sparse_moe.shared_experts.w1.weight",
    ):
        assert not parallel._expert_names(SimpleNamespace(is_expert=False, hf_param=name))


def test_expert_tp_is_guarded() -> None:
    with pytest.raises(ValueError, match="expert-TP"):
        parallel.convert_expert(SimpleNamespace(tp_size=2, pp_size=1), None, None, None, None)


def test_full_shard_comparison_ignores_header_metadata_but_detects_weight_changes(tmp_path: Path) -> None:
    from compare_kimi_k3_hf_exports import compare_shard
    from safetensors.torch import save_file

    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    tensor = torch.arange(128, dtype=torch.uint8).reshape(4, 32)
    save_file({"packed": tensor}, str(left / "a.safetensors"), metadata={"note": "baseline"})
    save_file({"packed": tensor}, str(right / "a.safetensors"), metadata={"note": "candidate"})
    result = compare_shard(str(left), str(right), "a.safetensors")
    assert not result["exact_file_match"]
    assert result["mismatched_keys"] == []
    tensor[0, 0] = 255
    save_file({"packed": tensor}, str(right / "a.safetensors"))
    assert compare_shard(str(left), str(right), "a.safetensors")["mismatched_keys"] == ["packed"]


@pytest.mark.parametrize("merge_lora", [False, True])
def test_worker_disables_early_quantization_before_lora_merge(tmp_path: Path, monkeypatch, merge_lora: bool) -> None:
    pytest.importorskip("megatron.bridge.models.conversion.quantization_utils")
    installed = []
    transform = SimpleNamespace(layout=object())
    monkeypatch.setattr(baseline, "_install_export_hooks", lambda *args: None)

    def mappings(*args):
        installed.append(True)
        return lambda: None

    monkeypatch.setattr(parallel, "install_parallel_mappings", mappings)

    def worker(rank, master, port, args):
        baseline._install_export_hooks(None, args.input_dir, transform)
        return {"tensors": 1}

    monkeypatch.setattr(baseline, "_worker", worker)
    args = SimpleNamespace(
        repo_root=str(Path(__file__).resolve().parents[2]),
        input_dir="unused",
        staging_dir=str(tmp_path),
        merge_lora=merge_lora,
    )
    if merge_lora:
        assert parallel._worker(0, "unused", 0, args) == {"tensors": 1}
        assert not installed
    else:
        # Stub worker did not quantize anything: full export must retain its guard.
        with pytest.raises(RuntimeError, match="never invoked"):
            parallel._worker(0, "unused", 0, args)
        assert installed
