# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the Megatron-FSDP + PP patches (``fsdp_pp_patch``).

Cover the mesh helpers and checkpoint key mapping, including a CPU DCP/Adam
round trip. No GPU, distributed process group, or Megatron runtime is required.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp


# Load the CPU-only patch without the backend package's GPU initialization.
_SPEC = importlib.util.spec_from_file_location(
    "fsdp_pp_patch_under_test",
    Path(__file__).resolve().parents[4] / "relax/backends/megatron/fsdp_pp_patch.py",
)
fsdp_pp_patch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fsdp_pp_patch)
_rename_layers_to_global = fsdp_pp_patch._rename_layers_to_global
check_fsdp_pp_checkpoint_support = fsdp_pp_patch.check_fsdp_pp_checkpoint_support
pp_stage_window = fsdp_pp_patch.pp_stage_window
stage_local_arange = fsdp_pp_patch.stage_local_arange


@pytest.mark.parametrize("role", ["reference", "actor_fwd", "actor"])
@pytest.mark.parametrize("base_fsdp", [False, True])
def test_actor_init_applies_role_overrides_before_fsdp_setup(monkeypatch, role, base_fsdp):
    """Exercise real initialization until the base actor would start Ray."""
    root = Path(__file__).resolve().parents[4]
    actor_path = root / "relax/backends/megatron/actor.py"
    tree = ast.parse(actor_path.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    method = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "_init")
    method.decorator_list = []
    actor.body = [method]
    utils_tree = ast.parse((root / "relax/utils/utils.py").read_text())
    process_args = next(
        node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_args"
    )
    calls = []

    class ReachedBaseInit(Exception):
        pass

    class TrainRayActor:
        def init(self, *args):
            raise ReachedBaseInit

    namespace = {
        "TrainRayActor": TrainRayActor,
        "monkey_patch_torch_dist": lambda args: calls.append(("dist", args.use_megatron_fsdp)),
        "_patch_clip_grad_for_fsdp_empty_grads": lambda: calls.append("clip"),
        "check_fsdp_pp_checkpoint_support": lambda args, role: calls.append(("checkpoint", role)),
    }
    for name in (
        "patch_fsdp_mesh_for_pp",
        "patch_dtensor_meta_check_for_pp",
        "patch_bridge_shared_embedding_broadcast_for_pp",
        "patch_embedding_grad_reshard_for_pp",
        "patch_checkpoint_layer_numbering_for_pp",
    ):
        namespace[name] = lambda name=name: calls.append(name)
    monkeypatch.setitem(
        sys.modules, "relax.utils.checkpoint_write_patch", SimpleNamespace(patch_checkpoint_write=lambda: None)
    )
    compiled = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            process_args,
            actor,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(compiled), str(actor_path), "exec"), namespace)
    args = SimpleNamespace(
        use_megatron_fsdp=base_fsdp,
        pipeline_model_parallel_size=1,
        ref_actor_config={"use_megatron_fsdp": not base_fsdp, "pipeline_model_parallel_size": 2},
        log_probs_max_tokens_per_gpu=128,
        ref_load="reference-checkpoint",
    )

    with pytest.raises(ReachedBaseInit):
        namespace["MegatronTrainRayActor"]()._init(args, role)

    expected_fsdp = base_fsdp if role == "actor" else not base_fsdp
    assert calls[0] == ("dist", expected_fsdp)
    assert ("clip" in calls) == expected_fsdp
    assert ("patch_fsdp_mesh_for_pp" in calls) == (expected_fsdp and role != "actor")


def _default_order_pp_group(world_size: int, pp_size: int, rank: int) -> list[int]:
    """PP group of ``rank`` under Megatron's default ``'tp-cp-ep-dp-pp'``
    order.

    Derived independently of the code under test: ``pp`` is the slowest-varying
    axis, so stage ``i`` owns the contiguous block
    ``[i * stage_size, (i + 1) * stage_size)`` and a rank's pipeline peers all sit
    at the same offset inside every block.
    """
    stage_size = world_size // pp_size
    return [rank % stage_size + i * stage_size for i in range(pp_size)]


# ---------------------------------------------------------------------------
# pp_stage_window / stage_local_arange
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "world_size,pp_size",
    [(8, 1), (8, 2), (8, 4), (16, 2), (16, 4), (16, 8), (12, 3), (24, 4)],
)
def test_pp_stage_window_matches_megatron_default_rank_order(world_size, pp_size):
    """Every rank lands in the contiguous block its pipeline stage owns."""
    stage_size = world_size // pp_size
    for rank in range(world_size):
        base, size = pp_stage_window(world_size, rank, _default_order_pp_group(world_size, pp_size, rank))
        assert size == stage_size
        assert base == (rank // stage_size) * stage_size
        assert base <= rank < base + size, "a rank must fall inside its own stage window"


def test_stage_local_arange_partitions_the_world_into_contiguous_blocks():
    """The per-rank windows must tile the world exactly — no gap, no
    overlap."""
    world_size, pp_size = 16, 4
    blocks = {
        tuple(stage_local_arange(world_size, rank, _default_order_pp_group(world_size, pp_size, rank)).tolist())
        for rank in range(world_size)
    }
    assert blocks == {(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)}


@pytest.mark.parametrize("world_size", [1, 8, 16])
def test_stage_local_arange_is_identity_at_pp1(world_size):
    """The patch must not perturb the already-working PP==1 mesh: at pp_size ==
    1 the shim has to reproduce ``torch.arange(world_size)`` exactly."""
    for rank in range(world_size):
        assert torch.equal(stage_local_arange(world_size, rank, [rank]), torch.arange(world_size))


def test_pp_stage_window_rejects_non_contiguous_stages():
    """``--use-tp-pp-dp-mapping`` (order ``'tp-cp-ep-pp-dp'``) makes pp an
    inner axis, so a stage's ranks are strided by 1 rather than by stage_size.

    The whole stage-window approach breaks down there and must fail loudly, not
    silently build a wrong mesh.
    """
    with pytest.raises(RuntimeError, match="use-tp-pp-dp-mapping"):
        pp_stage_window(world_size=8, rank=0, pp_ranks=[0, 1])


@pytest.mark.parametrize(
    "world_size,rank,pp_ranks,expected",
    [
        (8, 0, [], "Empty"),
        (8, 0, [0, 1, 2], "not divisible"),
        (8, 3, [0, 4], "not a member"),
    ],
)
def test_pp_stage_window_rejects_malformed_pp_ranks(world_size, rank, pp_ranks, expected):
    with pytest.raises(ValueError, match=expected):
        pp_stage_window(world_size, rank, pp_ranks)


# ---------------------------------------------------------------------------
# _rename_layers_to_global
# ---------------------------------------------------------------------------


def test_rename_layers_to_global_is_noop_at_offset_zero():
    """PP stage 0 (and effectively PP==1) must leave the state dict
    untouched."""
    state = {"decoder.layers.0.mlp.weight": "w0", "decoder.layers.1.mlp.weight": "w1"}
    _rename_layers_to_global(state, 0)
    assert state == {"decoder.layers.0.mlp.weight": "w0", "decoder.layers.1.mlp.weight": "w1"}


def test_rename_layers_to_global_shifts_keys_but_keeps_the_same_tensor_objects():
    """The load path depends on the tensor *object* surviving the rename:
    Megatron snapshots ``state_dict["model"]`` before this hook runs and
    restores it after ``torch.distributed.checkpoint`` has filled the renamed
    dict in place.

    Both dicts must point at the same tensors, otherwise the loaded values are
    dropped.
    """
    weight = torch.zeros(2)
    state = {"decoder.layers.0.mlp.weight": weight, "decoder.layers.1.mlp.weight": torch.ones(2)}

    _rename_layers_to_global(state, 8)

    assert sorted(state) == ["decoder.layers.8.mlp.weight", "decoder.layers.9.mlp.weight"]
    assert state["decoder.layers.8.mlp.weight"] is weight


def test_rename_layers_to_global_skips_mtp_and_unnumbered_keys():
    """MTP and its nested HybridStack retain independent layer numbering."""
    state = {
        "mtp.layers.0.weight": "mtp",
        "language_model.mtp.layers.0.mtp_model_layer.layers.0.mlp.weight": "hybrid_mtp",
        "embedding.word_embeddings.weight": "emb",
        "decoder.layers.2.weight": "dec",
    }

    _rename_layers_to_global(state, 4)

    assert set(state) == {
        "mtp.layers.0.weight",
        "language_model.mtp.layers.0.mtp_model_layer.layers.0.mlp.weight",
        "embedding.word_embeddings.weight",
        "decoder.layers.6.weight",
    }


# ---------------------------------------------------------------------------
# check_fsdp_pp_checkpoint_support
# ---------------------------------------------------------------------------


def _ckpt_args(**overrides):
    """Args that sit exactly on the guarded combination, so a single override
    flips the outcome."""
    values = {
        "use_megatron_fsdp": True,
        "pipeline_model_parallel_size": 2,
        "ckpt_format": "fsdp_dtensor",
        "save": "/tmp/ckpt",
        "virtual_pipeline_model_parallel_size": None,
        "mtp_num_layers": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mtp_num_layers", [None, 1])
def test_check_fsdp_pp_checkpoint_support_allows_the_covered_combination(mtp_num_layers):
    check_fsdp_pp_checkpoint_support(_ckpt_args(mtp_num_layers=mtp_num_layers))


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual pipeline"),
        ({"virtual_pipeline_model_parallel_size": 2, "mtp_num_layers": 1}, "virtual pipeline"),
    ],
)
def test_check_fsdp_pp_checkpoint_support_refuses_uncovered_combinations(override, expected):
    """Refuse rather than silently mis-number layers in the saved
    checkpoint."""
    with pytest.raises(RuntimeError, match=expected):
        check_fsdp_pp_checkpoint_support(_ckpt_args(**override))


@pytest.mark.parametrize(
    "gate",
    [
        {"use_megatron_fsdp": False},
        {"pipeline_model_parallel_size": 1},
        {"ckpt_format": "torch_dist"},
        {"save": None},
    ],
)
def test_check_fsdp_pp_checkpoint_support_short_circuits_on_each_gate(gate):
    """Each early-return gate is exercised against an otherwise-fatal VPP
    config, so the test fails if that gate ever stops firing."""
    check_fsdp_pp_checkpoint_support(_ckpt_args(virtual_pipeline_model_parallel_size=2, **gate))


def test_mtp_checkpoint_restores_weights_adam_and_group_metadata(tmp_path, monkeypatch):
    """Real CPU DCP/Adam round trip with MCore's shallow-snapshot load
    protocol.

    Assemble two PP stages in one process to expose duplicate checkpoint keys
    without adding a distributed fixture. The CPU adapter omits only MCore's
    GPU tensor-layout transforms; the actual Relax preprocessing hook is used.
    """
    checkpointing = SimpleNamespace(preprocess_fsdp_dtensor_state_dict=lambda args, state, model: state.copy())
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing)
    monkeypatch.setattr(fsdp_pp_patch, "_pp_global_layer_offset", lambda config: config.offset)
    fsdp_pp_patch.patch_checkpoint_layer_numbering_for_pp()
    args = _ckpt_args(mtp_num_layers=1)

    def make_stage(stage, *, loading=False):
        names = ["language_model.decoder.layers.0.weight"]
        if stage == 1:
            names += ["language_model.mtp.layers.0.mtp_model_layer.layers.0.weight"]
        params = {name: torch.nn.Parameter(torch.arange(3.0) + stage + i) for i, name in enumerate(names)}
        adam = torch.optim.AdamW(params.values(), lr=0.01 * (stage + 1), weight_decay=0.1 * (stage + 1))
        for param in params.values():
            param.grad = torch.full_like(param, 0.25 * (stage + 1))
        adam.step()
        group_meta = {key: value for key, value in adam.param_groups[0].items() if key != "params"}
        raw = {
            "model": {name: param.detach() for name, param in params.items()},
            "optimizer": {
                "state": {f"module.{name}": adam.state[param] for name, param in params.items()},
                "param_to_group_meta": {f"module.{name}": group_meta for name in params},
            },
        }
        if loading:
            for param in params.values():
                param.data.zero_()
                for value in adam.state[param].values():
                    value.zero_()
            group_meta.update(lr=0.9, weight_decay=0.9)
        snapshots = {"model": raw["model"].copy(), "optimizer": raw["optimizer"].copy()}
        processed = checkpointing.preprocess_fsdp_dtensor_state_dict(
            args, raw, SimpleNamespace(config=SimpleNamespace(offset=stage * 26))
        )
        return params, adam, snapshots, processed

    saved = [make_stage(stage) for stage in range(2)]
    combined = {"model": {}, "optimizer": {"state": {}, "param_to_group_meta": {}}}
    for _, _, _, processed in saved:
        for target, source in [
            (combined["model"], processed["model"]),
            *[(combined["optimizer"][key], value) for key, value in processed["optimizer"].items()],
        ]:
            assert target.keys().isdisjoint(source), "PP stages must not overwrite each other's state"
            target.update(source)
    dcp.save(combined, checkpoint_id=tmp_path)

    for stage, (expected_params, expected_adam, expected_raw, _) in enumerate(saved):
        params, adam, raw, processed = make_stage(stage, loading=True)
        dcp.load(processed, checkpoint_id=tmp_path)
        # MCore restores the pre-preprocessing snapshots before model/optimizer load.
        processed.update(raw)
        assert processed["model"].keys() == expected_raw["model"].keys()
        for name, param in params.items():
            torch.testing.assert_close(param, expected_params[name], rtol=0, atol=0)
            state_key = f"module.{name}"
            for field, value in processed["optimizer"]["state"][state_key].items():
                torch.testing.assert_close(value, expected_adam.state[expected_params[name]][field], rtol=0, atol=0)
            metadata = processed["optimizer"]["param_to_group_meta"][state_key]
            assert metadata == expected_raw["optimizer"]["param_to_group_meta"][state_key]
        adam.param_groups[0].update(metadata)
        # Loaded moments, step, LR and weight decay must reproduce the next update.
        for optimizer, parameters in ((adam, params), (expected_adam, expected_params)):
            for param in parameters.values():
                param.grad = torch.full_like(param, 0.75)
            optimizer.step()
        for name, param in params.items():
            torch.testing.assert_close(param, expected_params[name], rtol=0, atol=0)

    # Existing non-MTP checkpoint metadata keeps its on-disk naming convention.
    args.mtp_num_layers = None
    _, _, raw, processed = make_stage(1)
    assert processed["optimizer"]["param_to_group_meta"] is raw["optimizer"]["param_to_group_meta"]
