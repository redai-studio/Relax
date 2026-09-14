# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for snapshot identity and the real actor log-probability branch.

Load this small helper directly and execute the actor branch with a fake model
so these tests do not import Megatron/Ray or initialize distributed GPU
services.
"""

import ast
import runpy
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_HELPER = runpy.run_path(str(_ROOT / "relax/backends/megatron/ref_logprob_reuse.py"))
WeightSnapshotVersions = _HELPER["WeightSnapshotVersions"]
can_reuse_ref_log_probs = _HELPER["can_reuse_ref_log_probs"]


def _args(**overrides: object) -> Namespace:
    args = Namespace(
        keep_old_actor=False,
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        custom_megatron_before_log_prob_hook_path=None,
        custom_megatron_before_train_step_hook_path=None,
        use_opd=False,
        opd_log_prob_top_k=0,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=32768,
        log_probs_max_tokens_per_gpu=32768,
    )
    vars(args).update(overrides)
    return args


def _same_versions() -> WeightSnapshotVersions:
    versions = WeightSnapshotVersions()
    versions.mark_modified()
    versions.backup("actor")
    versions.backup("ref")
    return versions


def test_unknown_snapshots_never_compare_equal() -> None:
    versions = WeightSnapshotVersions()
    assert not versions.same_snapshot("actor", "ref")
    assert not versions.current_matches("ref")
    versions.backup("actor")
    versions.copy(src_tag="actor", dst_tag="ref")
    assert not versions.same_snapshot("actor", "ref")


def test_independent_checkpoint_loads_do_not_share_identity() -> None:
    versions = WeightSnapshotVersions()
    versions.mark_modified()
    versions.backup("actor")
    versions.mark_modified()  # Independently loaded ref, even if path is identical.
    versions.backup("ref")
    versions.restore("actor")
    assert versions.current_matches("actor")
    assert not versions.current_matches("ref")
    assert not can_reuse_ref_log_probs(_args(), versions)


def test_actor_update_invalidates_ref_until_ref_refresh() -> None:
    versions = _same_versions()
    assert can_reuse_ref_log_probs(_args(), versions)
    versions.mark_modified()
    versions.backup("actor")
    assert not can_reuse_ref_log_probs(_args(), versions)
    versions.backup("ref")
    assert can_reuse_ref_log_probs(_args(), versions)
    versions.mark_modified()  # Loading a different checkpoint invalidates equality.
    versions.backup("ref")
    assert not can_reuse_ref_log_probs(_args(), versions)


def test_old_actor_queue_uses_snapshot_identity_not_current_actor_tag() -> None:
    versions = _same_versions()
    versions.backup("rollout_actor")
    versions.mark_modified()
    versions.backup("actor")
    versions.backup("ref")
    versions.copy(src_tag="rollout_actor", dst_tag="old_actor")
    assert not can_reuse_ref_log_probs(_args(keep_old_actor=True), versions)
    versions.backup("rollout_actor")
    versions.copy(src_tag="rollout_actor", dst_tag="old_actor")
    assert can_reuse_ref_log_probs(_args(keep_old_actor=True), versions)
    versions.restore("old_actor")
    assert versions.current_matches("ref")


def test_unknown_restore_or_copy_clears_prior_identity() -> None:
    versions = _same_versions()
    versions.restore("missing")
    versions.backup("ref")
    assert not versions.same_snapshot("actor", "ref")
    versions.restore("actor")
    versions.backup("ref")
    versions.copy(src_tag="missing", dst_tag="ref")
    assert not versions.same_snapshot("actor", "ref")


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_rollout_logprobs": True},
        {"use_routing_replay": True},
        {"use_rollout_routing_replay": True},
        {"custom_megatron_before_log_prob_hook_path": "custom.hook"},
        {"custom_megatron_before_train_step_hook_path": "custom.hook"},
        {"use_opd": True},
        {"log_probs_max_tokens_per_gpu": 65536},
    ],
)
def test_forward_semantics_requiring_separate_pass_fall_back(overrides: dict) -> None:
    assert not can_reuse_ref_log_probs(_args(**overrides), _same_versions())


def test_mismatch_metrics_can_reuse_real_actor_forward_with_rollout_logprobs() -> None:
    args = _args(use_rollout_logprobs=True, get_mismatch_metrics=True)
    assert can_reuse_ref_log_probs(args, _same_versions())


def _run_actor_logprob_branch(actor: object, rollout_data: dict) -> None:
    tree = ast.parse((_ROOT / "relax/backends/megatron/actor.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_actor")
    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "should_compute_old_log_probs"
    )
    code = compile(ast.Module(body=[branch], type_ignores=[]), "actor_logprob_branch", "exec")
    namespace = {
        "self": actor,
        "rollout_data": rollout_data,
        "should_compute_old_log_probs": True,
        "data_iterator": object(),
        "data_iterator_logprobs": object(),
        "num_microbatches": [2],
        "num_microbatches_logprobs": [2],
        "can_reuse_ref_log_probs": can_reuse_ref_log_probs,
        "rollout_id": 1,
        "is_megatron_main_rank": lambda: True,
        "logger": SimpleNamespace(info=lambda message: None),
    }
    exec(code, namespace)


class _FakeActor:
    def __init__(self, versions: WeightSnapshotVersions, args: Namespace | None = None) -> None:
        self.args = args or _args()
        self._weight_snapshot_versions = versions
        self.weights_backuper = SimpleNamespace(backup_tags=["actor", "ref", "old_actor"])
        self._active_model_tag = "actor"
        self.log_probs = {"actor": [-0.3, -0.8], "ref": [-0.3, -0.8], "old_actor": [-0.3, -0.8]}
        self.calls: list[str] = []

    def _switch_model(self, tag: str) -> None:
        self._active_model_tag = tag
        self._weight_snapshot_versions.restore(tag)

    def compute_log_prob(self, *_: object, store_prefix: str, **kwargs: object) -> dict:
        self.calls.append(self._active_model_tag)
        return {store_prefix + "log_probs": list(self.log_probs[self._active_model_tag])}


def test_actor_branch_skips_identical_ref_forward_and_preserves_reference_values() -> None:
    actor = _FakeActor(_same_versions())
    rollout_data = {}
    _run_actor_logprob_branch(actor, rollout_data)
    assert actor.calls == ["actor"]
    assert rollout_data["ref_log_probs"] == actor.log_probs["ref"]
    assert rollout_data["ref_log_probs"] == rollout_data["log_probs"]
    assert rollout_data["ref_log_probs"] is not rollout_data["log_probs"]
    assert actor._active_model_tag == "actor"


def test_actor_branch_evaluates_different_reference_checkpoint() -> None:
    versions = _same_versions()
    versions.mark_modified()
    versions.backup("ref")
    actor = _FakeActor(versions)
    actor.log_probs["ref"] = [-1.2, -2.4]
    rollout_data = {}
    _run_actor_logprob_branch(actor, rollout_data)
    assert actor.calls == ["ref", "actor"]
    assert rollout_data["ref_log_probs"] == [-1.2, -2.4]
    assert rollout_data["ref_log_probs"] != rollout_data["log_probs"]


def test_actor_branch_reuses_matching_old_actor_then_restores_current_actor() -> None:
    versions = _same_versions()
    versions.backup("old_actor")
    versions.mark_modified()
    versions.backup("actor")
    actor = _FakeActor(versions, _args(keep_old_actor=True))
    actor.log_probs["actor"] = [-1.0, -2.0]
    rollout_data = {}
    _run_actor_logprob_branch(actor, rollout_data)
    assert actor.calls == ["old_actor"]
    assert rollout_data["ref_log_probs"] == [-0.3, -0.8]
    assert actor._active_model_tag == "actor"
    assert versions.current_matches("actor")


def test_actor_branch_never_substitutes_rollout_engine_probabilities() -> None:
    actor = _FakeActor(_same_versions(), _args(use_rollout_logprobs=True))
    rollout_data = {"rollout_log_probs": [-7.0, -9.0]}
    _run_actor_logprob_branch(actor, rollout_data)
    assert actor.calls == ["ref"]
    assert rollout_data["ref_log_probs"] == [-0.3, -0.8]
    assert "log_probs" not in rollout_data


@pytest.mark.parametrize("matching_ref", [False, True, None])
def test_zero_advantage_filter_runs_before_building_model_iterators(matching_ref: bool | None) -> None:
    tree = ast.parse((_ROOT / "relax/backends/megatron/actor.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_actor")
    iterator_index = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "get_data_iterator"
    )
    code = compile(ast.Module(body=method.body[: iterator_index + 1], type_ignores=[]), "actor_pruning", "exec")
    versions = _same_versions()
    if not matching_ref:
        versions.mark_modified()
        versions.backup("ref")
    actor = _FakeActor(versions, _args(advantage_estimator="grpo"))
    if matching_ref is None:
        del actor._weight_snapshot_versions  # MTP-only/SFT can skip weight management.
    actor.model = object()
    actor._rank = 0
    original = {"total_lengths": [10, 20]}
    filtered = {"total_lengths": [10]}
    calls = []

    def filter_rows(args: Namespace, data: dict, *, reference_matches_actor: bool) -> tuple[dict, int]:
        assert data is original
        assert reference_matches_actor is bool(matching_ref)
        calls.append("filter")
        return filtered, 1

    def get_iterator(args: Namespace, model: object, data: dict) -> tuple[object, list[int]]:
        assert data is filtered
        calls.append("iterator")
        return object(), [1]

    namespace = {
        "self": actor,
        "rollout_data": original,
        "filter_zero_advantage_rows": filter_rows,
        "get_data_iterator": get_iterator,
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(code, namespace)
    assert calls == ["filter", "iterator"]
    assert namespace["rollout_data"] is filtered
