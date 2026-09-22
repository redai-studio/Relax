# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for baseline node-group pinning."""

import os
from argparse import Namespace
from pathlib import Path

import pytest

from relax.core.node_group_affinity import (
    _get_elastic_node_group,
    maybe_pin_baseline_to_stable,
    require_control_plane_resource,
    require_control_plane_resource_on_node,
    with_control_plane_affinity,
)


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("RELAX_INITIAL_NODE_GROUP", raising=False)
    return monkeypatch


def _write_yaml(tmp_path, enabled: bool):
    """Write a minimal autoscaler config."""
    path = tmp_path / "autoscaler.yaml"
    path.write_text(f"enabled: {'true' if enabled else 'false'}\n")
    return str(path)


def test_pins_stable_for_enabled_elastic_job(clean_env, tmp_path):
    """autoscaler_config present + YAML enabled: true -> env set to stable."""
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    maybe_pin_baseline_to_stable(args)

    assert os.environ.get("RELAX_INITIAL_NODE_GROUP") == "stable"


def test_disabled_autoscaler_does_not_set_env(clean_env, tmp_path):
    """YAML enabled: false -> not an active elastic job -> env stays unset (so
    a disabled-autoscaler run never risks hanging on missing markers)."""
    cfg = _write_yaml(tmp_path, enabled=False)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    maybe_pin_baseline_to_stable(args)

    assert not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()


def test_enable_affinity_false_no_longer_affects_env(clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=False)
    maybe_pin_baseline_to_stable(args)

    assert os.environ.get("RELAX_INITIAL_NODE_GROUP") == "stable"


def test_non_elastic_job_does_not_set_env(clean_env):
    args = Namespace(autoscaler_config=None, enable_affinity=True)
    maybe_pin_baseline_to_stable(args)

    assert not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()


def test_unreadable_config_does_not_set_env(clean_env, tmp_path):
    """A missing / unreadable config is treated as "not enabled": we log a
    warning and leave the env unset rather than crash startup."""
    args = Namespace(autoscaler_config=str(tmp_path / "does_not_exist.yaml"), enable_affinity=True)
    maybe_pin_baseline_to_stable(args)

    assert not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()


def test_explicit_env_value_is_preserved(clean_env, tmp_path):
    clean_env.setenv("RELAX_INITIAL_NODE_GROUP", "custom")
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    maybe_pin_baseline_to_stable(args)

    assert os.environ.get("RELAX_INITIAL_NODE_GROUP") == "custom"


def _affinity_parser():
    import argparse

    pytest.importorskip("sglang.srt.server_args")

    try:
        from relax.utils.arguments import get_slime_extra_args_provider
    except (ImportError, AssertionError) as exc:
        pytest.skip(f"relax.utils.arguments unavailable: {exc}")

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    get_slime_extra_args_provider()(parser)
    return parser


def test_enable_affinity_defaults_true():
    parser = _affinity_parser()
    args, _ = parser.parse_known_args([])
    assert args.enable_affinity is True


def test_enable_affinity_flag_sets_true():
    parser = _affinity_parser()
    args, _ = parser.parse_known_args(["--enable-affinity"])
    assert args.enable_affinity is True


def test_no_enable_affinity_flag_sets_false():
    """BooleanOptionalAction registers the paired ``--no-enable-affinity``
    form, now consumed by create_placement_group (node_group_affinity)."""
    parser = _affinity_parser()
    args, _ = parser.parse_known_args(["--no-enable-affinity"])
    assert args.enable_affinity is False


def test_control_plane_affinity_gate_matrix(clean_env, tmp_path):
    enabled = _write_yaml(tmp_path, enabled=True)
    disabled_path = tmp_path / "autoscaler-disabled.yaml"
    disabled_path.write_text("enabled: false\n")
    disabled = str(disabled_path)

    assert _get_elastic_node_group(Namespace(autoscaler_config=None, enable_affinity=True)) is None
    assert _get_elastic_node_group(Namespace(autoscaler_config=disabled, enable_affinity=True)) is None
    assert _get_elastic_node_group(Namespace(autoscaler_config=enabled, enable_affinity=False)) is None
    assert _get_elastic_node_group(Namespace(autoscaler_config=enabled, enable_affinity=True)) == "stable"


def test_control_plane_affinity_ignores_manual_env_without_autoscaler(clean_env):
    clean_env.setenv("RELAX_INITIAL_NODE_GROUP", "stable")
    options = with_control_plane_affinity(
        Namespace(autoscaler_config=None, enable_affinity=True),
        {"num_cpus": 1},
    )
    assert options == {"num_cpus": 1}


def test_control_plane_affinity_merges_without_mutating_input(clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    clean_env.setenv("RELAX_INITIAL_NODE_GROUP", "custom")
    original = {"num_cpus": 1, "runtime_env": {"env_vars": {"A": "B"}}, "resources": {"other": 2}}

    options = with_control_plane_affinity(
        Namespace(autoscaler_config=cfg, enable_affinity=True),
        original,
    )

    assert options == {
        "num_cpus": 1,
        "runtime_env": {"env_vars": {"A": "B"}},
        "resources": {"other": 2, "custom_cpu": 1},
    }
    assert original["resources"] == {"other": 2}


def test_control_plane_hard_node_affinity_fails_on_missing_marker(monkeypatch, clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    monkeypatch.setattr(
        "ray.nodes",
        lambda: [{"NodeID": "node-1", "Alive": True, "Resources": {"CPU": 8}}],
    )

    with pytest.raises(RuntimeError, match="stable_cpu"):
        require_control_plane_resource_on_node(args, "node-1")


def test_control_plane_hard_node_affinity_accepts_matching_marker(monkeypatch, clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    monkeypatch.setattr(
        "ray.nodes",
        lambda: [{"NodeID": "node-1", "Alive": True, "Resources": {"stable_cpu": 8}}],
    )

    require_control_plane_resource_on_node(args, "node-1")


def test_maybe_pin_caches_resolved_control_plane_group(clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)

    maybe_pin_baseline_to_stable(args)
    Path(cfg).write_text("enabled: false\n")

    assert _get_elastic_node_group(args) == "stable"


def test_control_plane_resource_missing_fails_fast(monkeypatch, clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    monkeypatch.setattr("ray.cluster_resources", lambda: {"CPU": 8})

    with pytest.raises(RuntimeError, match="stable_cpu"):
        require_control_plane_resource(args, retries=1, retry_delay=0)


def test_control_plane_resource_retries_during_worker_registration(monkeypatch, clean_env, tmp_path):
    cfg = _write_yaml(tmp_path, enabled=True)
    args = Namespace(autoscaler_config=cfg, enable_affinity=True)
    resources = iter([{"CPU": 8}, {"CPU": 8, "stable_cpu": 1}])
    monkeypatch.setattr("ray.cluster_resources", lambda: next(resources))
    monkeypatch.setattr("relax.core.node_group_affinity.time.sleep", lambda _: None)

    require_control_plane_resource(args, retries=2, retry_delay=0)


def test_control_plane_resource_check_is_noop_for_regular_job(monkeypatch, clean_env):
    args = Namespace(autoscaler_config=None, enable_affinity=True)
    monkeypatch.setattr("ray.cluster_resources", lambda: pytest.fail("must not query Ray"))

    require_control_plane_resource(args)
