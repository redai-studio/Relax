# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for placement-group node affinity."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from relax.core.service import Service, _require_node_group_markers, create_placement_group


def test_missing_markers_raise_after_retries():
    with (
        patch("relax.distributed.ray.placement_group.ray.cluster_resources", return_value={"GPU": 8.0, "CPU": 64.0}),
        patch("relax.distributed.ray.placement_group.time.sleep") as mock_sleep,
    ):
        with pytest.raises(RuntimeError) as exc:
            _require_node_group_markers("stable", retries=3, retry_delay=0.01)

    msg = str(exc.value)
    assert "stable_gpu" in msg
    assert "stable_cpu" in msg
    assert "--no-enable-affinity" in msg
    # 3 attempts -> 2 sleeps between them.
    assert mock_sleep.call_count == 2


def test_markers_appear_on_retry():
    """A stable node that registers its markers late must be tolerated."""
    calls = [
        {"GPU": 8.0, "CPU": 64.0},  # first probe: not yet registered
        {"GPU": 8.0, "CPU": 64.0, "stable_gpu": 8.0, "stable_cpu": 8.0},  # then present
    ]
    with (
        patch("relax.distributed.ray.placement_group.ray.cluster_resources", side_effect=calls),
        patch("relax.distributed.ray.placement_group.time.sleep"),
    ):
        _require_node_group_markers("stable", retries=3, retry_delay=0.01)  # no raise


def _run_create_placement_group(num_gpus=2, node_group_affinity=True, cluster_resources=None):
    """Create a placement group against a mocked Ray runtime."""
    captured = {}

    def _fake_placement_group(bundles, strategy="PACK"):
        captured["bundles"] = bundles
        captured["strategy"] = strategy
        return MagicMock(name="pg")

    def _fake_ray_get(arg):
        # Second call passes a list of get_ip_and_gpu_id futures -> return one
        # (ip, gpu_id) tuple per bundle. First call is pg.ready() (unused).
        if isinstance(arg, list):
            return [("10.0.0.1", i) for i in range(len(arg))]
        return None

    cr_mock = MagicMock(return_value=cluster_resources or {"GPU": 8.0, "CPU": 64.0})

    with (
        patch("relax.distributed.ray.placement_group.device_utils.get_ray_accelerator_name", return_value="GPU"),
        patch("relax.distributed.ray.placement_group.get_ray_accelerator_kwargs", return_value={"num_gpus": 1}),
        patch("relax.distributed.ray.placement_group.placement_group", side_effect=_fake_placement_group),
        patch("relax.distributed.ray.placement_group.PlacementGroupSchedulingStrategy", MagicMock()),
        patch("relax.distributed.ray.placement_group.InfoActor", MagicMock()),
        patch("relax.distributed.ray.placement_group.ray.get", side_effect=_fake_ray_get),
        patch("relax.distributed.ray.placement_group.ray.kill", MagicMock()),
        patch("relax.distributed.ray.placement_group.ray.cluster_resources", cr_mock),
        patch("relax.distributed.ray.placement_group.time.sleep"),
    ):
        pg, reordered_indices, reordered_gpu_ids = create_placement_group(
            num_gpus, node_group_affinity=node_group_affinity
        )
    return captured, cr_mock


def test_create_pg_stable_with_markers_adds_marker_bundles(monkeypatch):
    """env=stable + affinity=True + cluster declares markers -> every bundle
    carries the {group}_gpu/{group}_cpu markers and the PG builds without
    raising."""
    monkeypatch.setenv("RELAX_INITIAL_NODE_GROUP", "stable")
    resources = {"GPU": 8.0, "CPU": 64.0, "stable_gpu": 8.0, "stable_cpu": 8.0}
    captured, cr_mock = _run_create_placement_group(num_gpus=2, node_group_affinity=True, cluster_resources=resources)

    assert len(captured["bundles"]) == 2
    for bundle in captured["bundles"]:
        assert bundle["stable_gpu"] == 1
        assert bundle["stable_cpu"] == 1
        assert bundle["GPU"] == 1
        assert bundle["CPU"] == 1
    # Marker presence was actually probed.
    assert cr_mock.call_count >= 1


def test_create_pg_stable_missing_markers_raises(monkeypatch):
    """env=stable + affinity=True but the cluster never declares the markers ->
    RuntimeError after the retry loop (rather than a forever-hang on
    pg.ready())."""
    monkeypatch.setenv("RELAX_INITIAL_NODE_GROUP", "stable")
    with pytest.raises(RuntimeError) as exc:
        _run_create_placement_group(
            num_gpus=2,
            node_group_affinity=True,
            cluster_resources={"GPU": 8.0, "CPU": 64.0},  # no stable_* markers
        )
    assert "stable_gpu" in str(exc.value)
    assert "stable_cpu" in str(exc.value)


def test_create_pg_affinity_false_skips_marker_check(monkeypatch):
    """node_group_affinity=False (opt-out) even with env=stable -> markers are
    NOT probed and bundles carry NO marker, so a role can escape onto elastic
    nodes and a marker-less cluster is unaffected."""
    monkeypatch.setenv("RELAX_INITIAL_NODE_GROUP", "stable")
    captured, cr_mock = _run_create_placement_group(
        num_gpus=2,
        node_group_affinity=False,
        cluster_resources={"GPU": 8.0, "CPU": 64.0},  # no markers -> would raise if probed
    )

    for bundle in captured["bundles"]:
        assert "stable_gpu" not in bundle
        assert "stable_cpu" not in bundle
        assert bundle["GPU"] == 1
        assert bundle["CPU"] == 1
    # No marker probe at all when affinity is opted out.
    assert cr_mock.call_count == 0


def test_create_pg_no_env_is_plain_unconstrained(monkeypatch):
    """env unset -> ordinary PG: no marker probe, no marker bundle.

    Guarantees non-elastic clusters are completely unaffected by the affinity
    feature.
    """
    monkeypatch.delenv("RELAX_INITIAL_NODE_GROUP", raising=False)
    captured, cr_mock = _run_create_placement_group(
        num_gpus=3,
        node_group_affinity=True,  # affinity on, but env empty -> no-op
        cluster_resources={"GPU": 8.0, "CPU": 64.0},
    )

    assert len(captured["bundles"]) == 3
    for bundle in captured["bundles"]:
        assert bundle == {"GPU": 1, "CPU": 1}
    assert cr_mock.call_count == 0


def _build_service(config):
    """Construct a service without starting Ray Serve."""
    with (
        patch("relax.core.service.create_placement_group", return_value=("pg", [], [])) as mock_cpg,
        patch.object(Service, "_deploy", return_value=None),
    ):
        Service(cls=MagicMock(), role="actor", healthy=MagicMock(), config=config, num_gpus=2)
    return mock_cpg


def test_service_forwards_enable_affinity_false():
    """config.enable_affinity=False -> create_placement_group is called with
    node_group_affinity=False (the escape valve, independent of env)."""
    mock_cpg = _build_service(Namespace(enable_affinity=False))
    mock_cpg.assert_called_once_with(num_gpus=2, node_group_affinity=False)


def test_service_forwards_enable_affinity_true():
    """config.enable_affinity=True -> node_group_affinity=True."""
    mock_cpg = _build_service(Namespace(enable_affinity=True))
    mock_cpg.assert_called_once_with(num_gpus=2, node_group_affinity=True)


@pytest.mark.parametrize("has_data_source", [False, True])
def test_service_deploy_pins_both_bind_branches_for_enabled_autoscaler(tmp_path, has_data_source):
    config_path = tmp_path / "autoscaler.yaml"
    config_path.write_text("enabled: true\n")
    deployment_cls = MagicMock()
    deployment = deployment_cls.options.return_value
    service = object.__new__(Service)
    service.cls = deployment_cls
    service.config = Namespace(autoscaler_config=str(config_path), enable_affinity=True)
    service.runtime_env = {"env_vars": {"A": "B"}}
    service.healthy = MagicMock()
    service.num_gpus = 0
    service.role = "rollout" if has_data_source else "actor"
    service.data_source = MagicMock() if has_data_source else None

    with patch("relax.core.service.serve.run", return_value=MagicMock()):
        service._deploy(None)

    assert deployment_cls.options.call_args.kwargs["ray_actor_options"] == {
        "runtime_env": {"env_vars": {"A": "B"}},
        "resources": {"stable_cpu": 1},
    }
    deployment.bind.assert_called_once()


def test_service_deploy_is_unconstrained_without_autoscaler():
    deployment_cls = MagicMock()
    service = object.__new__(Service)
    service.cls = deployment_cls
    service.config = Namespace(autoscaler_config=None, enable_affinity=True)
    service.runtime_env = None
    service.healthy = MagicMock()
    service.num_gpus = 0
    service.role = "actor"
    service.data_source = None

    with patch("relax.core.service.serve.run", return_value=MagicMock()):
        service._deploy(None)

    assert deployment_cls.options.call_args.kwargs["ray_actor_options"] == {"runtime_env": None}


def test_service_deferred_deploy_keeps_placement_group_ownership():
    pgs = ("service-owned-pg", [0], [0])
    with (
        patch("relax.core.service.create_placement_group", return_value=pgs) as mock_cpg,
        patch.object(Service, "_deploy", return_value=None) as mock_deploy,
    ):
        service = Service(
            cls=MagicMock(),
            role="actor",
            healthy=MagicMock(),
            config=Namespace(enable_affinity=True),
            num_gpus=1,
            defer_deploy=True,
        )
        mock_deploy.assert_not_called()
        service.deploy()

    assert service.pgs is pgs
    mock_cpg.assert_called_once_with(num_gpus=1, node_group_affinity=True)
    mock_deploy.assert_called_once_with(pgs)

    with pytest.raises(RuntimeError, match="already been deployed"):
        service.deploy()


@pytest.mark.parametrize("role", ["sft", "actor"])
def test_sft_data_node_affinity_is_role_specific_and_preserves_resources(monkeypatch, role):
    monkeypatch.setenv("RELAX_SFT_DATA_NODE_IP", "10.0.0.1")
    service = object.__new__(Service)
    service.cls = MagicMock()
    service.config = Namespace()
    service.runtime_env = None
    service.healthy = MagicMock()
    service.num_gpus = 0
    service.role = role
    service.data_source = None
    with (
        patch("relax.core.service.with_control_plane_affinity", return_value={"resources": {"stable_cpu": 1}}),
        patch(
            "relax.core.service.ray.nodes",
            return_value=[
                {
                    "NodeID": "data",
                    "NodeManagerAddress": "10.0.0.1",
                    "Alive": True,
                    "Resources": {"node:10.0.0.1": 1, "stable_cpu": 1},
                }
            ],
        ),
        patch("relax.core.service.serve.run", return_value=MagicMock()),
    ):
        service._deploy(None)
    resources = service.cls.options.call_args.kwargs["ray_actor_options"]["resources"]
    assert resources == ({"stable_cpu": 1, "node:10.0.0.1": 0.001} if role == "sft" else {"stable_cpu": 1})


def test_placement_factory_legacy_exports_are_identical():
    from relax.distributed.ray import placement_group as placement

    assert create_placement_group is placement.create_placement_group
    assert _require_node_group_markers is placement._require_node_group_markers


def test_placement_sorts_nodes_numerically_then_gpu_and_releases_only_info_actors(monkeypatch):
    from relax.distributed.ray import placement_group as placement

    monkeypatch.delenv("RELAX_INITIAL_NODE_GROUP", raising=False)
    pg = MagicMock()
    info_actors = [MagicMock() for _ in range(4)]
    coordinates = [("192.0.2.10", 0), ("192.0.2.2", 3), ("192.0.2.2", 1), ("192.0.2.10", 2)]
    with (
        patch.object(placement.device_utils, "get_ray_accelerator_name", return_value="GPU"),
        patch.object(placement, "get_ray_accelerator_kwargs", return_value={"num_gpus": 1}),
        patch.object(placement, "placement_group", return_value=pg) as factory,
        patch.object(placement, "InfoActor") as info,
        patch.object(placement, "PlacementGroupSchedulingStrategy") as strategy,
        patch.object(placement.ray, "get", side_effect=[None, coordinates]),
        patch.object(placement.ray, "kill") as kill,
    ):
        info.options.return_value.remote.side_effect = info_actors
        result = create_placement_group(4)

    assert result == (pg, [2, 1, 0, 3], [1, 3, 0, 2])
    factory.assert_called_once_with([{"GPU": 1, "CPU": 1}] * 4, strategy="PACK")
    assert [entry.kwargs["placement_group_bundle_index"] for entry in strategy.call_args_list] == [0, 1, 2, 3]
    assert all(entry.kwargs["placement_group"] is pg for entry in strategy.call_args_list)
    assert [entry.args[0] for entry in kill.call_args_list] == info_actors


def test_service_reuses_shared_pg_without_factory_or_cleanup():
    shared_pg = (MagicMock(), [0], [0])
    with (
        patch("relax.core.service.create_placement_group") as factory,
        patch("relax.core.service.remove_placement_group") as remove,
        patch.object(Service, "_deploy"),
    ):
        service = Service(
            cls=MagicMock(),
            role="actor",
            healthy=MagicMock(),
            config=Namespace(enable_affinity=True),
            num_gpus=1,
            actor_rollout_pgs=shared_pg,
        )
        assert service._ensure_placement_group() is shared_pg
    factory.assert_not_called()
    remove.assert_not_called()


@pytest.mark.parametrize("state", ["missing", "dead", "wrong_group"])
def test_sft_data_node_rejects_unschedulable_affinity(monkeypatch, state):
    monkeypatch.setenv("RELAX_SFT_DATA_NODE_IP", "192.0.2.1")
    service = object.__new__(Service)
    service.cls = MagicMock()
    service.config = Namespace(_relax_control_plane_node_group="stable", enable_affinity=True)
    service.runtime_env = None
    service.role = "sft"
    nodes = (
        []
        if state == "missing"
        else [
            {
                "NodeID": "data",
                "NodeManagerAddress": "192.0.2.1",
                "Alive": state != "dead",
                "Resources": {"node:192.0.2.1": 1},
            }
        ]
    )
    with patch("relax.core.service.ray.nodes", return_value=nodes):
        with pytest.raises((ValueError, RuntimeError), match="not available|conflicts"):
            service._deploy(None)
    service.cls.options.assert_not_called()
