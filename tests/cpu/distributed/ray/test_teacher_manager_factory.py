# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace


def _install_fake_teacher_manager(monkeypatch, captured):
    from relax.distributed.ray import inference_manager

    class Handle:
        get_urls = SimpleNamespace(remote=lambda: "get_urls-ref")

    def create(args, role, model, pg, **kwargs):
        captured["remote_args"] = (args, role, model, pg)
        captured["remote_kwargs"] = kwargs
        captured["handle"] = Handle()
        return captured["handle"]

    monkeypatch.setattr(inference_manager, "create_model_manager", create)
    monkeypatch.setattr(inference_manager, "register_legacy_name", lambda name, manager: captured.update(name=name))


def test_create_managed_opd_teacher_manager_offloads_shared_pg_teacher(monkeypatch, tmp_path):
    import ray

    from relax.utils.opd.opd_utils import create_managed_opd_teacher_manager

    captured = {"calls": []}
    _install_fake_teacher_manager(monkeypatch, captured)
    monkeypatch.setattr(
        ray,
        "get",
        lambda ref: ["http://teacher/generate"] if ref == "get_urls-ref" else None,
    )

    autoscaler_config = tmp_path / "autoscaler.yaml"
    autoscaler_config.write_text("enabled: true\n")
    args = SimpleNamespace(
        offload_rollout=True,
        autoscaler_config=str(autoscaler_config),
        enable_affinity=True,
    )
    manager, urls = create_managed_opd_teacher_manager(
        args,
        num_replicas=1,
        gpus_per_replica=4,
        pg=("pg", list(range(8)), list(range(8))),
        shared_pg=True,
        bundle_offset=4,
        runtime_env={"env_vars": {"A": "B"}},
    )

    assert manager is captured["handle"]
    assert urls == ["http://teacher/generate"]
    assert captured["name"] == "relax_teacher_manager"
    assert captured["remote_args"] == (args, "teacher", "__default__", ("pg", list(range(8)), list(range(8))))
    assert captured["remote_kwargs"] == {
        "runtime_env": {"env_vars": {"A": "B"}},
        "num_replicas": 1,
        "gpus_per_replica": 4,
        "shared_pg": True,
        "bundle_offset": 4,
    }
