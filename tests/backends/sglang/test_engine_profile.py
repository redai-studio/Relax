# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine profiles: one ``SGLangEngine`` serves the policy, GenRM judges and
OPD teachers, and only the policy's weights may change."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


try:
    import relax.backends.sglang.sglang_engine as m

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing sglang dependencies")

STATIC_PROFILES = ("genrm", "teacher")

WEIGHT_UPDATES = {
    "update_weights_from_tensor": lambda engine: engine.update_weights_from_tensor(["blob"]),
    "update_weights_from_distributed": lambda engine: engine.update_weights_from_distributed(
        ["w"], ["float32"], [[1]], "g"
    ),
    "init_weights_update_group": lambda engine: engine.init_weights_update_group("192.0.2.1", 1, 0, 2, "g", "nccl"),
    "load_lora_adapter_from_tensors": lambda engine: engine.load_lora_adapter_from_tensors("lora", "blob", {}),
    "update_lora_from_distributed": lambda engine: engine.update_lora_from_distributed(
        "lora", ["w"], ["float32"], [[1]], {}, "g"
    ),
}


def _make_engine(monkeypatch, profile: str):
    """A started-looking engine whose HTTP requests are recorded, not sent."""
    engine = m.SGLangEngine(SimpleNamespace(fully_async=True, rollout_external=False), rank=0, profile=profile)
    engine.node_rank = 0
    engine.server_host = "192.0.2.1"
    engine.server_port = 16000
    requests_made: list[str] = []
    monkeypatch.setattr(
        type(engine),
        "_make_request",
        lambda self, endpoint, payload=None, timeout=None: requests_made.append(endpoint),
    )
    return engine, requests_made


@pytest.mark.parametrize("profile", STATIC_PROFILES)
@pytest.mark.parametrize("operation", sorted(WEIGHT_UPDATES))
def test_engine_profile_static_rejects_weight_update(monkeypatch, profile, operation):
    engine, requests_made = _make_engine(monkeypatch, profile)

    with pytest.raises(RuntimeError, match=f"{operation} is not available on a '{profile}' engine.*static model"):
        WEIGHT_UPDATES[operation](engine)

    # Rejected, not forwarded and not silently ignored.
    assert requests_made == []


@pytest.mark.parametrize("profile", STATIC_PROFILES)
def test_engine_profile_static_rejects_dcs_registration(monkeypatch, profile):
    engine, _ = _make_engine(monkeypatch, profile)
    monkeypatch.setattr(m, "create_client", lambda **kwargs: pytest.fail("a static engine must not reach DCS"))

    with pytest.raises(RuntimeError, match="register_dcs is not available"):
        engine.register_dcs()


@pytest.mark.parametrize("operation", sorted(WEIGHT_UPDATES))
def test_engine_profile_policy_allows_weight_update(monkeypatch, operation):
    engine, requests_made = _make_engine(monkeypatch, "policy")

    WEIGHT_UPDATES[operation](engine)

    assert requests_made == [operation]


def test_engine_profile_defaults_to_policy():
    engine = m.SGLangEngine(SimpleNamespace(), rank=0)
    # Shells built without ``__init__`` (as several tests do) are policy engines too.
    shell = m.SGLangEngine.__new__(m.SGLangEngine)

    assert engine.profile is m.POLICY_PROFILE
    assert shell.profile is m.POLICY_PROFILE


def test_engine_profile_unknown_name_is_rejected():
    with pytest.raises(
        ValueError, match=r"Unknown engine profile 'judge'; available: \['genrm', 'policy', 'teacher'\]"
    ):
        m.SGLangEngine(SimpleNamespace(), rank=0, profile="judge")


def _record_init(monkeypatch, engine) -> dict:
    """Replace everything ``init`` would launch with recorders."""
    calls: dict = {"builder": None, "init_normal": None, "dcs": 0}

    def record_builder(name):
        def builder(*args, **kwargs):
            calls["builder"] = name
            return {"node_rank": 0, "host": "192.0.2.1", "port": 16000}, []

        return builder

    engine_cls = type(engine)
    monkeypatch.setattr(m, "_compute_server_args", record_builder("policy"))
    monkeypatch.setattr(m, "_compute_genrm_server_args", record_builder("genrm"))
    monkeypatch.setattr(m, "get_host_info", lambda: ("node", "192.0.2.1"))
    monkeypatch.setattr(
        engine_cls, "_init_normal", lambda self, server_args, **kw: calls.__setitem__("init_normal", kw)
    )
    monkeypatch.setattr(engine_cls, "register_dcs", lambda self: calls.__setitem__("dcs", calls["dcs"] + 1))
    return calls


def _init(engine, **kwargs):
    engine.init(dist_init_addr="192.0.2.1:17000", port=16000, nccl_port=16001, **kwargs)


def test_engine_profile_teacher_never_joins_weight_sync_or_the_router(monkeypatch):
    args = SimpleNamespace(
        fully_async=True, rollout_external=False, sglang_router_ip="192.0.2.9", sglang_router_port=3000
    )
    engine = m.SGLangEngine(args, rank=0, profile="teacher")
    calls = _record_init(monkeypatch, engine)

    # Even if the caller forgets the skip flags the teacher manager passes today.
    _init(engine)

    assert calls["dcs"] == 0
    assert (engine.router_ip, engine.router_port, engine._skip_router_registration) == ("", 0, True)
    # Otherwise it launches exactly like a policy engine.
    assert calls["builder"] == "policy"
    assert calls["init_normal"] == {}


def test_engine_profile_policy_init_is_unchanged(monkeypatch):
    args = SimpleNamespace(
        fully_async=True, rollout_external=False, sglang_router_ip="192.0.2.9", sglang_router_port=3000
    )
    engine = m.SGLangEngine(args, rank=0)
    calls = _record_init(monkeypatch, engine)

    _init(engine)
    assert (engine.router_ip, engine.router_port, engine._skip_router_registration) == ("192.0.2.9", 3000, False)
    assert (calls["builder"], calls["init_normal"], calls["dcs"]) == ("policy", {}, 1)

    _init(engine, router_ip="192.0.2.8", router_port=3001, skip_dcs_registration=True, skip_router_registration=True)
    assert (engine.router_ip, engine.router_port, engine._skip_router_registration) == ("192.0.2.8", 3001, True)
    assert calls["dcs"] == 1


def test_engine_profile_only_genrm_drains_before_release(monkeypatch):
    drained: list[str] = []
    monkeypatch.setattr(
        m.SGLangEngine, "_drain_and_release_memory_occupation", lambda self: drained.append(self.profile.name)
    )
    monkeypatch.setattr(m.SGLangEngine, "flush_cache", lambda self: None)

    for profile in ("policy", "teacher", "genrm"):
        engine, _ = _make_engine(monkeypatch, profile)
        engine.release_memory_occupation()

    assert drained == ["genrm"]


def test_engine_profile_names_used_by_managers_exist():
    # GenRMManager and TeacherManager pass these names as plain strings.
    assert m.resolve_engine_profile("genrm") is m.GENRM_PROFILE
    assert m.resolve_engine_profile("teacher") is m.TEACHER_PROFILE
