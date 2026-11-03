# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine selection in the GenRM Serve replica.

The replica reads the manager's canonical discovery snapshot on every pick,
while keeping its round-robin position when the eligible addresses are unchanged.
Scaling, offload and recovery must affect admission on the next request.

Three ways this has gone wrong, one test class each:
  - the ``itertools.cycle`` outliving the list it was built for (IndexError one
    request after a recovery),
  - the replica addressing an engine's pre-rebuild port forever,
  - an empty list being cached as if it were valid, which strands the replica
    permanently even after recovery succeeds.

These drive the real ``GenRM`` methods -- the class is unwrapped from its Serve
deployment and instantiated without ``__init__`` so no cluster is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


try:
    import relax.components.genrm as genrm_module

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray[serve] + relax deps")

# RFC 5737 TEST-NET-1, reserved for documentation and examples. Deliberately not
# 10.x/192.168.x: the gitleaks pre-commit hook rejects private-range literals so
# real cluster addresses cannot be committed by accident.
LIVE_SIX = [(f"192.0.2.{i}", 16001) for i in range(6)]
LIVE_FIVE = LIVE_SIX[:5]  # one engine retired; the manager compacts the list
REBUILT = ("192.0.2.5", 16007)  # rank 5 back after recovery, on a fresh port
REBUILT_ON_NEW_PORT = LIVE_FIVE + [REBUILT]


class _Replica:
    """A bare ``GenRM`` driving its real engine-picking methods.

    Patch ``ray.get`` in the method's actual globals: Serve may copy that
    namespace when rewriting its class-based routes. The manager publishes
    explicit snapshots, independently of how often the replica reads them.
    """

    def __init__(self, monkeypatch, hosts_ports):
        cls = genrm_module.GenRM.func_or_class
        method_globals = cls._pick_engine.__globals__
        assert method_globals.get("ray") is not None, "engine picking no longer resolves 'ray' from its globals"
        self._replica = object.__new__(cls)
        self._replica._engine_caches = {"__default__": genrm_module._EngineCacheState()}
        # Only ``get_inference_snapshot.remote()`` is ever reached; the token it
        # returns is resolved by the patched ``ray.get`` below, mirroring how a
        # real ObjectRef is consumed.
        self._replica.genrm_managers = {
            "__default__": SimpleNamespace(get_inference_snapshot=SimpleNamespace(remote=lambda: _TOKEN))
        }

        self.fetches = 0
        self.publish(hosts_ports)

        def fake_ray_get(token):
            assert token is _TOKEN, "only the discovery call should reach ray.get"
            self.fetches += 1
            return self.snapshot

        monkeypatch.setitem(method_globals, "ray", SimpleNamespace(get=fake_ray_get))

    def publish(self, hosts_ports, state="ready"):
        self.snapshot = {
            "role": "genrm",
            "topology_revision": self.fetches,
            "phase": "inference",
            "default_model": "default",
            "routes": {"default": "default"},
            "models": {
                "default": {
                    "state": state,
                    "router_url": None,
                    "engines": [
                        {
                            "engine_id": f"default/{rank}",
                            "base_url": f"http://{host}:{port}",
                            "state": "ready",
                            "direct_eligible": True,
                        }
                        for rank, (host, port) in enumerate(hosts_ports)
                    ],
                }
            },
        }

    def pick(self):
        _key, idx, host, port = self._replica._pick_engine(None)
        return idx, host, port


_TOKEN = object()


@pytest.fixture
def replica(monkeypatch):
    def _make(hosts_ports):
        return _Replica(monkeypatch, hosts_ports)

    return _make


class TestRoundRobin:
    def test_fresh_snapshots_preserve_round_robin_position(self, replica):
        r = replica(LIVE_SIX)
        assert [r.pick()[0] for _ in range(12)] == list(range(6)) * 2
        assert r.fetches == 12, "every pick must check current admission"

    def test_shrunk_list_does_not_go_out_of_range(self, replica):
        """The manager compacts over the dead engine, so a cycle built for the
        6-element list would hand back index 5 into a 5-element one."""
        r = replica(LIVE_SIX)
        for _ in range(6):
            r.pick()
        r.publish(LIVE_FIVE)
        for _ in range(20):
            idx, host, port = r.pick()  # must not IndexError
            assert 0 <= idx < len(LIVE_FIVE)
            assert (host, port) in LIVE_FIVE

    def test_refresh_picks_up_the_rebuilt_port(self, replica):
        """A rebuilt engine comes back on a different port; the replica must
        stop addressing the old one."""
        r = replica(LIVE_FIVE)
        assert REBUILT not in {r.pick()[1:] for _ in range(10)}

        r.publish(REBUILT_ON_NEW_PORT)
        assert REBUILT in {r.pick()[1:] for _ in range(12)}


class TestAdmission:
    @pytest.mark.parametrize("state", ["sleeping", "draining", "onloading", "failed", "dead"])
    def test_model_state_closes_admission_and_recovery_is_immediate(self, replica, state):
        r = replica(LIVE_SIX)
        assert r.pick()[1:] in LIVE_SIX
        # Even a snapshot retaining ready engine records cannot bypass the
        # model-level lifecycle gate.
        r.publish(LIVE_SIX, state=state)
        with pytest.raises(genrm_module.HTTPException) as exc:
            r.pick()
        assert exc.value.status_code == 503
        r.publish(LIVE_SIX)
        assert r.pick()[1:] in LIVE_SIX
        assert r.fetches == 3

    @pytest.mark.parametrize("state", ["starting", "draining", "dead"])
    def test_unready_engine_is_excluded_immediately(self, replica, state):
        r = replica(LIVE_SIX)
        assert r.pick()[1:] == LIVE_SIX[0]
        r.snapshot["models"]["default"]["engines"][0]["state"] = state
        assert {r.pick()[1:] for _ in range(10)} == set(LIVE_SIX[1:])

    def test_non_ingress_worker_is_never_selected(self, replica):
        r = replica(LIVE_SIX)
        r.snapshot["models"]["default"]["engines"][0]["direct_eligible"] = False
        assert {r.pick()[1:] for _ in range(10)} == set(LIVE_SIX[1:])


class TestEmptyEngineList:
    """The manager reports ``[]`` for the whole window between offload()
    retiring every dead engine and the next onload() rebuilding them."""

    def test_no_engines_raises(self, replica):
        r = replica([])
        with pytest.raises(genrm_module.HTTPException) as exc:
            r.pick()
        assert exc.value.status_code == 503

    def test_empty_list_is_not_cached_forever(self, replica):
        """Regression: caching ``[]`` stranded the replica permanently.

        ``_pick_engine`` raises before any HTTP request is made, so the retry
        loop in ``_call_engine`` -- the only caller of
        ``_invalidate_engine_cache`` -- is never reached. Nothing would ever
        re-read the list, so GenRM stayed dead for the rest of training even
        after recovery rebuilt every engine.
        """
        r = replica([])
        with pytest.raises(genrm_module.HTTPException) as exc:
            r.pick()
        assert exc.value.status_code == 503

        # Engines are back. No invalidate() call here on purpose: nothing in the
        # request path can issue one from this state.
        r.publish(LIVE_SIX)
        assert r.pick()[1:] in LIVE_SIX
        assert r.fetches == 2

    def test_empty_list_keeps_checking_for_recovery(self, replica):
        r = replica([])
        for _ in range(50):
            with pytest.raises(genrm_module.HTTPException) as exc:
                r.pick()
            assert exc.value.status_code == 503
        assert r.fetches == 50
        r.publish(LIVE_SIX)
        assert r.pick()[1:] in LIVE_SIX


class TestManagerLookup:
    def test_selects_manager_by_route_key(self):
        cls = genrm_module.GenRM.func_or_class
        replica = object.__new__(cls)
        quality_manager = object()
        safety_manager = object()
        replica.genrm_managers = {"quality": quality_manager, "safety": safety_manager}

        assert replica.get_genrm_manager("quality") is quality_manager
        assert replica.get_genrm_manager("safety") is safety_manager

    def test_omitted_route_key_requires_exactly_one_instance(self):
        cls = genrm_module.GenRM.func_or_class
        replica = object.__new__(cls)
        manager = object()
        replica.genrm_managers = {"quality": manager}

        assert replica.get_genrm_manager() is manager

        replica.genrm_managers["safety"] = object()
        with pytest.raises(RuntimeError, match="route_key"):
            replica.get_genrm_manager()
