# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Manager lookup contracts retained after legacy engine selection was
removed."""

import pytest


try:
    import relax.components.genrm as genrm_module

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray[serve] + relax deps")


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
