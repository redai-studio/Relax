# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mapping context isolation without importing distributed training
backends."""

import ast
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import MethodType, SimpleNamespace

import pytest


@pytest.fixture
def mapping_context(monkeypatch):
    class Mapping:
        def __init__(self):
            self.pp_group, self._tp_group, self._etp_group, self.ep_group = [object() for _ in range(4)]

        def gather_from_ep_ranks(self, weights, module, name):
            return {name: ("original", weights)}

    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.models.conversion.param_mapping",
        SimpleNamespace(MegatronParamMapping=Mapping),
    )
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_update/bridge_converter.py"
    tree = ast.parse(path.read_text())
    converter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BridgeConverter")
    converter.body = [
        node
        for node in converter.body
        if isinstance(node, ast.FunctionDef) and node.name in {"collect_all_mappings", "_disable_mapping_collectives"}
    ]
    noop = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_noop_gather_from_ep_ranks"
    )
    namespace = {"contextmanager": contextmanager, "MethodType": MethodType}
    exec(compile(ast.Module(body=[noop, converter], type_ignores=[]), str(path), "exec"), namespace)
    return Mapping, namespace["BridgeConverter"]()._disable_mapping_collectives


@pytest.mark.parametrize("method_location", ["inherited", "class", "instance"])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_mapping_collectives_restore_exact_state(mapping_context, method_location, raise_inside):
    Mapping, disable = mapping_context

    class Child(Mapping):
        pass

    def custom(self, weights, module, name):
        return {name: ("custom", weights)}

    if method_location == "class":
        Child.gather_from_ep_ranks = custom
    mapping = Child()
    if method_location == "instance":
        mapping.gather_from_ep_ranks = MethodType(custom, mapping)
    mapping.inner = Mapping()
    mapping.inner.parent = mapping  # Recursive collection must handle cycles.
    original = vars(mapping).copy()
    inner_original = vars(mapping.inner).copy()
    class_original = dict(vars(Child))
    expected = mapping.gather_from_ep_ranks("weights", None, "key")
    try:
        with disable(mapping):
            for item in (mapping, mapping.inner):
                assert (item.pp_group, item._tp_group, item._etp_group, item.ep_group) == (None,) * 4
                assert item.gather_from_ep_ranks("weights", None, "key") == {"key": "weights"}
            if raise_inside:
                raise RuntimeError("conversion failed")
    except RuntimeError as error:
        assert raise_inside and str(error) == "conversion failed"
    assert vars(mapping) == original
    assert vars(mapping.inner) == inner_original
    assert dict(vars(Child)) == class_original
    assert mapping.gather_from_ep_ranks("weights", None, "key") == expected


def test_mapping_collectives_nested_context_preserves_outer_patch(mapping_context):
    Mapping, disable = mapping_context
    mapping = Mapping()
    original = vars(mapping).copy()
    with disable(mapping):
        outer = mapping.gather_from_ep_ranks
        with disable(mapping):
            assert mapping.gather_from_ep_ranks(1, None, "key") == {"key": 1}
        assert mapping.gather_from_ep_ranks is outer
        assert mapping.ep_group is None
    assert vars(mapping) == original


def test_mapping_collectives_concurrent_instances_do_not_interfere(mapping_context):
    Mapping, disable = mapping_context
    first, second, untouched = Mapping(), Mapping(), Mapping()
    original_method = Mapping.gather_from_ep_ranks
    original_states = [vars(mapping).copy() for mapping in (first, second, untouched)]
    entered, release = Event(), Event()

    def convert_first():
        with disable(first):
            entered.set()
            assert release.wait(timeout=5)
            assert first.gather_from_ep_ranks(1, None, "key") == {"key": 1}
            assert first.ep_group is None

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(convert_first)
        try:
            assert entered.wait(timeout=5)
            with disable(second):
                assert second.gather_from_ep_ranks(2, None, "key") == {"key": 2}
                assert untouched.gather_from_ep_ranks(3, None, "key") == {"key": ("original", 3)}
            assert second.gather_from_ep_ranks(2, None, "key") == {"key": ("original", 2)}
        finally:
            release.set()
        future.result(timeout=5)
    assert Mapping.gather_from_ep_ranks is original_method
    for mapping, state in zip((first, second, untouched), original_states):
        assert vars(mapping) == state
