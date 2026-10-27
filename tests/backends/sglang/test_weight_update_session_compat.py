# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise native sessions mixed with Relax's legacy quantization endpoint."""

import ast
import sys
from contextlib import nullcontext
from types import MethodType, ModuleType, SimpleNamespace

import pytest

from tests.backends.sglang.source_utils import sglang_source_path


msgspec = pytest.importorskip("msgspec")


@pytest.fixture
def manager(monkeypatch):
    path = sglang_source_path("srt/managers/scheduler_components/weight_updater.py")
    tree = ast.parse(path.read_text())
    sessions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_WeightUpdateSession"]
    if not sessions:
        pytest.skip("Requires SGLang 0.5.21 weight update sessions")
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SchedulerWeightUpdaterManager"
    )
    names = {"begin_weight_update", "end_weight_update", "post_process_weights", "update_weights_from_tensor"}
    nodes = sessions + [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    events = []
    loader = ModuleType("sglang.srt.model_loader.loader")
    loader.post_load_weights = lambda model: events.append((model.role, "post_load"))
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    namespace = {
        "msgspec": msgspec,
        "logger": SimpleNamespace(error=lambda *_: None),
        "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=lambda **_: None)),
        **{
            f"{name}ReqOutput": SimpleNamespace
            for name in ("BeginWeightUpdate", "EndWeightUpdate", "PostProcessWeights", "UpdateWeightsFromTensor")
        },
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    runners = []
    for role in ("target", "draft"):
        model = SimpleNamespace(role=role, packed=False, fail=False)

        def restore(model=model):
            model.packed = False
            events.append((model.role, "restore"))

        def pack(model=model):
            assert not model.packed, f"{model.role} weights were packed twice"
            model.packed = True
            events.append((model.role, "pack"))

        def post(
            restore_weights_before_load=False, post_process_quantization=False, model=model, restore=restore, pack=pack
        ):
            if model.fail:
                return False, "injected failure"
            if restore_weights_before_load:
                restore()
            if post_process_quantization:
                pack()
            return True, "Success"

        def end(*, run_post_load, model=model, pack=pack):
            if run_post_load:
                loader.post_load_weights(model)
            pack()

        def load(*, named_tensors, load_format, model=model):
            model.packed = False
            events.append((model.role, "load"))
            return True, "Success"

        runner = SimpleNamespace(
            model=model,
            post_process_weights=post,
            weight_updater=SimpleNamespace(
                begin_weight_update=restore, end_weight_update=end, update_weights_from_tensor=load
            ),
        )
        runners.append((role, runner))

    obj = SimpleNamespace(
        _session=None,
        _select_runners=lambda selector="all": [
            (role, runner) for role, runner in runners if selector in ("all", role)
        ],
        tp_cpu_group=object(),
        tp_worker=SimpleNamespace(deserialize_own_rank=lambda value: value),
        draft_worker=object(),
        _observe_weight_load=lambda _: nullcontext(),
        flush_cache_after_weight_update=lambda _: None,
        record_weight_version_after_update=lambda _: None,
        events=events,
        runners=runners,
    )
    for name in names:
        setattr(obj, name, MethodType(namespace[name], obj))
    return obj


def _post(manager, *, restore=False, quant=True):
    return manager.post_process_weights(
        SimpleNamespace(restore_weights_before_load=restore, post_process_quantization=quant)
    )


def _load(manager):
    return manager.update_weights_from_tensor(
        SimpleNamespace(
            serialized_named_tensors=[],
            selector="all",
            disable_draft_model=None,
            load_format=None,
            weight_version=None,
        )
    )


@pytest.mark.parametrize("protocol", ["legacy", "native", "mixed"])
def test_weight_update_protocol_packs_each_cycle_once(manager, protocol):
    for _ in range(2):
        if protocol != "native":
            assert _post(manager, restore=True, quant=False).success
        if protocol != "legacy":
            assert manager.begin_weight_update(SimpleNamespace(selector="all")).success
        assert _load(manager).success
        if protocol != "native":
            assert _post(manager).success
        if protocol != "legacy":
            assert manager.end_weight_update(SimpleNamespace()).success
        assert manager._session is None
    assert manager.events.count(("target", "pack")) == 2
    assert manager.events.count(("draft", "pack")) == 2


@pytest.mark.parametrize("mutation", ["bucket", "restore"])
def test_new_weights_invalidate_legacy_postprocessing(manager, mutation):
    manager.begin_weight_update(SimpleNamespace(selector="all"))
    _load(manager)
    _post(manager)
    assert manager._session.post_processed
    if mutation == "bucket":
        _load(manager)
    else:
        _post(manager, restore=True, quant=False)
    assert not manager._session.post_processed
    manager.end_weight_update(SimpleNamespace())
    assert manager.events.count(("target", "pack")) == 2


@pytest.mark.parametrize("restore", [False, True])
def test_p2p_session_post_load_precedes_quantization_for_selected_runner(manager, restore):
    manager.begin_weight_update(SimpleNamespace(selector="target"))
    manager.events.clear()
    assert _post(manager, restore=restore).success
    assert manager.end_weight_update(SimpleNamespace()).success
    expected = [("target", "post_load"), ("target", "pack"), ("draft", "pack")]
    if restore:
        expected = [("target", "restore"), *expected[:2], ("draft", "restore"), expected[2]]
    assert manager.events == expected


def test_failed_postprocess_does_not_mark_session_complete(manager):
    manager.begin_weight_update(SimpleNamespace(selector="all"))
    manager.runners[0][1].model.fail = True
    assert not _post(manager).success
    assert not manager._session.post_processed
    assert not any(event == "pack" for _, event in manager.events)
