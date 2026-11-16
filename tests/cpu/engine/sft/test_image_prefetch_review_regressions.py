# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only regressions for processor dispatch and prefetch timing
ownership."""

import ast
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[4]
_MARKER = "_sft_hf_image_rebuild_supported"


def load_definition(path, name, **namespace):
    source = _ROOT / path
    node = next(n for n in ast.parse(source.read_text()).body if getattr(n, "name", None) == name)
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], node], type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "processor,supported",
    [
        (SimpleNamespace(media_processor=object()), False),
        (SimpleNamespace(image_processor=object()), True),
        (SimpleNamespace(image_processor=object(), media_processor=object()), False),
        (SimpleNamespace(), False),
    ],
)
def test_worker_capability_controls_hf_reference_transport(processor, supported):
    pixels, grid = object(), object()
    worker = load_definition(
        "relax/utils/data/processor_pool.py",
        "process_sample_from_paths_in_worker",
        _load_media_for_worker=lambda paths: paths,
        prepare_mm_inputs_for_ipc=lambda media: media,
        process_sample_in_worker=lambda *args: ([1], {"pixel_values": pixels, "image_grid_thw": grid}),
        get_worker_processor=lambda: processor,
        SFT_HF_IMAGE_REBUILD_SUPPORTED=_MARKER,
    )
    _, inputs = worker("prompt", {"image": ["/shared/a.jpg"]}, {})
    assert inputs[_MARKER] is supported
    split = Mock(return_value=({"image_grid_thw": grid}, {"kind": "hf"}))
    transport = load_definition(
        "relax/engine/sft/dataset/image_transport.py",
        "ImageReferenceTransport",
        SFT_HF_IMAGE_REBUILD_SUPPORTED=_MARKER,
        KIMI_K3_SFT_REQUEST="k3",
        KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND="k3-kind",
        HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND="hf",
        split_pixel_values_from_mm_inputs=split,
    )(True)
    rendered = SimpleNamespace(
        image_fallback=False, image_refs=["/shared/a.jpg"], processor_kwargs=None, rendered_text="prompt", idx=0
    )
    retained, descriptor = transport.split(inputs, rendered, "test")
    assert _MARKER not in retained
    if supported:
        assert descriptor == {"kind": "hf"}
        split.assert_called_once()
    else:
        assert retained["pixel_values"] is pixels
        assert descriptor is None
        split.assert_not_called()


def test_background_fetch_defers_pixels_and_timer_writes_until_microbatch(monkeypatch):
    main_thread = threading.get_ident()
    writes = []

    def add(name, elapsed):
        assert threading.get_ident() == main_thread
        writes.append((name, elapsed))

    timer = lambda: SimpleNamespace(add=add)
    features = {"pixel_values": SimpleNamespace(numel=lambda: 6, element_size=lambda: 2)}
    pending = []

    def submit(*args):
        pending.append(args)
        future = Future()
        future.set_result((features, {"read_s": 0.25, "process_s": 0.5}))
        return future

    pool = SimpleNamespace(executor=SimpleNamespace(submit=submit))
    builder = load_definition(
        "relax/utils/data/image_rebuild.py",
        "build_batch_image_features",
        build_image_features_in_worker=object(),
        Timer=timer,
    )
    module = ModuleType("relax.utils.data.image_rebuild")
    module.build_batch_image_features = builder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    lazy_class = load_definition(
        "relax/engine/sft/image_prefetch.py",
        "LazySFTImageInput",
        Timer=timer,
        current_thread=threading.current_thread,
        main_thread=threading.main_thread,
    )
    resolve_microbatch = load_definition(
        "relax/engine/sft/image_prefetch.py", "resolve_sft_image_micro_batch", LazySFTImageInput=lazy_class
    )
    helper_class = load_definition(
        "relax/engine/sft/image_prefetch.py",
        "SFTImagePrefetch",
        Timer=timer,
        SFT_IMAGE_REFS_FIELD="refs",
        extract_image_ref_descriptors=lambda raw: raw["refs"],
        LazySFTImageInput=lazy_class,
    )
    helper = helper_class(SimpleNamespace(), lambda **kw: ([{"refs": [{}, None, {}]}, None], 0.1))
    helper.owns_vision = lambda model: True
    helper._ensure_pool = lambda: pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(
            helper.fetch,
            3,
            model=None,
            data_fields=[],
            batch_size=3,
            partition_id="sft_3",
            task_name="train",
            sampling_config={},
        ).result()
    assert pending == []
    assert writes == []
    batch = {"refs": [{}, None, {}], "multimodal_train_inputs": [{}, None, {}]}
    helper.resolve(batch, 3, model=None)
    assert writes == []
    resolve_microbatch(batch)
    assert writes == [("sft_rank_image_read", 0.25), ("sft_rank_image_process", 0.5)] * 2
    assert helper.features == {}
    assert batch["multimodal_train_inputs"][0]["pixel_values"] is features["pixel_values"]
    helper.resolve(batch, 3, model=None)
    assert len(writes) == 4
    # CPU prepacking may resolve a microbatch on another thread; its Timer is
    # intentionally untouched while the decoded payload remains transient.
    background_batch = {"refs": [{}], "multimodal_train_inputs": [{}]}
    helper.resolve(background_batch, 4, model=None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(resolve_microbatch, background_batch).result()
    assert len(writes) == 4
