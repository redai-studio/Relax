# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Image-ref descriptor contract for ``--sft-image-preprocess-on-rank``."""

import numpy as np
import pytest
import torch
from PIL import Image

from relax.utils.data import image_rebuild
from relax.utils.data.image_refs import (
    HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND,
    KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND,
    SFT_IMAGE_REF_DESCRIPTOR_VERSION,
    SFT_IMAGE_REFS_FIELD,
    build_image_ref_descriptor,
    extract_stable_image_refs,
    split_pixel_values_from_mm_inputs,
    validate_image_ref_descriptor,
)


def test_extract_stable_image_refs_keeps_placeholder_order():
    refs = extract_stable_image_refs(
        ["/data/a.jpg", {"path": "/data/b.png"}, "/data/c.jpg"],
        idx=7,
        source_name="unit",
    )

    assert refs == ["/data/a.jpg", "/data/b.png", "/data/c.jpg"]


def test_extract_stable_image_refs_accepts_none_and_empty():
    assert extract_stable_image_refs(None, idx=0, source_name="unit") == []
    assert extract_stable_image_refs([], idx=0, source_name="unit") == []


@pytest.mark.parametrize("payload", ["data:image/png;base64,AAAA", {"path": "data:image/png;base64,AAAA"}])
def test_extract_stable_image_refs_rejects_inline_payloads(payload):
    with pytest.raises(ValueError, match="file-path image sources"):
        extract_stable_image_refs([payload], idx=1, source_name="unit")


@pytest.mark.parametrize("payload", [b"raw-bytes", {"base64": "AAAA"}, {"bytes": b"x"}, Image.new("RGB", (2, 2))])
def test_extract_stable_image_refs_rejects_non_reference_sources(payload):
    with pytest.raises(ValueError, match="file-path image sources"):
        extract_stable_image_refs([payload], idx=2, source_name="unit")


def test_split_pixel_values_from_mm_inputs_replaces_pixels_with_descriptor():
    mm_inputs = {
        "pixel_values": torch.zeros(12, 3),
        "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 8]]),
    }

    retained, descriptor = split_pixel_values_from_mm_inputs(
        mm_inputs,
        ["/data/a.jpg", "/data/b.jpg"],
        idx=0,
        source_name="unit",
        kind=KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND,
    )

    assert "pixel_values" not in retained
    assert retained["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 8]]
    assert descriptor["version"] == SFT_IMAGE_REF_DESCRIPTOR_VERSION
    assert descriptor["kind"] == KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND
    assert descriptor["image_refs"] == ["/data/a.jpg", "/data/b.jpg"]
    assert descriptor["pixel_shape"] == [12, 3]
    assert descriptor["image_grid_thw"] == [[1, 2, 4], [1, 2, 8]]
    validate_image_ref_descriptor(descriptor)


def test_split_pixel_values_rejects_grid_ref_count_mismatch():
    mm_inputs = {"pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 4]])}

    with pytest.raises(ValueError, match="do not match"):
        split_pixel_values_from_mm_inputs(
            mm_inputs, ["/a.jpg", "/b.jpg"], idx=3, source_name="unit", kind=KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND
        )


def test_split_pixel_values_rejects_missing_geometry():
    with pytest.raises(ValueError, match="pixel_values and image_grid_thw"):
        split_pixel_values_from_mm_inputs(
            {"image_grid_thw": torch.tensor([[1, 2, 4]])},
            ["/a.jpg"],
            idx=4,
            source_name="unit",
            kind=KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND,
        )


def test_validate_image_ref_descriptor_rejects_unknown_version_kind_and_refs():
    base = {
        "version": SFT_IMAGE_REF_DESCRIPTOR_VERSION,
        "kind": KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND,
        "image_refs": ["/a.jpg"],
        "pixel_shape": [4, 3],
    }

    assert validate_image_ref_descriptor(base) is base
    with pytest.raises(ValueError, match="version"):
        validate_image_ref_descriptor({**base, "version": 99})
    with pytest.raises(ValueError, match="kind"):
        validate_image_ref_descriptor({**base, "kind": "other_model_v1"})
    with pytest.raises(ValueError, match="image_refs"):
        validate_image_ref_descriptor({**base, "image_refs": []})
    with pytest.raises(ValueError, match="must be a dict"):
        validate_image_ref_descriptor("/a.jpg")


def test_validate_image_ref_descriptor_accepts_hf_kind_and_rejects_unknown():
    base = {
        "version": SFT_IMAGE_REF_DESCRIPTOR_VERSION,
        "kind": HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND,
        "image_refs": ["/a.jpg"],
        "pixel_shape": [4, 3],
    }

    assert validate_image_ref_descriptor(base) is base
    with pytest.raises(ValueError, match="supported kinds"):
        validate_image_ref_descriptor({**base, "kind": "unknown_model_v1"})


def test_build_image_ref_descriptor_rejects_unregistered_kind():
    with pytest.raises(ValueError, match="unsupported image-ref descriptor kind"):
        build_image_ref_descriptor(["/a.jpg"], torch.zeros(2, 3), torch.tensor([[1, 2, 4]]), kind="unknown_v1")


def test_build_image_ref_descriptor_round_trips_grid_ints():
    descriptor = build_image_ref_descriptor(
        ["/a.jpg"], torch.zeros(2, 3), torch.tensor([[1, 2, 4]]), kind=KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND
    )

    assert descriptor["image_grid_thw"] == [[1, 2, 4]]
    assert all(isinstance(value, int) for row in descriptor["image_grid_thw"] for value in row)


def test_extract_image_ref_descriptors_round_trips_tensordict_payload():
    from tensordict import TensorDict

    from relax.utils.data.image_refs import extract_image_ref_descriptors
    from relax.utils.utils import dict_to_tensordict

    descriptor = {
        "version": SFT_IMAGE_REF_DESCRIPTOR_VERSION,
        "kind": KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND,
        "image_refs": ["/a.jpg"],
        "pixel_shape": [4, 3],
        "image_grid_thw": [[1, 2, 4]],
    }
    payload = dict_to_tensordict({SFT_IMAGE_REFS_FIELD: [descriptor, None]}, batch_size=2)

    assert extract_image_ref_descriptors(payload) == [descriptor, None]
    # Post-conversion plain dicts take the same path in the foreground resolver.
    assert extract_image_ref_descriptors({SFT_IMAGE_REFS_FIELD: [descriptor, None]}) == [descriptor, None]
    assert extract_image_ref_descriptors(TensorDict({}, batch_size=2)) is None
    assert extract_image_ref_descriptors(None) is None


class _HFImageProcessor:
    """Minimal Qwen-VL-style image processor: pixel-only, config-driven."""

    def __init__(self):
        self.received_sizes = []

    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        self.received_sizes = [image.size for image in images]
        return {
            "pixel_values": torch.zeros(6, 3),
            "image_grid_thw": torch.tensor([[1, 2, 4], [1, 4, 6]]),
        }


class _HFStyleProcessor:
    """No preprocess_medias / no tokenizer access: if the rebuild dispatch ever
    picked the Kimi K3 builder for this kind, it would crash here."""

    def __init__(self):
        self.image_processor = _HFImageProcessor()


def _hf_descriptor(**overrides):
    from relax.utils.data.image_refs import SFT_IMAGE_REF_DESCRIPTOR_VERSION

    descriptor = {
        "version": SFT_IMAGE_REF_DESCRIPTOR_VERSION,
        "kind": HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND,
        "image_refs": ["/data/a.jpg", "/data/b.jpg"],
        "pixel_shape": [6, 3],
        "image_grid_thw": [[1, 2, 4], [1, 4, 6]],
    }
    descriptor.update(overrides)
    return descriptor


def test_hf_kind_rebuild_uses_image_processor_only(monkeypatch):
    from PIL import Image

    from relax.utils.data import processor_pool

    processor = _HFStyleProcessor()
    monkeypatch.setattr(processor_pool, "_worker_processor", processor)
    monkeypatch.setattr(image_rebuild, "_worker_threads_limited", True)
    monkeypatch.setattr(processor_pool, "_worker_multimodal_config", None)
    monkeypatch.setattr(
        image_rebuild,
        "load_image",
        lambda ref: Image.new("RGB", (8, 12)) if ref == "/data/a.jpg" else Image.new("RGB", (10, 16)),
    )

    features, timings = image_rebuild.build_image_features_in_worker(_hf_descriptor())

    assert processor.image_processor.received_sizes == [(8, 12), (10, 16)]
    assert features["pixel_values"].dtype == torch.bfloat16
    assert features["pixel_values"].is_shared()
    assert features["image_grid_thw"].tolist() == [[1, 2, 4], [1, 4, 6]]
    assert set(timings) == {"read_s", "process_s"}


@pytest.mark.parametrize("size", [(35, 21), (500, 1)])
def test_producer_and_rebuild_preserve_pixel_values(monkeypatch, tmp_path, size):
    from relax.utils.data import processor_pool
    from relax.utils.multimodal.config import MultimodalConfig

    class Qwen2VLImageProcessorFixture:
        __module__ = "transformers.models.qwen2_vl.image_processing_qwen2_vl"
        patch_size = 14

        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            return {
                "pixel_values": torch.cat(
                    [torch.from_numpy(np.array(image)).reshape(-1, 3).float() / 255 for image in images]
                ),
                "image_grid_thw": torch.tensor([[1, image.height, image.width] for image in images]),
            }

    class Processor:
        image_processor = Qwen2VLImageProcessorFixture()

        def __call__(self, text, images, return_tensors):
            return {"input_ids": [[1, 2]], **self.image_processor(images, return_tensors)}

    pixels = (np.arange(size[0] * size[1] * 3) % 256).astype(np.uint8).reshape(size[1], size[0], 3)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    monkeypatch.setattr(processor_pool, "_worker_processor", Processor())
    monkeypatch.setattr(processor_pool, "_worker_multimodal_config", MultimodalConfig(image_max_token_num=4))
    monkeypatch.setattr(image_rebuild, "_worker_threads_limited", True)

    tokens, produced = processor_pool.process_sample_from_paths_in_worker(
        "prompt", {"image": [str(path)]}, {"return_tensors": "pt"}
    )
    descriptor = build_image_ref_descriptor(
        [str(path)], produced["pixel_values"], produced["image_grid_thw"], HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND
    )
    rebuilt, _ = image_rebuild.build_image_features_in_worker(descriptor)

    assert tokens == [1, 2]
    assert rebuilt["pixel_values"].dtype == produced["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(rebuilt["pixel_values"], produced["pixel_values"])
    assert torch.equal(rebuilt["image_grid_thw"], produced["image_grid_thw"])


class _FakeFuture:
    def __init__(self, produce):
        self._produce = produce

    def result(self):
        return self._produce()


class _RecordingExecutor:
    """Records submission order; ``result()`` asserts none is collected
    early."""

    def __init__(self):
        self.submitted: list[str] = []

    def submit(self, _fn, descriptor):
        self.submitted.append(descriptor["image_refs"][0])

        def produce():
            # Pool parallelism depends on every sample being in flight before the
            # first result is awaited; the old submit/result loop failed this.
            assert len(self.submitted) == 2, "all samples must be submitted before any result is collected"
            return (
                {
                    "pixel_values": torch.zeros(2, 3),
                    "image_grid_thw": torch.tensor([[1, 1, 2]]),
                    "ref": descriptor["image_refs"][0],
                },
                {"read_s": 0.25, "process_s": 0.5},
            )

        return _FakeFuture(produce)


def test_build_batch_image_features_parallelizes_and_keeps_sample_order():
    from unittest.mock import MagicMock

    executor = _RecordingExecutor()
    pool = MagicMock()
    pool.executor = executor
    descriptors = [_hf_descriptor(image_refs=["/data/a.jpg"]), None, _hf_descriptor(image_refs=["/data/b.jpg"])]

    features, timings = image_rebuild.build_batch_image_features(pool, descriptors)
    assert timings == {"sft_rank_image_read": 0.5, "sft_rank_image_process": 1.0}

    assert executor.submitted == ["/data/a.jpg", "/data/b.jpg"]  # text-only rows are not submitted
    # The prefetch stash and the foreground resolver look features up by index.
    assert [None if features[i] is None else features[i]["ref"] for i in range(3)] == [
        "/data/a.jpg",
        None,
        "/data/b.jpg",
    ]
