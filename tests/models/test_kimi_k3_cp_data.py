# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU integration of Relax's VL batch packing and K3's CP token layout."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("vision_mode", ["trainable", "frozen", "tp_frozen", "other_model"])
@pytest.mark.parametrize("prepack", [False, True])
def test_kimi_k3_processor_batch_model_keeps_only_k3_grid_on_cpu(monkeypatch, vision_mode, prepack):
    pytest.importorskip("megatron.core.transformer.module")
    import numpy as np
    from megatron.core import mpu

    from relax.backends.megatron import data
    from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
    from relax.utils.data import processor_pool
    from relax.utils.data.kimi_k3 import KIMI_K3_SFT_LOSS_MASK, KIMI_K3_SFT_REQUEST, make_kimi_k3_sft_request
    from tests.utils.data.test_kimi_k3 import _SegmentTokenizer

    class Processor:
        tokenizer = _SegmentTokenizer()
        media_proc_cfg = {"merge_kernel_size": 2}

        def __init__(self):
            self.media_processor = self

        def preprocess_medias(self, medias):
            return medias, ["<|media_pad|>" for _ in medias]

        def preprocess(self, medias, return_tensors):
            return {"pixel_values": torch.ones(8, 3), "grid_thws": torch.tensor([[1, 2, 2], [1, 2, 2]])}

    processor = Processor()
    monkeypatch.setattr(processor_pool, "_worker_processor", processor)
    monkeypatch.setattr(processor_pool, "_worker_multimodal_config", None)
    sample = CanonicalSample(
        [
            CanonicalMessage("user", [{"type": "image"}, {"type": "image"}], False),
            CanonicalMessage("assistant", "answer", True),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    ids, inputs = processor_pool.process_sample_in_worker(
        "unused",
        {"images": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2},
        {KIMI_K3_SFT_REQUEST: make_kimi_k3_sft_request(sample)},
    )
    grid = inputs["image_grid_thw"]
    mask = inputs.pop(KIMI_K3_SFT_LOSS_MASK)
    args = SimpleNamespace(dynamic_context_parallel=False)
    spec = importlib.util.spec_from_file_location(
        "_k3_grid_configuration", Path(__file__).parents[2] / "relax/models/kimi_k3/configuration.py"
    )
    configuration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration)
    if vision_mode != "other_model":
        configuration.configure_runtime(SimpleNamespace(), args)
    monkeypatch.setattr(data, "get_args", lambda: args)
    for name, value in (
        ("get_context_parallel_world_size", 1),
        ("get_context_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 1),
    ):
        monkeypatch.setattr(mpu, name, lambda value=value: value)
    batch = data.get_batch(
        iter(
            [
                (
                    dict(
                        tokens=[torch.tensor(ids)],
                        total_lengths=[len(ids)],
                        response_lengths=[len(ids)],
                        loss_masks=[mask],
                        multimodal_train_inputs=[inputs],
                    ),
                    None,
                )
            ]
        ),
        keys=["tokens", "loss_masks", "multimodal_train_inputs"],
        pad_multiplier=1,
        is_vl_model=True,
        pack_device=torch.device("cpu") if prepack else None,
    )
    # The H2D prefetch path runs this mover after CPU packing; meta exposes any
    # attempt by vision to read a device grid back as Python geometry without CUDA.
    moved = data.move_tensors_to_device(batch, torch.device("meta"), non_blocking=True)
    assert inputs["image_grid_thw"] is grid
    assert "grid_thws" not in inputs  # Producer/descriptor schema is unchanged.
    mm_inputs = moved["multimodal_train_inputs"]
    assert mm_inputs["pixel_values"].device.type == "meta"
    if vision_mode == "other_model":
        assert mm_inputs["image_grid_thw"].device.type == "meta"
        assert "grid_thws" not in mm_inputs
        return
    assert mm_inputs["grid_thws"] is grid
    assert "image_grid_thw" not in mm_inputs
    assert moved["multimodal_num_items"]["grid_thws"] == [2]
    data.record_tensors_on_stream({"grid_thws": grid}, None)

    class LanguageModel(torch.nn.Module):
        pre_process = post_process = True
        share_embeddings_and_output_weights = False
        pg_collection = SimpleNamespace(tp=object())

        def embedding(self, input_ids, position_ids=None):
            return torch.zeros(input_ids.shape[1], input_ids.shape[0], 3, device=input_ids.device)

        def forward(self, **kwargs):
            return kwargs["decoder_input"]

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, device="meta"), requires_grad=vision_mode == "trainable")
            self.grids = []

        def forward(self, pixels, grid_thws):
            assert grid_thws.device.type == "cpu"
            self.grids.extend(grid_thws.tolist())
            return [pixels.new_zeros((h * w // 4, 4, 3)) for _, h, w in grid_thws.tolist()]

    class Projector(torch.nn.Module):
        def forward(self, features):
            return [feature.mean(dim=1) for feature in features]

    spec = importlib.util.spec_from_file_location(
        "_k3_grid_model", Path(__file__).parents[2] / "relax/models/kimi_k3/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vision = Vision()
    monkeypatch.setattr(module, "_build_vision_modules", lambda _: (vision, Projector()))
    # Exercise TP vision's host geometry reads; collectives are shape-only here.
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(module.dist, "all_reduce", lambda tensor, group: None)
    config = SimpleNamespace(
        provide_language_model=lambda **_: LanguageModel(),
        sequence_parallel=False,
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<|media_pad|>"),
        vision_dp_when_tp=vision_mode == "tp_frozen",
        vision_config=SimpleNamespace(merge_kernel_size=(2, 2), vt_hidden_size=3),
    )
    model = module.KimiK3VLModel(config).eval()
    result = model(moved["tokens"], **mm_inputs)
    assert result.device.type == "meta"
    assert vision.grids == grid.tolist()[: 1 if vision_mode == "tp_frozen" else 2]


def test_kimi_k3_grid_stays_on_host_when_prefetch_moves_pixels_to_device():
    pytest.importorskip("megatron.core.transformer.module")
    from relax.backends.megatron import data

    grid = torch.tensor([[1, 4, 4]])
    source = {"multimodal_train_inputs": {"grid_thws": grid, "pixel_values": torch.ones(16, 3, 14, 14)}}
    result = data.move_tensors_to_device(source, torch.device("meta"))
    assert result["multimodal_train_inputs"]["grid_thws"] is grid
    assert result["multimodal_train_inputs"]["pixel_values"].device.type == "meta"
    # Host geometry must not be passed to CUDA record_stream.
    data.record_tensors_on_stream({"grid_thws": grid}, None)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_kimi_k3_cp_layout_consumes_real_prepacked_batch_and_loss_coordinates(monkeypatch, cp_rank):
    pytest.importorskip("megatron.core.transformer.module")
    from megatron.core import mpu

    from relax.backends.megatron import cp_utils, data

    for name, value in (
        ("get_context_parallel_world_size", 2),
        ("get_context_parallel_rank", cp_rank),
        ("get_tensor_model_parallel_world_size", 2),
    ):
        monkeypatch.setattr(mpu, name, lambda value=value: value)
    monkeypatch.setattr(data, "get_args", lambda: SimpleNamespace(dynamic_context_parallel=False))
    lengths = [9, 5]
    tokens = [torch.arange(1, length + 1) for length in lengths]
    batch = data.get_batch(
        iter(
            [
                (
                    dict(
                        tokens=tokens,
                        total_lengths=lengths,
                        response_lengths=lengths,
                        loss_masks=[torch.ones(length) for length in lengths],
                    ),
                    None,
                )
            ]
        ),
        keys=["tokens", "loss_masks", "total_lengths", "response_lengths"],
        pad_multiplier=1,
        is_vl_model=True,
        pack_device=torch.device("cpu"),
    )
    # Exercise the same metadata copier used by SFT's CPU prefetch worker.
    batch = data.move_tensors_to_device(batch, torch.device("cpu"))
    packed = batch["vlm_packed_seq_params"]
    assert packed.cu_seqlens_q_cpu == [0, 16, 24]
    assert batch["padded_total_lengths"] == [16, 8]
    torch.testing.assert_close(packed.cu_seqlens_q, torch.tensor([0, 16, 24], dtype=torch.int32))

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_process = True
            self.post_process = True
            self.share_embeddings_and_output_weights = False
            self.pg_collection = SimpleNamespace(tp=None, cp=SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank))
            self.call = None
            self.embedding_input = None

        def embedding(self, input_ids, position_ids=None):
            self.embedding_input = input_ids
            return input_ids.float().unsqueeze(-1).transpose(0, 1)

        def forward(self, **kwargs):
            self.call = kwargs
            return kwargs["decoder_input"]

    spec = importlib.util.spec_from_file_location(
        "_k3_cp_data_model", Path(__file__).parents[2] / "relax/models/kimi_k3/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_build_vision_modules", lambda _: (torch.nn.Identity(), torch.nn.Identity()))
    config = SimpleNamespace(
        provide_language_model=lambda **_: LanguageModel(), image_token_id=31, sequence_parallel=False
    )
    model = module.KimiK3VLModel(config).eval()
    output = model(batch["unsplit_tokens"], attention_mask=batch["unsplit_attention_mask"], packed_seq_params=packed)
    expected = []
    expected_valid_count = 0
    for sample, length, padded_length in zip(tokens, lengths, batch["padded_total_lengths"], strict=True):
        chunk = padded_length // 4
        padded = torch.nn.functional.pad(sample, (0, padded_length - length))
        for begin in (cp_rank * chunk, (3 - cp_rank) * chunk):
            expected.append(padded[begin : begin + chunk])
            # SFT predicts token i+1; the last real token has no target.
            expected_valid_count += max(0, min(begin + chunk, length - 1) - begin)
    torch.testing.assert_close(output[:, 0, 0], torch.cat(expected).float())
    assert model.language_model.embedding_input.shape == (1, 12)
    torch.testing.assert_close(model.language_model.embedding_input[0], torch.cat(expected))
    assert model.language_model.call["padding_mask"].shape == (1, 12)
    count = cp_utils.get_cp_local_num_tokens(
        lengths,
        lengths,
        batch["loss_masks"],
        qkv_format="thd",
        padded_total_lengths=batch["padded_total_lengths"],
    )
    assert int(count) == expected_valid_count
