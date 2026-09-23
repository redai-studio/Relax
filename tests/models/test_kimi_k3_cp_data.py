# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU integration of Relax's VL batch packing and K3's CP token layout."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


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
