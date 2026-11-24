# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import json
import os
import weakref
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.backends.fsdp import sft as sft_module
from relax.backends.fsdp.arguments import validate_i2i_sft_config
from relax.engine.sft.condition_tracker import (
    ConditionTracker,
    compute_past_key_values_bytes,
    summarize_baseline_dir,
)
from relax.engine.sft.image import ImageBatchSampler


def _make_rows() -> list[dict[str, object]]:
    return [
        {"prompt": "edit prompt A", "image": "t0.png", "edit_image": ["r0_a.png", "r0_b.png"]},
        {"prompt": "edit prompt B", "image": "t1.png", "edit_image": ["r1.png"]},
        {"prompt": "edit prompt A", "image": "t2.png", "edit_image": ["r0_a.png", "r0_b.png"]},
        {"prompt": "edit prompt C", "image": "t3.png", "edit_image": ["r3.png"]},
        {"prompt": "edit prompt D", "image": "t4.png", "edit_image": ["r4.png"]},
    ]


def test_condition_repeat_and_padding_counts_across_epochs(tmp_path: Path) -> None:
    rows = _make_rows()
    dataset_size = len(rows)
    global_batch_size = 4
    world_size = 2
    micro_batches = global_batch_size // world_size
    seed = 42
    config = {"max_pixels": 1048576, "fixed_pixels_tiers": None, "seed": seed}
    sampler = ImageBatchSampler(dataset_size, global_batch_size, seed)

    trackers = [
        ConditionTracker(
            rank=r,
            world_size=world_size,
            micro_batches=micro_batches,
            dataset_size=dataset_size,
            output_dir=tmp_path,
            warmup_steps=1,
            seed=seed,
        )
        for r in range(world_size)
    ]

    # Run 2 epochs of 2 steps each: 4 steps consume 16 sample positions
    for step in range(4):
        batch_ids = sampler.batch(step)
        for r in range(world_size):
            tracker = trackers[r]
            tracker.begin_step(step, torch.device("cpu"))
            rank_ids = batch_ids[r::world_size]
            for micro, sample_id in enumerate(rank_ids):
                row = rows[sample_id]
                tracker.begin_micro(
                    micro=micro,
                    sample_id=sample_id,
                    row=row,
                    config=config,
                    training_mode="lora",
                    height=64,
                    width=64,
                )
                edit_list = list(row["edit_image"])
                with tracker.host_stage("ref_image_preprocess_ms"):
                    for img_idx in range(len(edit_list)):
                        pv = torch.zeros((16, 64), dtype=torch.bfloat16)
                        ghw = torch.tensor([[4, 4]], dtype=torch.int64)
                        tracker.record_transfer("ref_pixel_values", pv, image_index=img_idx)
                        tracker.record_transfer("ref_grid_hw", ghw, image_index=img_idx)

                tracker.set_prefix_context(str(row["prompt"]), is_positive=True, think_mode=False)
                input_ids = torch.zeros((1, 12), dtype=torch.int64)
                tracker.record_transfer("prefix_input_ids", input_ids)

                with tracker.gpu_stage("ref_vision_encode_ms"):
                    tracker.record_encode(
                        "ref_vision_encode",
                        num_items=len(edit_list),
                        feature_bytes={"ref_vision_embeds_bytes": len(edit_list) * 1024},
                    )
                with tracker.gpu_stage("prefix_encode_ms"):
                    tracker.record_encode(
                        "prefix_encode",
                        num_items=1,
                        feature_bytes={"past_key_values_bytes": 4096, "indexes_image_bytes": 128},
                    )

                with tracker.host_stage("target_image_preprocess_ms"):
                    latents = torch.zeros((1, 16, 8, 8), dtype=torch.bfloat16)
                    tracker.record_transfer("target_input_latents", latents)

                with tracker.gpu_stage("gen_vision_encode_ms"):
                    tracker.record_encode("gen_vision_encode", num_items=1, eligible=False)
                with tracker.gpu_stage("dit_forward_ms"):
                    pass
                with tracker.gpu_stage("backward_ms"):
                    pass

            with tracker.gpu_stage("optimizer_ms"):
                pass
            tracker.end_step(step, data_wait_seconds=0.002, train_seconds=0.020)

    summaries = [t.save_summary() for t in trackers]
    agg = summarize_baseline_dir(tmp_path)

    # 4 steps * global_batch_size 4 = 16 positions over 5 unique rows; each epoch pads 8 - 5 = 3, 6 in total
    assert agg["sample_stats"]["total_samples"] == 16
    assert agg["sample_stats"]["global_unique_rows"] == 5
    assert agg["sample_stats"]["padding_samples"] == 6
    assert sum(s["sample_stats"]["padding_samples"] for s in summaries) == 6
    assert agg["sample_stats"]["global_repeated_samples"] == 16 - 5

    # Rows 0 and 2 share (prompt, edit_image, height, width), so there are 4 unique conditions
    ref_enc = agg["encoding_stats"]["ref_vision_encode"]
    prefix_enc = agg["encoding_stats"]["prefix_encode"]
    gen_enc = agg["encoding_stats"]["gen_vision_encode"]

    assert ref_enc["eligible"] is True
    assert ref_enc["calls"] == 16
    assert ref_enc["global_repeated_calls"] == 16 - 4
    assert "cache_lookups" not in ref_enc
    assert "cache_hits" not in ref_enc
    assert "cache_hit_rate" not in ref_enc

    # Rows 0 and 2 carry 2 reference images, and encoded_images counts every image
    all_consumed_ids = [sid for step in range(4) for sid in sampler.batch(step)]
    expected_encoded_images = sum(len(rows[sid]["edit_image"]) for sid in all_consumed_ids)
    assert ref_enc["encoded_images"] == expected_encoded_images

    assert prefix_enc["eligible"] is True
    assert prefix_enc["calls"] == 16
    assert prefix_enc["global_repeated_calls"] == 16 - 4
    assert "cache_lookups" not in prefix_enc
    assert "cache_hits" not in prefix_enc
    assert "cache_hit_rate" not in prefix_enc

    assert gen_enc["eligible"] is False
    assert gen_enc["calls"] == 16
    assert gen_enc["local_repeated_calls"] == 0
    assert gen_enc["global_repeated_calls"] == 0

    assert len(agg["repeat_evidence"]["global_examples"]) > 0


def test_dp_tier_micro_balance_padding_flags(tmp_path: Path) -> None:
    dataset_size = 5
    global_batch_size = 4
    world_size = 2
    seed = 7
    config = {
        "max_pixels": 1048576,
        "fixed_pixels_tiers": [262144, 1048576],
        "seed": seed,
        "dp_tier_micro_balance": True,
        "dp_tier_rank_rotation": True,
    }
    rows = _make_rows()
    sampler = ImageBatchSampler(dataset_size, global_batch_size, seed)
    trackers = [
        ConditionTracker(
            rank=r,
            world_size=world_size,
            micro_batches=global_batch_size // world_size,
            dataset_size=dataset_size,
            output_dir=tmp_path,
            warmup_steps=0,
            seed=seed,
        )
        for r in range(world_size)
    ]

    expected_micro_records: dict[int, list[tuple[int, list[int], list[bool]]]] = {r: [] for r in range(world_size)}
    for step in range(4):
        tagged = sampler.batch_with_padding(step)
        raw_ids = [sid for sid, _ in tagged]
        from relax.engine.sft.image import assign_image_batch_positions

        positions = assign_image_batch_positions(raw_ids, config, step=step, world=world_size)
        for r in range(world_size):
            trackers[r].begin_step(step, torch.device("cpu"))
            rank_positions = positions[r::world_size]
            step_sample_ids: list[int] = []
            step_padding_flags: list[bool] = []
            for micro, pos in enumerate(rank_positions):
                sid, is_pad = tagged[pos]
                step_sample_ids.append(int(sid))
                step_padding_flags.append(bool(is_pad))
                trackers[r].begin_micro(micro, sid, rows[sid], config, "lora", 64, 64, is_padding=is_pad)
                assert trackers[r]._current_micro_records[micro]["is_padding"] is is_pad
            expected_micro_records[r].append((step, step_sample_ids, step_padding_flags))
            trackers[r].end_step(step, 0.001, 0.01)

    for r in range(world_size):
        step_lines = [
            json.loads(line)
            for line in (tmp_path / f"rank_{r}_steps.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(step_lines) == len(expected_micro_records[r])
        for rec, (expected_step, expected_sids, expected_pads) in zip(step_lines, expected_micro_records[r]):
            assert rec["step"] == expected_step
            assert [m["is_padding"] for m in rec["micros"]] == expected_pads
            assert [m["sample_id"] for m in rec["micros"]] == expected_sids

    for t in trackers:
        t.save_summary()
    agg = summarize_baseline_dir(tmp_path)
    assert agg["sample_stats"]["padding_samples"] == 6
    assert sum(t.padding_samples for t in trackers) == 6


def test_transfer_bytes_and_feature_tensor_sizes_without_retaining_tensors(tmp_path: Path) -> None:
    tracker = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=1,
        dataset_size=2,
        output_dir=tmp_path,
        warmup_steps=0,
        seed=42,
    )
    row = {"prompt": "test", "image": "target.png", "edit_image": ["ref.png"]}
    config = {"max_pixels": 1048576, "fixed_pixels_tiers": None, "seed": 42}

    # Step 0: first occurrence
    tracker.begin_step(0, torch.device("cpu"))
    tracker.begin_micro(0, 0, row, config, "lora", 64, 64)
    tracker.set_prefix_context("test", is_positive=True, think_mode=False)

    bf16_tensor = torch.ones((32, 64), dtype=torch.bfloat16)
    int64_tensor = torch.ones((1, 20), dtype=torch.int64)
    kv_k = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16)
    kv_v = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16)
    kv_cache = [(kv_k, kv_v)]
    vision_embeds = torch.ones((1, 16, 256), dtype=torch.bfloat16)
    indexes_image = torch.ones((3, 16), dtype=torch.int64)

    ref_bf16 = weakref.ref(bf16_tensor)
    ref_kv = weakref.ref(kv_k)

    expected_bf16_bytes = 32 * 64 * 2
    expected_int64_bytes = 20 * 8
    expected_kv_bytes = 2 * (1 * 8 * 16 * 128 * 2)
    expected_vision_bytes = 1 * 16 * 256 * 2
    expected_idx_bytes = 3 * 16 * 8

    assert compute_past_key_values_bytes(kv_cache) == expected_kv_bytes

    tracker.record_transfer("ref_pixel_values", bf16_tensor, image_index=0)
    tracker.record_transfer("prefix_input_ids", int64_tensor)
    tracker.record_encode(
        "ref_vision_encode",
        num_items=1,
        feature_bytes={"ref_vision_embeds_bytes": vision_embeds.numel() * vision_embeds.element_size()},
    )
    tracker.record_encode(
        "prefix_encode",
        num_items=1,
        feature_bytes={
            "past_key_values_bytes": compute_past_key_values_bytes(kv_cache),
            "indexes_image_bytes": int(indexes_image.numel() * indexes_image.element_size()),
        },
    )
    tracker.end_step(0, 0.001, 0.01)

    del bf16_tensor
    del kv_k
    del kv_v
    del kv_cache
    assert ref_bf16() is None
    assert ref_kv() is None

    # Step 1: the same sample appears again and counts as repeated_transfer_bytes
    tracker.begin_step(1, torch.device("cpu"))
    tracker.begin_micro(0, 0, row, config, "lora", 64, 64)
    tracker.set_prefix_context("test", is_positive=True, think_mode=False)
    tracker.record_transfer("ref_pixel_values", torch.ones((32, 64), dtype=torch.bfloat16), image_index=0)
    tracker.record_transfer("prefix_input_ids", torch.ones((1, 20), dtype=torch.int64))
    tracker.end_step(1, 0.001, 0.01)

    summary = tracker.save_summary()
    ref_pv_stats = summary["transfer_stats"]["ref_pixel_values"]
    assert ref_pv_stats["total_transfer_calls"] == 2
    assert ref_pv_stats["total_transfer_bytes"] == expected_bf16_bytes * 2
    assert ref_pv_stats["repeated_transfer_calls"] == 1
    assert ref_pv_stats["repeated_transfer_bytes"] == expected_bf16_bytes

    prefix_id_stats = summary["transfer_stats"]["prefix_input_ids"]
    assert prefix_id_stats["total_transfer_bytes"] == expected_int64_bytes * 2
    assert prefix_id_stats["repeated_transfer_bytes"] == expected_int64_bytes

    feat_stats = summary["feature_tensor_bytes"]
    assert feat_stats["ref_vision_embeds_bytes"]["max_bytes"] == expected_vision_bytes
    assert feat_stats["past_key_values_bytes"]["max_bytes"] == expected_kv_bytes
    assert feat_stats["indexes_image_bytes"]["max_bytes"] == expected_idx_bytes


def test_gpu_peak_memory_preserves_lifetime_max_and_step_peak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=1,
        dataset_size=2,
        output_dir=tmp_path,
        warmup_steps=0,
    )
    tracker._cuda_available = True

    state = {"alloc": 5000, "reserved": 8000, "reset_count": 0}

    def fake_max_alloc(_device: object = None) -> int:
        return state["alloc"]

    def fake_max_reserved(_device: object = None) -> int:
        return state["reserved"]

    def fake_reset(_device: object = None) -> None:
        state["reset_count"] += 1
        state["alloc"] = 0
        state["reserved"] = 0

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fake_max_alloc)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", fake_max_reserved)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fake_reset)

    # The peaks are 5000 / 8000 before step 0, and the step 0 peaks are 3000 / 6000
    cuda_dev = torch.device("cuda", 0)
    tracker.begin_step(0, cuda_dev)
    assert state["reset_count"] == 1
    assert tracker.max_allocated == 5000
    assert tracker.max_reserved == 8000

    state["alloc"] = 3000
    state["reserved"] = 6000
    lifetime_alloc, lifetime_reserved = tracker.consume_step_memory(cuda_dev)
    assert (lifetime_alloc, lifetime_reserved) == (5000, 8000)
    assert tracker._current_step_peak_allocated == 3000
    assert tracker._current_step_peak_reserved == 6000
    tracker.end_step(0, 0.001, 0.01)

    # The step 1 peaks rise to 7000 / 9000, and the lifetime peaks follow
    tracker.begin_step(1, cuda_dev)
    state["alloc"] = 7000
    state["reserved"] = 9000
    lifetime_alloc, lifetime_reserved = tracker.consume_step_memory(cuda_dev)
    assert (lifetime_alloc, lifetime_reserved) == (7000, 9000)
    assert tracker._current_step_peak_allocated == 7000
    assert tracker._current_step_peak_reserved == 9000
    tracker.end_step(1, 0.001, 0.01)

    summary = tracker.save_summary()
    assert summary["memory_stats"]["step_peak_allocated_max_bytes"] == 7000
    assert summary["memory_stats"]["step_peak_allocated_mean_bytes"] == 5000.0
    assert summary["memory_stats"]["lifetime_peak_allocated_bytes"] == 7000
    assert summary["memory_stats"]["lifetime_peak_reserved_bytes"] == 9000


def _run_two_step_sft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enable_tracker: bool) -> dict[str, object]:
    sft_module.seed_training(1234)

    rows = [
        {"prompt": "p0", "image": "i0.png", "edit_image": ["e0.png"]},
        {"prompt": "p1", "image": "i1.png", "edit_image": ["e1.png"]},
    ]
    config: dict[str, object] = {
        "data_root": str(tmp_path),
        "dataset_size": 2,
        "max_pixels": 65536,
        "fixed_pixels_tiers": None,
        "seed": 1234,
    }
    if enable_tracker:
        config["baseline_stats_dir"] = str(tmp_path / "stats")
        config["baseline_warmup_steps"] = 0

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.25, -0.75], dtype=torch.float32))
            self.pipe = SimpleNamespace(dit=SimpleNamespace(tracker=None), tracker=None)

        def forward(self, data: dict[str, object]) -> torch.Tensor:
            tracker = getattr(self.pipe, "tracker", None)
            x = data["tensor"]
            if tracker is not None:
                with tracker.host_stage("ref_image_preprocess_ms"):
                    tracker.record_transfer("ref_pixel_values", x, image_index=0)
                tracker.set_prefix_context(str(data["prompt"]), is_positive=True, think_mode=False)
                with tracker.gpu_stage("ref_vision_encode_ms"):
                    tracker.record_encode("ref_vision_encode", 1, {"ref_vision_embeds_bytes": 64})
                with tracker.gpu_stage("prefix_encode_ms"):
                    tracker.record_encode(
                        "prefix_encode", 1, {"past_key_values_bytes": 128, "indexes_image_bytes": 16}
                    )
                with tracker.gpu_stage("gen_vision_encode_ms"):
                    tracker.record_encode("gen_vision_encode", 1, eligible=False)
                with tracker.gpu_stage("dit_forward_ms"):
                    noise = torch.randn_like(self.weight)
                    out = (self.weight * x + noise * 0.1).square().mean()
            else:
                noise = torch.randn_like(self.weight)
                out = (self.weight * x + noise * 0.1).square().mean()
            return out

    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sampler = ImageBatchSampler(2, 2, 1234)

    samples = [
        {"prompt": "p0", "image": SimpleNamespace(height=32, width=32), "tensor": torch.tensor([0.5, -1.5])},
        {"prompt": "p1", "image": SimpleNamespace(height=32, width=32), "tensor": torch.tensor([-0.25, 2.0])},
    ]

    class _StubDataset:
        def __init__(self) -> None:
            self.rows = rows
            self.config = config

        def __getitem__(self, idx: int) -> dict[str, object]:
            return samples[idx]

    trainer = sft_module.FSDPSFTTrainer.__new__(sft_module.FSDPSFTTrainer)
    trainer.world = 1
    trainer.rank = 0
    trainer.group = None
    trainer.step = 0
    trainer.prefetch = None
    trainer.condition_cache = None
    trainer.parameters = [p for p in model.parameters() if p.requires_grad]
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer.profiler = SimpleNamespace(step=lambda _s: None)
    trainer.sampler = sampler
    trainer.dataset = _StubDataset()
    trainer.args = Namespace(
        i2i_sft_config=config,
        global_batch_size=2,
        clip_grad=1.0,
        num_rollout=2,
        save=None,
        save_interval=0,
        seed=1234,
    )
    trainer.actor = SimpleNamespace(
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        _trainable_mode="lora",
        _read_train_partition=lambda s: {"sample_id": torch.tensor(sampler.batch(s), dtype=torch.int64)},
        _rank0_then_agree=lambda work, what="": work(),
    )

    if enable_tracker:
        trainer.tracker = ConditionTracker(
            rank=0,
            world_size=1,
            micro_batches=2,
            dataset_size=2,
            output_dir=Path(str(config["baseline_stats_dir"])),
            warmup_steps=0,
            seed=1234,
        )
        model.pipe.tracker = trainer.tracker
        model.pipe.dit.tracker = trainer.tracker
    else:
        trainer.tracker = None

    reported: list[dict[str, object]] = []
    monkeypatch.setattr("relax.backends.fsdp.actor._collect_weight_sync_errors", lambda err: [err] if err else [])
    monkeypatch.setattr("relax.utils.distributed_utils.get_gloo_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *_a, **_kw: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        sft_module,
        "clip_sharded_gradients",
        lambda params, max_norm, _group: torch.nn.utils.clip_grad_norm_(params, max_norm),
    )
    monkeypatch.setattr(sft_module, "_report_sft_metrics", lambda _args, vals: reported.append(vals))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_a, **_kw: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_a, **_kw: 0)

    trainer.train(0)
    trainer.train(1)

    return {
        "losses": [r["train/loss"] for r in reported],
        "grad_norms": [r["train/grad_norm"] for r in reported],
        "weight": model.weight.detach().clone(),
        "rng_state": torch.get_rng_state().clone(),
    }


def test_training_parity_with_and_without_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    res_off = _run_two_step_sft(tmp_path / "off", monkeypatch, enable_tracker=False)
    res_on = _run_two_step_sft(tmp_path / "on", monkeypatch, enable_tracker=True)

    assert res_off["losses"] == res_on["losses"]
    assert res_off["grad_norms"] == res_on["grad_norms"]
    assert torch.equal(res_off["weight"], res_on["weight"])
    assert torch.equal(res_off["rng_state"], res_on["rng_state"])
    assert (tmp_path / "on" / "stats" / "rank_0_steps.jsonl").is_file()
    assert (tmp_path / "on" / "stats" / "rank_0_summary.json").is_file()


def test_validate_i2i_sft_config_baseline_fields(tmp_path: Path) -> None:
    base_args = Namespace(
        loss_type="sft",
        task_type="causal_lm",
        generation_task="i2i",
        train_backend="fsdp",
        model_path="/worker/model",
        prompt_data="/worker/data.json",
        i2i_sft_config={"data_root": "/worker", "dataset_size": 8},
        global_batch_size=8,
        rollout_batch_size=8,
        fsdp_trainable_mode="lora",
        lora_rank=32,
        lora_alpha=32,
        model_adapter_path=None,
        num_rollout=4,
        num_epoch=None,
        seed=42,
        num_updates_per_batch=1,
        fsdp_lr_scheduler="constant",
        fsdp_param_dtype="bf16",
        fsdp_reduce_dtype="fp32",
        fsdp_master_dtype=None,
    )
    validate_i2i_sft_config(base_args)
    assert "baseline_stats_dir" not in base_args.i2i_sft_config
    assert "baseline_warmup_steps" not in base_args.i2i_sft_config

    enabled_args = Namespace(
        **{
            **vars(base_args),
            "i2i_sft_config": {
                "data_root": "/worker",
                "dataset_size": 8,
                "baseline_stats_dir": str(tmp_path / "baseline_out"),
                "baseline_warmup_steps": 3,
            },
        }
    )
    validate_i2i_sft_config(enabled_args)
    assert enabled_args.i2i_sft_config["baseline_stats_dir"] == str(tmp_path / "baseline_out")
    assert enabled_args.i2i_sft_config["baseline_warmup_steps"] == 3

    for bad_dir in ("", "   ", 123):
        bad_args = Namespace(
            **{
                **vars(base_args),
                "i2i_sft_config": {"data_root": "/worker", "dataset_size": 8, "baseline_stats_dir": bad_dir},
            }
        )
        with pytest.raises(ValueError, match="baseline_stats_dir"):
            validate_i2i_sft_config(bad_args)

    for bad_warmup in (-1, True, "5"):
        bad_args = Namespace(
            **{
                **vars(base_args),
                "i2i_sft_config": {"data_root": "/worker", "dataset_size": 8, "baseline_warmup_steps": bad_warmup},
            }
        )
        with pytest.raises(ValueError, match="baseline_warmup_steps"):
            validate_i2i_sft_config(bad_args)


class _StubTokenizer:
    _SPECIAL = {"<img>": 1, "<IMG_CONTEXT>": 2, "</img>": 3}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL.get(token, 4)

    def __call__(
        self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True
    ) -> dict[str, torch.Tensor]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in self._SPECIAL.items():
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(4 + (ord(text[i]) % 200))
                i += 1
        return {"input_ids": torch.tensor([ids], dtype=torch.int64)}


def test_sensenova_pipeline_units_instrumentation_with_tracker(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    from PIL import Image

    from relax.models.sensenova.adapter import SenseNovaSFTModel, configure_attention
    from relax.models.sensenova.model import SenseNovaU1DiT
    from relax.models.sensenova.pipeline import SenseNovaU1ImagePipeline

    configure_attention("torch")
    device = torch.device("cpu")
    dtype = torch.float32

    pipe = SenseNovaU1ImagePipeline(device=device, torch_dtype=dtype)
    pipe.dit = SenseNovaU1DiT(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        pad_token_id=0,
        vision_hidden_size=64,
    ).to(device=device, dtype=dtype)
    pipe.tokenizer = _StubTokenizer()
    pipe.scheduler.set_timesteps(1000, training=True)

    model = SenseNovaSFTModel(pipe, training_mode="lora", prefix_gradient_checkpointing=False)
    tracker = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=1,
        dataset_size=1,
        output_dir=tmp_path / "pipe_stats",
        warmup_steps=0,
        seed=42,
    )
    pipe.tracker = tracker
    pipe.dit.tracker = tracker

    target_img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    ref_img_a = Image.new("RGB", (64, 64), color=(10, 20, 30))
    ref_img_b = Image.new("RGB", (64, 64), color=(40, 50, 60))
    row = {"prompt": "combine two reference images", "image": "t.png", "edit_image": ["a.png", "b.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}

    for step in range(2):
        tracker.begin_step(step, device)
        tracker.begin_micro(0, 0, row, config, "lora", target_img.height, target_img.width)
        loss = model({"prompt": row["prompt"], "image": target_img, "edit_image": [ref_img_a, ref_img_b]})
        assert torch.isfinite(loss)
        tracker.end_step(step, data_wait_seconds=0.001, train_seconds=0.01)

    summary = tracker.save_summary()
    agg = summarize_baseline_dir(tmp_path / "pipe_stats")

    assert summary["encoding_stats"]["ref_vision_encode"]["calls"] == 2
    assert summary["encoding_stats"]["ref_vision_encode"]["encoded_images"] == 4
    assert summary["encoding_stats"]["ref_vision_encode"]["repeated_calls"] == 1
    assert summary["encoding_stats"]["prefix_encode"]["calls"] == 2
    assert summary["encoding_stats"]["prefix_encode"]["repeated_calls"] == 1
    assert summary["encoding_stats"]["gen_vision_encode"]["calls"] == 2
    assert summary["encoding_stats"]["gen_vision_encode"]["eligible"] is False
    assert summary["transfer_stats"]["ref_pixel_values"]["total_transfer_calls"] == 4
    assert summary["transfer_stats"]["ref_pixel_values"]["repeated_transfer_calls"] == 2
    assert summary["transfer_stats"]["prefix_input_ids"]["total_transfer_calls"] == 2
    assert summary["transfer_stats"]["prefix_input_ids"]["repeated_transfer_calls"] == 1
    assert summary["transfer_stats"]["target_input_latents"]["total_transfer_calls"] == 2
    assert summary["feature_tensor_bytes"]["ref_vision_embeds_bytes"]["max_bytes"] > 0
    assert summary["feature_tensor_bytes"]["past_key_values_bytes"]["max_bytes"] > 0
    assert summary["feature_tensor_bytes"]["indexes_image_bytes"]["max_bytes"] > 0
    assert agg["encoding_stats"]["ref_vision_encode"]["local_repeated_calls"] == 1
    assert agg["encoding_stats"]["ref_vision_encode"]["global_repeated_calls"] == 1
    assert agg["encoding_stats"]["prefix_encode"]["local_repeated_calls"] == 1
    assert agg["encoding_stats"]["prefix_encode"]["global_repeated_calls"] == 1


def test_summarize_baseline_dir_detects_missing_stale_and_mixed_runs(tmp_path: Path) -> None:
    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}

    # 仅收集到 rank 0 而缺失 rank 1 时，严格模式必须报错，allow_incomplete=True 须显式标明不完整。
    missing_dir = tmp_path / "missing_rank"
    t0 = ConditionTracker(
        rank=0, world_size=2, micro_batches=1, dataset_size=2, output_dir=missing_dir, warmup_steps=0, run_id="run-a"
    )
    t0.begin_step(0, torch.device("cpu"))
    t0.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
    t0.end_step(0, 0.001, 0.01)
    t0.save_summary()
    with pytest.raises(ValueError, match="missing ranks \\[1\\]"):
        summarize_baseline_dir(missing_dir)
    partial = summarize_baseline_dir(missing_dir, allow_incomplete=True)
    assert partial["complete"] is False
    assert partial["missing_ranks"] == [1]

    # 混入不同 run_id 的另一个 rank 日志时必须报错。
    t1_other = ConditionTracker(
        rank=1, world_size=2, micro_batches=1, dataset_size=2, output_dir=missing_dir, warmup_steps=0, run_id="run-b"
    )
    t1_other.begin_step(0, torch.device("cpu"))
    t1_other.begin_micro(0, 1, row, config, "lora", 64, 64, is_padding=False)
    t1_other.end_step(0, 0.001, 0.01)
    t1_other.save_summary()
    with pytest.raises(ValueError, match="different runs"):
        summarize_baseline_dir(missing_dir, allow_incomplete=True)

    # 两个 rank 的步骤范围不一致时严格模式必须报错。
    step_mismatch_dir = tmp_path / "step_mismatch"
    for rank, steps in ((0, [0, 1]), (1, [0])):
        tr = ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=step_mismatch_dir,
            warmup_steps=0,
            run_id="run-c",
        )
        for step in steps:
            tr.begin_step(step, torch.device("cpu"))
            tr.begin_micro(0, rank, row, config, "lora", 64, 64, is_padding=False)
            tr.end_step(step, 0.001, 0.01)
        tr.save_summary()
    with pytest.raises(ValueError, match="step mismatch"):
        summarize_baseline_dir(step_mismatch_dir)

    # 单 rank 缺失 summary.json 或 summary.total_steps 与 steps.jsonl 行数不符时严格模式必须报错。
    no_summary_dir = tmp_path / "missing_summary"
    tr_single = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=1,
        dataset_size=1,
        output_dir=no_summary_dir,
        warmup_steps=0,
        run_id="run-d",
    )
    tr_single.begin_step(0, torch.device("cpu"))
    tr_single.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
    tr_single.end_step(0, 0.001, 0.01)
    with pytest.raises(ValueError, match="missing rank summaries \\[0\\]"):
        summarize_baseline_dir(no_summary_dir)
    partial_single = summarize_baseline_dir(no_summary_dir, allow_incomplete=True)
    assert partial_single["complete"] is False
    assert partial_single["missing_summaries"] == [0]

    tr_single.save_summary()
    summary_path = no_summary_dir / "rank_0_summary.json"
    summary_path.write_text(summary_path.read_text(encoding="utf-8").replace('"total_steps": 1', '"total_steps": 2'))
    with pytest.raises(ValueError, match="summary total_steps mismatch"):
        summarize_baseline_dir(no_summary_dir)


def test_tracker_io_error_does_not_abort_step_or_mask_training_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}
    tracker = ConditionTracker(
        rank=0, world_size=1, micro_batches=1, dataset_size=1, output_dir=tmp_path / "io_err", warmup_steps=0
    )
    tracker.begin_step(0, torch.device("cpu"))
    tracker.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
    original_open = Path.open

    def fail_jsonl(self: Path, *args, **kwargs):
        if self == tracker.steps_jsonl_path:
            raise OSError("disk full")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_jsonl)
    tracker.end_step(0, 0.001, 0.01)
    assert tracker.io_failed is True
    assert "disk full" in (tracker.io_error or "")
    summary = tracker.save_summary()
    assert summary["complete"] is False

    # 清理阶段 save_summary 抛出异常时不得覆盖原始训练异常，且仍须关闭 condition_cache。
    trainer = sft_module.FSDPSFTTrainer.__new__(sft_module.FSDPSFTTrainer)
    trainer.prefetch = Mock()
    trainer.tracker = Mock()
    trainer.tracker.save_summary.side_effect = OSError("summary disk full")
    trainer.condition_cache = Mock()
    trainer.condition_cache.stats.return_value = {}
    trainer.rank = 0
    original = RuntimeError("original training failure")
    trainer._train_step = Mock(side_effect=original)
    with pytest.raises(RuntimeError) as raised:
        trainer.train(0)
    assert raised.value is original
    trainer.prefetch.close.assert_called_once()
    trainer.condition_cache.close.assert_called_once()


def test_summarize_baseline_dir_reports_authoritative_condition_cache_rates(tmp_path: Path) -> None:
    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}
    cache_dir = tmp_path / "cache_rates"
    for rank, stats in (
        (
            0,
            {
                "requests": 4,
                "lookups": 4,
                "local_hits": 3,
                "effective_reuses": 2,
                "group_forced_online": 1,
                "local_hit_rate": 0.75,
                "effective_reuse_rate": 0.5,
            },
        ),
        (
            1,
            {
                "requests": 4,
                "lookups": 4,
                "local_hits": 2,
                "effective_reuses": 2,
                "group_forced_online": 0,
                "local_hit_rate": 0.5,
                "effective_reuse_rate": 0.5,
            },
        ),
    ):
        tr = ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=cache_dir,
            warmup_steps=0,
            run_id="cache-run",
        )
        tr.begin_step(0, torch.device("cpu"))
        tr.begin_micro(0, rank, row, config, "lora", 64, 64, is_padding=False)
        tr.condition_cache_stats = stats
        tr.end_step(0, 0.001, 0.01)
        tr.save_summary()
    agg = summarize_baseline_dir(cache_dir)
    assert agg["complete"] is True
    assert agg["condition_cache_summary"] == {
        "ranks_covered": [0, 1],
        "requests": 8,
        "lookups": 8,
        "local_hits": 5,
        "effective_reuses": 4,
        "group_forced_online": 1,
        "local_hit_rate": 5 / 8,
        "effective_reuse_rate": 4 / 8,
    }
    for enc_stats in agg["encoding_stats"].values():
        assert "cache_hit_rate" not in enc_stats


def test_expected_step_range_detects_early_exit(tmp_path: Path) -> None:
    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}

    # 场景 1：显式传入 complete=False 标记提前结束。
    explicit_dir = tmp_path / "explicit_early"
    explicit_summaries = []
    for rank in range(2):
        tr = ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=explicit_dir,
            warmup_steps=0,
            run_id="run-explicit",
            expected_step_range=(0, 2),
        )
        tr.begin_step(0, torch.device("cpu"))
        tr.begin_micro(0, rank, row, config, "lora", 64, 64, is_padding=False)
        tr.end_step(0, 0.001, 0.01)
        explicit_summaries.append(tr.save_summary(complete=False))

    for summary in explicit_summaries:
        assert summary["complete"] is False
        assert summary["expected_step_range"] == [0, 2]
        assert summary["terminated_early"] is True

    with pytest.raises(ValueError, match="Incomplete baseline records"):
        summarize_baseline_dir(explicit_dir)
    partial_explicit = summarize_baseline_dir(explicit_dir, allow_incomplete=True)
    assert partial_explicit["complete"] is False
    assert partial_explicit["expected_step_range"] == [0, 2]
    assert partial_explicit["incomplete_summaries"] == [0, 1]
    assert partial_explicit["expected_range_mismatches"] == [0, 1]

    # 场景 2：未显式传入 complete 时，仅靠 expected_step_range 未走完触发不完整。
    implicit_dir = tmp_path / "implicit_early"
    implicit_summaries = []
    for rank in range(2):
        tr = ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=implicit_dir,
            warmup_steps=0,
            run_id="run-implicit",
            expected_step_range=(0, 2),
        )
        tr.begin_step(0, torch.device("cpu"))
        tr.begin_micro(0, rank, row, config, "lora", 64, 64, is_padding=False)
        tr.end_step(0, 0.001, 0.01)
        implicit_summaries.append(tr.save_summary())

    for summary in implicit_summaries:
        assert summary["complete"] is False
        assert summary["expected_step_range"] == [0, 2]
        assert "terminated_early" not in summary

    with pytest.raises(ValueError, match="expected_step_range"):
        summarize_baseline_dir(implicit_dir)
    partial_implicit = summarize_baseline_dir(implicit_dir, allow_incomplete=True)
    assert partial_implicit["complete"] is False
    assert partial_implicit["expected_step_range"] == [0, 2]
    assert partial_implicit["expected_range_mismatches"] == [0, 1]

    # 非法 expected_step_range 初始化应抛出 ValueError。
    for bad_range in ((2, 1), (-1, 2), (True, 2)):
        with pytest.raises(ValueError, match="expected_step_range"):
            ConditionTracker(
                rank=0,
                world_size=1,
                micro_batches=1,
                dataset_size=1,
                output_dir=tmp_path / f"bad_range_{bad_range[0]}_{bad_range[1]}",
                expected_step_range=bad_range,  # type: ignore[arg-type]
            )

    # 多个 rank 声明不一致的 expected_step_range 时必须报错。
    disagree_dir = tmp_path / "disagree_range"
    for rank, step_range in ((0, (0, 1)), (1, (0, 2))):
        tr = ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=disagree_dir,
            warmup_steps=0,
            run_id="run-disagree",
            expected_step_range=step_range,
        )
        tr.begin_step(0, torch.device("cpu"))
        tr.begin_micro(0, rank, row, config, "lora", 64, 64, is_padding=False)
        tr.end_step(0, 0.001, 0.01)
        tr.save_summary()
    with pytest.raises(ValueError, match="disagree on expected_step_range"):
        summarize_baseline_dir(disagree_dir, allow_incomplete=True)


def test_expected_step_range_supports_resume_window(tmp_path: Path) -> None:
    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}
    resume_dir = tmp_path / "resume"

    for rank in range(2):
        tr = ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=resume_dir,
            warmup_steps=0,
            run_id="run-resume",
            expected_step_range=(2, 4),
        )
        for step in (2, 3):
            tr.begin_step(step, torch.device("cpu"))
            tr.begin_micro(0, rank, row, config, "lora", 64, 64, is_padding=False)
            tr.end_step(step, 0.001, 0.01)
        summary = tr.save_summary()
        assert summary["complete"] is True
        assert summary["expected_step_range"] == [2, 4]
        assert summary["total_steps"] == 2

    agg = summarize_baseline_dir(resume_dir)
    assert agg["complete"] is True
    assert agg["expected_step_range"] == [2, 4]
    assert agg["total_steps"] == 2


def test_tracker_refuses_to_overwrite_existing_rank_files(tmp_path: Path) -> None:
    steps_dir = tmp_path / "existing_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    steps_path = steps_dir / "rank_0_steps.jsonl"
    existing_steps_content = '{"rank": 0, "step": 99}\n'
    steps_path.write_text(existing_steps_content, encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exist"):
        ConditionTracker(rank=0, world_size=1, micro_batches=1, dataset_size=1, output_dir=steps_dir)
    assert steps_path.read_text(encoding="utf-8") == existing_steps_content
    assert [p.name for p in steps_dir.iterdir()] == ["rank_0_steps.jsonl"]

    summary_dir = tmp_path / "existing_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "rank_0_summary.json"
    existing_summary_content = '{"rank": 0, "complete": true}\n'
    summary_path.write_text(existing_summary_content, encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exist"):
        ConditionTracker(rank=0, world_size=1, micro_batches=1, dataset_size=1, output_dir=summary_dir)
    assert summary_path.read_text(encoding="utf-8") == existing_summary_content
    assert [p.name for p in summary_dir.iterdir()] == ["rank_0_summary.json"]


def test_summarize_baseline_dir_handles_truncated_and_corrupted_files(tmp_path: Path) -> None:
    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}

    # 1. 步骤日志末尾存在截断 JSON 行。
    trunc_dir = tmp_path / "truncated_tail"
    tr_trunc = ConditionTracker(
        rank=0, world_size=1, micro_batches=1, dataset_size=1, output_dir=trunc_dir, warmup_steps=0, run_id="run-trunc"
    )
    tr_trunc.begin_step(0, torch.device("cpu"))
    tr_trunc.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
    tr_trunc.end_step(0, 0.001, 0.01)
    tr_trunc.save_summary()
    steps_file = trunc_dir / "rank_0_steps.jsonl"
    steps_file.write_text(steps_file.read_text(encoding="utf-8") + '{"rank": 0, "step": 1', encoding="utf-8")

    with pytest.raises(ValueError, match="Truncated trailing record"):
        summarize_baseline_dir(trunc_dir)
    partial_trunc = summarize_baseline_dir(trunc_dir, allow_incomplete=True)
    assert partial_trunc["complete"] is False
    assert partial_trunc["total_steps"] == 1
    assert partial_trunc["truncated_step_logs"] == {"0": {"file": "rank_0_steps.jsonl", "line": 2}}

    # 2. 步骤日志中间行损坏时，allow_incomplete=True 仍必须抛出 ValueError。
    mid_dir = tmp_path / "corrupted_middle"
    tr_mid = ConditionTracker(
        rank=0, world_size=1, micro_batches=1, dataset_size=1, output_dir=mid_dir, warmup_steps=0, run_id="run-mid"
    )
    for step in (0, 1):
        tr_mid.begin_step(step, torch.device("cpu"))
        tr_mid.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
        tr_mid.end_step(step, 0.001, 0.01)
    tr_mid.save_summary()
    mid_lines = (mid_dir / "rank_0_steps.jsonl").read_text(encoding="utf-8").splitlines()
    (mid_dir / "rank_0_steps.jsonl").write_text(
        "\n".join([mid_lines[0], '{"rank": 0, "broken": ', mid_lines[1]]) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Corrupted record"):
        summarize_baseline_dir(mid_dir)
    with pytest.raises(ValueError, match="Corrupted record"):
        summarize_baseline_dir(mid_dir, allow_incomplete=True)

    # 3. rank_0_summary.json 损坏为半截 JSON。
    bad_sum_dir = tmp_path / "corrupted_summary"
    tr_bad_sum = ConditionTracker(
        rank=0, world_size=1, micro_batches=1, dataset_size=1, output_dir=bad_sum_dir, warmup_steps=0, run_id="run-sum"
    )
    tr_bad_sum.begin_step(0, torch.device("cpu"))
    tr_bad_sum.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
    tr_bad_sum.end_step(0, 0.001, 0.01)
    tr_bad_sum.save_summary()
    (bad_sum_dir / "rank_0_summary.json").write_text('{"rank": 0, "complete": ', encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted rank summary"):
        summarize_baseline_dir(bad_sum_dir)
    partial_sum = summarize_baseline_dir(bad_sum_dir, allow_incomplete=True)
    assert partial_sum["complete"] is False
    assert partial_sum["total_steps"] == 1
    assert partial_sum["sample_stats"]["total_samples"] == 1
    assert partial_sum["corrupted_summaries"] == {"0": "rank_0_summary.json"}
    assert partial_sum["missing_summaries"] == [0]


def test_atomic_summary_writes_leave_no_tmp_files_and_preserve_existing_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}

    # 正常写入后目录中无临时文件残留。
    clean_dir = tmp_path / "atomic_clean"
    tr = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=1,
        dataset_size=1,
        output_dir=clean_dir,
        warmup_steps=0,
        run_id="run-atomic",
    )
    tr.begin_step(0, torch.device("cpu"))
    tr.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
    tr.end_step(0, 0.001, 0.01)
    tr.save_summary()
    summarize_baseline_dir(clean_dir)
    assert list(clean_dir.glob("*.tmp*")) == []

    # 写入失败时应清理临时文件并保留原有文件内容。
    sentinel_baseline = '{"sentinel": "baseline"}\n'
    baseline_path = clean_dir / "baseline_summary.json"
    baseline_path.write_text(sentinel_baseline, encoding="utf-8")
    orig_rank_summary = (clean_dir / "rank_0_summary.json").read_text(encoding="utf-8")

    def fail_replace(_src: object, _dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        summarize_baseline_dir(clean_dir)
    assert baseline_path.read_text(encoding="utf-8") == sentinel_baseline
    assert list(clean_dir.glob("*.tmp*")) == []

    tr.condition_cache_stats = {"requests": 1}
    failed_summary = tr.save_summary()
    assert failed_summary["complete"] is False
    assert tr.io_failed is True
    assert "simulated replace failure" in (tr.io_error or "")
    assert (clean_dir / "rank_0_summary.json").read_text(encoding="utf-8") == orig_rank_summary
    assert list(clean_dir.glob("*.tmp*")) == []


def test_summarize_baseline_dir_handles_utf8_truncation_and_terminated_corrupt_tail(tmp_path: Path) -> None:
    row = {"prompt": "p", "image": "t.png", "edit_image": ["r.png"]}
    config = {"max_pixels": 65536, "fixed_pixels_tiers": None, "seed": 42}

    # 1. 实际 tracker 写入两步，第二步传输键截断在 UTF-8 多字节字符的第一个字节之后（无换行符）。
    utf8_dir = tmp_path / "truncated_utf8"
    tr_utf8 = ConditionTracker(
        rank=0, world_size=1, micro_batches=1, dataset_size=1, output_dir=utf8_dir, warmup_steps=0, run_id="run-utf8"
    )
    transfer_key = "参考图/样本.png"
    for step in (0, 1):
        tr_utf8.begin_step(step, torch.device("cpu"))
        tr_utf8.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
        tr_utf8.record_transfer("ref_pixel_values", torch.ones((1, 4)), key=transfer_key)
        tr_utf8.end_step(step, 0.001, 0.01)
    tr_utf8.save_summary()

    steps_path = utf8_dir / "rank_0_steps.jsonl"
    raw_lines = steps_path.read_bytes().splitlines(keepends=True)
    assert len(raw_lines) == 2
    key_bytes = transfer_key.encode("utf-8")
    key_offset = raw_lines[1].index(key_bytes)
    steps_path.write_bytes(raw_lines[0] + raw_lines[1][: key_offset + 1])

    with pytest.raises(ValueError, match="Truncated trailing record"):
        summarize_baseline_dir(utf8_dir)
    partial_utf8 = summarize_baseline_dir(utf8_dir, allow_incomplete=True)
    assert partial_utf8["complete"] is False
    assert partial_utf8["total_steps"] == 1
    assert partial_utf8["truncated_step_logs"] == {"0": {"file": "rank_0_steps.jsonl", "line": 2}}

    # 2. 带有换行符的损坏末行（JSON 损坏或 UTF-8 损坏）在部分模式下仍须拒绝。
    for suffix_name, corrupt_tail in (
        ("json_newline", b'{"rank": 0, "step": 1\n'),
        ("utf8_newline", b'{"rank": 0, "key": "\xe5\n'),
    ):
        tail_dir = tmp_path / f"corrupt_{suffix_name}"
        tr_tail = ConditionTracker(
            rank=0,
            world_size=1,
            micro_batches=1,
            dataset_size=1,
            output_dir=tail_dir,
            warmup_steps=0,
            run_id=f"run-{suffix_name}",
        )
        tr_tail.begin_step(0, torch.device("cpu"))
        tr_tail.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
        tr_tail.end_step(0, 0.001, 0.01)
        tr_tail.save_summary()
        tail_file = tail_dir / "rank_0_steps.jsonl"
        tail_file.write_bytes(tail_file.read_bytes() + corrupt_tail)

        with pytest.raises(ValueError, match="Corrupted record"):
            summarize_baseline_dir(tail_dir)
        with pytest.raises(ValueError, match="Corrupted record"):
            summarize_baseline_dir(tail_dir, allow_incomplete=True)

    # 3. 中间行包含非法 UTF-8 字节时，严格与部分模式均须抛出 ValueError。
    mid_utf8_dir = tmp_path / "corrupted_middle_utf8"
    tr_mid_utf8 = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=1,
        dataset_size=1,
        output_dir=mid_utf8_dir,
        warmup_steps=0,
        run_id="run-mid-utf8",
    )
    for step in (0, 1):
        tr_mid_utf8.begin_step(step, torch.device("cpu"))
        tr_mid_utf8.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
        tr_mid_utf8.end_step(step, 0.001, 0.01)
    tr_mid_utf8.save_summary()
    mid_lines = (mid_utf8_dir / "rank_0_steps.jsonl").read_bytes().splitlines(keepends=True)
    assert len(mid_lines) == 2
    (mid_utf8_dir / "rank_0_steps.jsonl").write_bytes(mid_lines[0] + b'{"rank": 0, "key": "\xe5"}\n' + mid_lines[1])

    with pytest.raises(ValueError, match="Corrupted record"):
        summarize_baseline_dir(mid_utf8_dir)
    with pytest.raises(ValueError, match="Corrupted record"):
        summarize_baseline_dir(mid_utf8_dir, allow_incomplete=True)

    # 4. rank_0_summary.json 截断在 UTF-8 多字节字符中间时，部分模式应跳过损坏摘要并返回部分汇总。
    bad_sum_utf8_dir = tmp_path / "corrupted_summary_utf8"
    tr_sum_utf8 = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=1,
        dataset_size=1,
        output_dir=bad_sum_utf8_dir,
        warmup_steps=0,
        run_id="run-sum-utf8",
    )
    tr_sum_utf8.begin_step(0, torch.device("cpu"))
    tr_sum_utf8.begin_micro(0, 0, row, config, "lora", 64, 64, is_padding=False)
    tr_sum_utf8.end_step(0, 0.001, 0.01)
    tr_sum_utf8.save_summary()
    (bad_sum_utf8_dir / "rank_0_summary.json").write_bytes(b'{"rank": 0, "io_error": "\xe5')

    with pytest.raises(ValueError, match="Corrupted rank summary"):
        summarize_baseline_dir(bad_sum_utf8_dir)
    partial_sum_utf8 = summarize_baseline_dir(bad_sum_utf8_dir, allow_incomplete=True)
    assert partial_sum_utf8["complete"] is False
    assert partial_sum_utf8["total_steps"] == 1
    assert partial_sum_utf8["sample_stats"]["total_samples"] == 1
    assert partial_sum_utf8["corrupted_summaries"] == {"0": "rank_0_summary.json"}
    assert partial_sum_utf8["missing_summaries"] == [0]
