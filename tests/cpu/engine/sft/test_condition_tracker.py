# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import json
import os
import subprocess
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
from relax.engine.sft.image import ImageBatchSampler, assign_image_batch


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

    # 运行 2 个 epoch（每个 epoch 2 个 step，共 4 个 step，消费 16 个样本位置）
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
                        output_bytes=len(edit_list) * 1024,
                    )
                with tracker.gpu_stage("prefix_encode_ms"):
                    tracker.record_encode(
                        "prefix_encode",
                        num_items=1,
                        output_bytes=4096,
                        indexes_image_bytes=128,
                    )

                with tracker.host_stage("target_image_preprocess_ms"):
                    latents = torch.zeros((1, 16, 8, 8), dtype=torch.bfloat16)
                    tracker.record_transfer("target_input_latents", latents)

                with tracker.gpu_stage("gen_vision_encode_ms"):
                    tracker.record_encode("gen_vision_encode", num_items=1, output_bytes=2048, eligible=False)
                with tracker.gpu_stage("dit_forward_ms"):
                    pass
                with tracker.gpu_stage("backward_ms"):
                    pass

            with tracker.gpu_stage("optimizer_ms"):
                pass
            tracker.end_step(step, data_wait_seconds=0.002, train_seconds=0.020)

    summaries = [t.save_summary() for t in trackers]
    agg = summarize_baseline_dir(tmp_path)

    # 4 个 step * global_batch_size 4 = 16 个消费位置；唯一行 5 个；每个 epoch 补齐 8 - 5 = 3 个，2 个 epoch 共 6 个
    assert agg["sample_stats"]["total_samples"] == 16
    assert agg["sample_stats"]["global_unique_rows"] == 5
    assert agg["sample_stats"]["padding_samples"] == 6
    assert sum(s["sample_stats"]["padding_samples"] for s in summaries) == 6
    assert agg["sample_stats"]["global_repeated_samples"] == 16 - 5

    # 行 0 和行 2 共享相同的 (prompt, edit_image, height, width)，因此全局唯一条件数为 4
    ref_enc = agg["encoding_stats"]["ref_vision_encode"]
    prefix_enc = agg["encoding_stats"]["prefix_encode"]
    gen_enc = agg["encoding_stats"]["gen_vision_encode"]

    assert ref_enc["eligible"] is True
    assert ref_enc["calls"] == 16
    assert ref_enc["global_repeated_calls"] == 16 - 4
    assert ref_enc["cache_lookups"] == 0
    assert ref_enc["cache_hits"] == 0
    assert ref_enc["cache_hit_rate"] == 0.0

    # 当处理 2 张参考图（行 0 和行 2）时，encoded_images 按实际张数累加
    all_consumed_ids = [sid for step in range(4) for sid in sampler.batch(step)]
    expected_encoded_images = sum(len(rows[sid]["edit_image"]) for sid in all_consumed_ids)
    assert ref_enc["encoded_images"] == expected_encoded_images

    assert prefix_enc["eligible"] is True
    assert prefix_enc["calls"] == 16
    assert prefix_enc["global_repeated_calls"] == 16 - 4
    assert prefix_enc["cache_lookups"] == 0
    assert prefix_enc["cache_hits"] == 0
    assert prefix_enc["cache_hit_rate"] == 0.0

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

    for step in range(4):
        raw_ids = sampler.batch(step)
        balanced_ids = assign_image_batch(raw_ids, config, step=step, world=world_size)
        for r in range(world_size):
            trackers[r].begin_step(step, torch.device("cpu"))
            for micro, sid in enumerate(balanced_ids[r::world_size]):
                trackers[r].begin_micro(micro, sid, rows[sid], config, "lora", 64, 64)
            trackers[r].end_step(step, 0.001, 0.01)

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

    # 第 0 步：首次出现
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
        "ref_vision_encode", num_items=1, output_bytes=vision_embeds.numel() * vision_embeds.element_size()
    )
    tracker.record_encode(
        "prefix_encode",
        num_items=1,
        output_bytes=compute_past_key_values_bytes(kv_cache),
        indexes_image_bytes=int(indexes_image.numel() * indexes_image.element_size()),
    )
    tracker.end_step(0, 0.001, 0.01)

    del bf16_tensor
    del kv_k
    del kv_v
    del kv_cache
    assert ref_bf16() is None
    assert ref_kv() is None

    # 第 1 步：相同样本再次出现，验证 repeated_transfer_bytes
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

    # Step 0 开始前初始化显存峰值为 5000 / 8000，Step 0 运行中单步峰值为 3000 / 6000
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

    # Step 1 单步峰值升至 7000 / 9000，历史峰值同步更新为 7000 / 9000
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
                    tracker.record_encode("ref_vision_encode", 1, output_bytes=64)
                with tracker.gpu_stage("prefix_encode_ms"):
                    tracker.record_encode("prefix_encode", 1, output_bytes=128, indexes_image_bytes=16)
                with tracker.gpu_stage("gen_vision_encode_ms"):
                    tracker.record_encode("gen_vision_encode", 1, output_bytes=64, eligible=False)
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


def test_validate_i2i_sft_config_and_launcher_baseline_env(tmp_path: Path) -> None:
    base_args = Namespace(
        loss_type="sft",
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
        num_rollout=4,
        num_epoch=None,
        seed=42,
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

    # 验证启动脚本转发 BASELINE_STATS_DIR 与 BASELINE_WARMUP_STEPS
    root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.update(
        MODEL_PATH="/model",
        DATA_ROOT="/data",
        METADATA_PATH="/metadata",
        DATASET_SIZE="8",
        GPUS="2",
        GPUS_PER_NODE="2",
        GLOBAL_BATCH_SIZE="4",
        BASELINE_STATS_DIR=str(tmp_path / "run_stats"),
        BASELINE_WARMUP_STEPS="5",
    )
    command = """
ray() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
export -f ray
bash "$1"
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "capture",
            str(root / "scripts/training/diffusion/run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    cli_args = json.loads(result.stdout)
    i2i_cfg = json.loads(cli_args[cli_args.index("--i2i-sft-config") + 1])
    assert i2i_cfg["baseline_stats_dir"] == str(tmp_path / "run_stats")
    assert i2i_cfg["baseline_warmup_steps"] == 5


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

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
