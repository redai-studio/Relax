# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contracts for the pre-merger adapter; real Megatron GPU validation is
separate."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from relax.utils.multimodal.vision_feature_cache import (
    VISION_CACHE_INFO,
    FeatureLRU,
    _Qwen35DecoderCache,
    attach_vision_cache_info,
    clear_train_vision_cache,
    install_train_vision_cache,
)


class Config:
    def __init__(self, tp=1, pp=1, cp=1, sp=False, vision_dp=False):
        self.tensor_model_parallel_size = tp
        self.pipeline_model_parallel_size = pp
        self.context_parallel_size = cp
        self.sequence_parallel = sp
        self.vision_dp_when_cp = vision_dp


class Group:
    def __init__(self, size=1, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.config = SimpleNamespace(hidden_dropout=0, attention_dropout=0, sequence_parallel=False)
        self.deepstack_visual_indexes = []
        self.tp_group = None
        self.calls = 0
        self.rows = []

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None, packed_seq_params=None):
        self.calls += 1
        self.rows.append(len(hidden_states))
        # Independent image attention, so re-packing misses must retain boundaries.
        chunks = hidden_states.split(packed_seq_params.lengths)
        output = torch.cat([self.linear(x + x.mean(dim=0)).tanh() for x in chunks])
        if self.tp_group is not None:
            import torch.distributed as dist

            dist.all_reduce(output, group=self.tp_group)
            output /= dist.get_world_size(self.tp_group)
        return output, []


class Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(4, 8)
        self.pos_embed = torch.nn.Embedding(128, 8)
        self.decoder = Decoder()
        self.merger = torch.nn.Linear(8, 4)
        for module in (self.patch_embed, self.pos_embed, self.decoder):
            module.requires_grad_(False)

    def build_packed_seq_params(self, grid):
        return SimpleNamespace(lengths=[int(x.prod()) for x in grid])

    def forward(self, pixels, grid):
        lengths = [int(x.prod()) for x in grid]
        positions = torch.cat([torch.arange(x) for x in lengths])
        hidden = (self.patch_embed(pixels) + self.pos_embed(positions))[:, None]
        hidden, _ = self.decoder(
            hidden_states=hidden,
            attention_mask=None,
            rotary_pos_emb=torch.zeros(len(hidden), 1, 1, 2),
            packed_seq_params=self.build_packed_seq_params(grid),
        )
        return self.merger(hidden)


class Model(torch.nn.Module):
    def __init__(self, config=None, cp_group=None):
        super().__init__()
        self.config = config or Config()
        self.pg_collection = SimpleNamespace(cp=cp_group or Group())
        self.vision_model = Vision()
        self.language = torch.nn.Linear(4, 2)
        self.use_dist_train = False

    def forward(self, pixel_values, image_grid_thw):
        return self.language(self.vision_model(pixel_values, image_grid_thw))


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("RELAX_TRAIN_VISION_CACHE", "1")
    monkeypatch.setenv("RELAX_TRAIN_VISION_CACHE_MAX_BYTES", "4096")
    provider = "megatron.bridge.models.qwen_vl.qwen35_vl_provider"
    monkeypatch.setitem(sys.modules, provider, SimpleNamespace(Qwen35VLModelProvider=Config))


def inputs(values, session="session", grid=None):
    pixels = torch.cat([torch.full((4, 4), float(x)) for x in values])
    grid = grid or [(1, 2, 2)] * len(values)
    return attach_vision_cache_info(session, {"pixel_values": pixels, "image_grid_thw": torch.tensor(grid)})


def vanilla(model, data):
    return model(**{k: v for k, v in data.items() if k != VISION_CACHE_INFO})


def test_partial_hits_preserve_outputs_projection_and_language_gradients(enabled):
    torch.manual_seed(1)
    reference = Model()
    cached = copy.deepcopy(reference)
    install_train_vision_cache(cached)
    original_config = vars(cached.config).copy()
    for values in ([1, 2], [2, 3], [1, 3, 1]):
        data = inputs(values)
        expected = vanilla(reference, data)
        actual = cached(**data)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, a), (_, b) in zip(reference.named_parameters(), cached.named_parameters(), strict=True):
            if a.requires_grad:
                assert a.grad is not None, name
                torch.testing.assert_close(a.grad, b.grad)
                # Exercise a changed trainable projection; frozen features remain reusable.
                with torch.no_grad():
                    a.add_(a.grad, alpha=-0.01)
                    b.add_(b.grad, alpha=-0.01)
            else:
                assert a.grad is None and b.grad is None
        reference.zero_grad(set_to_none=True)
        cached.zero_grad(set_to_none=True)
    assert cached.vision_model.decoder.rows == [8, 4]
    assert vars(cached.config) == original_config
    assert cached._relax_train_vision_cache.cache.hits == 4
    assert all(
        not x.requires_grad and x.grad_fn is None for x in cached._relax_train_vision_cache.cache.entries.values()
    )


def test_session_grid_pixels_and_weight_changes_do_not_reuse_stale_features(enabled):
    model = Model()
    install_train_vision_cache(model)
    data = inputs([1])
    model(**data)
    model(**data)
    model(**inputs([1], session="other"))
    model(**inputs([1], grid=[(1, 1, 4)]))
    model(**inputs([2]))
    with torch.no_grad():
        model.vision_model.decoder.linear.weight.add_(1)
    model(**inputs([2]))
    assert model.vision_model.decoder.calls == 5
    model.load_state_dict(model.state_dict())
    assert model._relax_train_vision_cache.cache.nbytes == 0
    model(**data)
    clear_train_vision_cache([model])
    assert model._relax_train_vision_cache.cache.nbytes == 0


def test_lru_owns_storage_and_enforces_byte_budget():
    cache = FeatureLRU(64)
    source = torch.ones(8)
    cache.put("a", source)
    source.zero_()
    assert cache.get("a").sum() == 8
    cache.put("b", torch.ones(8))
    cache.put("c", torch.ones(8))
    assert list(cache.entries) == ["b", "c"] and cache.nbytes == 64
    cache.put("large", torch.ones(100))
    assert cache.nbytes == 64 and cache.get("large") is None


def test_metadata_is_generic_and_does_not_mutate_pixels():
    source = {"pixel_values": torch.ones(4, 4, dtype=torch.bfloat16), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    result = attach_vision_cache_info("session", source)
    assert VISION_CACHE_INFO not in source
    assert result["pixel_values"] is source["pixel_values"]
    assert result[VISION_CACHE_INFO][0]["grid"] == (1, 2, 2)
    assert attach_vision_cache_info(None, source) is source
    video = {**source, "pixel_values_videos": torch.ones(4, 4)}
    assert attach_vision_cache_info("session", video) is video


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("cp_size,cp_rank,pp", [(1, 0, 1), (2, 0, 2), (2, 1, 1), (4, 3, 2)])
def test_live_cp_metadata_and_language_sp_are_preserved(enabled, sp, cp_size, cp_rank, pp):
    model = Model(Config(tp=2, pp=pp, cp=cp_size, sp=sp, vision_dp=True), Group(cp_size, cp_rank))
    adapter = _Qwen35DecoderCache(model, 4096)
    infos = inputs([1, 2, 3])[VISION_CACHE_INFO]
    local = adapter.local_images(infos, {})
    expected = {(1, 0): infos, (2, 0): infos[:2], (2, 1): infos[2:], (4, 3): []}
    assert local == expected[cp_size, cp_rank]
    assert model.config.sequence_parallel is sp
    assert model.config.pipeline_model_parallel_size == pp


def test_default_off_and_pp_without_encoder(enabled, monkeypatch):
    monkeypatch.setenv("RELAX_TRAIN_VISION_CACHE", "0")
    model = Model()
    install_train_vision_cache(model)
    assert not hasattr(model, "_relax_train_vision_cache")
    monkeypatch.setenv("RELAX_TRAIN_VISION_CACHE", "1")
    model.vision_model = None
    model.forward = lambda pixel_values, image_grid_thw: pixel_values
    install_train_vision_cache(model)
    assert not hasattr(model, "_relax_train_vision_cache")
    data = inputs([1])
    torch.testing.assert_close(model(**data), data["pixel_values"])


@pytest.mark.parametrize("case", ["trainable", "deepstack", "dropout", "fp8", "graph", "vision_sp", "disaggregate"])
def test_unsupported_computation_fails_before_reuse(enabled, case):
    model = Model()
    if case == "trainable":
        model.vision_model.decoder.linear.requires_grad_(True)
    if case == "deepstack":
        model.vision_model.decoder.deepstack_visual_indexes = [0]
    if case == "dropout":
        model.vision_model.decoder.config.hidden_dropout = 0.1
    if case == "fp8":
        model.vision_model.decoder.config.fp8 = "hybrid"
    if case == "graph":
        model.vision_model.decoder.config.cuda_graph_impl = "transformer_engine"
    if case == "vision_sp":
        model.vision_model.decoder.config.sequence_parallel = True
    if case == "disaggregate":
        model.use_dist_train = True
    with pytest.raises(ValueError):
        install_train_vision_cache(model)


def _tp_worker(rank, path, tp, sp):
    """Real collectives exercise divergent rank caches and independent DP
    replicas."""
    import os
    from datetime import timedelta

    import torch.distributed as dist

    os.environ["RELAX_TRAIN_VISION_CACHE"] = "1"
    sys.modules["megatron.bridge.models.qwen_vl.qwen35_vl_provider"] = SimpleNamespace(Qwen35VLModelProvider=Config)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    try:
        group = None
        for start in range(0, 4, tp):
            candidate = dist.new_group(list(range(start, start + tp)), backend="gloo")
            if start <= rank < start + tp:
                group = candidate
        torch.manual_seed(10 + rank // tp)
        reference = Model(Config(tp=tp, sp=sp))
        model = copy.deepcopy(reference)
        model.vision_model.decoder.tp_group = group
        # Exercise creation of the CPU control channel used alongside NCCL.
        # Vision math/collectives below remain a CPU interface fixture.
        get_backend = dist.get_backend
        dist.get_backend = lambda candidate: "nccl" if candidate is group else get_backend(candidate)
        try:
            install_train_vision_cache(model)
        finally:
            dist.get_backend = get_backend
        adapter = model._relax_train_vision_cache
        data = inputs([rank // tp + 1])
        for _ in range(2):
            torch.testing.assert_close(model(**data), vanilla(reference, data))
        assert model.vision_model.decoder.calls == 1
        if dist.get_rank(group) == 0:
            adapter.cache.clear()  # Simulate rank-local eviction or invalidation.
        torch.testing.assert_close(model(**data), vanilla(reference, data))
        assert model.vision_model.decoder.calls == 2  # All TP members recompute.
        assert adapter.stats["tp_forced_misses"] == (0 if rank % tp == 0 else 1)
        assert adapter.stats["encoded_patches"] == 8
        assert len(adapter.cache.entries) == 1  # Other DP replicas never share entries.
        # Absent metadata on one TP rank must bypass on every member.
        maybe_bare = {k: v for k, v in data.items() if k != VISION_CACHE_INFO} if rank % tp == 0 else data
        torch.testing.assert_close(model(**maybe_bare), vanilla(reference, data))
        assert model.vision_model.decoder.calls == 3
        assert adapter.stats["bypass_calls"] == 1
        assert adapter.stats["encoded_patches"] == 12
        if tp > 1:
            differing = inputs([rank % tp + 1])
            with pytest.raises(ValueError, match="image order differs"):
                model(**differing)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,sp", [(1, False), (2, True), (4, True)])
def test_real_tp_consensus_and_dp_isolation(tmp_path, tp, sp):
    torch.multiprocessing.start_processes(
        _tp_worker, args=(str(tmp_path / "store"), tp, sp), nprocs=4, join=True, start_method="fork"
    )


def test_unknown_model_requires_an_explicit_adapter(enabled):
    model = Model()
    model.config = SimpleNamespace()
    with pytest.raises(ValueError, match="only for the Qwen3.5"):
        install_train_vision_cache(model)


def test_real_sample_conversion_and_multimodal_transfer(enabled, monkeypatch):
    import importlib
    from types import ModuleType

    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    parent = importlib.import_module("relax.utils.data")
    name = "relax.utils.data.stream_dataloader"
    monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    sys.modules.pop(name, None)
    monkeypatch.setattr(parent, "stream_dataloader", None, raising=False)
    stream_dataloader = importlib.import_module(name)
    from relax.utils.types import Sample
    from relax.utils.utils import convert_samples_to_train_data, dict_to_tensordict

    sample = Sample(
        tokens=[1, 2, 3], response_length=1, index=0, custom_advantage=0.5, metadata={"session_id": "generic-session"}
    )
    sample.multimodal_train_inputs = {k: v for k, v in inputs([1]).items() if k != VISION_CACHE_INFO}
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path="unused:advantage",
        advantage_estimator="grpo",
        debug_train_only=True,
        multimodal_keys=None,
        use_opd=False,
    )
    train = convert_samples_to_train_data(args, [sample])
    mm = train["multimodal_train_inputs"][0]
    assert mm[VISION_CACHE_INFO][0]["key"][0] == "generic-session"
    packed = dict_to_tensordict(train, 1)
    assert packed["multimodal_train_inputs"][0][VISION_CACHE_INFO] == mm[VISION_CACHE_INFO]
    spec, tensors = stream_dataloader._encode_multimodal_inputs([mm, None])
    monkeypatch.setattr(stream_dataloader, "_bcast_known_tensor", lambda src, *unused: src.clone())
    restored = stream_dataloader._broadcast_multimodal_inputs(spec, tensors, True, torch.device("cpu"), False)
    assert restored[0][VISION_CACHE_INFO] == mm[VISION_CACHE_INFO]
    assert restored[1] is None
    assert len(tensors) == 2  # Tiny cache metadata never becomes a training tensor.
    torch.testing.assert_close(restored[0]["pixel_values"], mm["pixel_values"])


@pytest.mark.parametrize("complete", [True, False])
def test_microbatch_packing_preserves_image_identity_order(enabled, monkeypatch, complete):
    import importlib
    from pathlib import Path
    from types import ModuleType

    from tests.backends.megatron.test_data_vpp import _load_data_module

    # Import data.py with existing CPU stubs, without executing GPU backend patches.
    parent = ModuleType("relax.backends.megatron")
    parent.__path__ = [
        str(Path(__file__).parents[2] / "relax/backends/megatron"),
        str(Path(_load_data_module.__code__.co_filename).parents[3] / "relax/backends/megatron"),
    ]
    monkeypatch.setitem(sys.modules, "relax.backends.megatron", parent)
    monkeypatch.setattr(importlib.import_module("relax.backends"), "megatron", parent, raising=False)
    module = _load_data_module(monkeypatch)
    monkeypatch.setattr(module, "get_args", lambda: SimpleNamespace())
    for name, value in [
        ("get_context_parallel_world_size", 1),
        ("get_context_parallel_rank", 0),
        ("get_tensor_model_parallel_world_size", 1),
    ]:
        monkeypatch.setattr(module.mpu, name, lambda value=value: value, raising=False)
    mm = [inputs([3, 1]), inputs([2], session="second")]
    if not complete:
        mm[1].pop(VISION_CACHE_INFO)
    batch = {
        "tokens": [[1, 2, 3], [4, 5]],
        "total_lengths": [3, 2],
        "response_lengths": [1, 1],
        "loss_masks": [[1], [1]],
        "max_seq_lens": [3],
        "multimodal_train_inputs": mm,
    }
    packed = module.get_batch(iter([(batch, None)]), ["tokens"], qkv_format="bshd", pack_device=torch.device("cpu"))
    combined = packed["multimodal_train_inputs"]
    torch.testing.assert_close(combined["pixel_values"], torch.cat([x["pixel_values"] for x in mm]))
    assert VISION_CACHE_INFO not in packed["multimodal_num_items"]
    if complete:
        assert combined[VISION_CACHE_INFO] == mm[0][VISION_CACHE_INFO] + mm[1][VISION_CACHE_INFO]
    else:
        assert VISION_CACHE_INFO not in combined  # Preserve full normal computation for incomplete metadata.


def _capture_cache_reports(monkeypatch):
    import json

    from relax.utils.multimodal import vision_feature_cache as module

    reports = []

    def capture(template, *args):
        message = template % args
        if " stats=" in message:
            reports.append((message, json.loads(message.split(" stats=", 1)[1])))

    monkeypatch.setattr(module.logger, "info", capture)
    return reports


def test_stats_distinguish_hits_batch_dedup_and_actual_decoder_work(enabled, monkeypatch):
    reports = _capture_cache_reports(monkeypatch)
    model = Model()
    install_train_vision_cache(model)

    def data(values):
        # B has twice as many patches: image counts alone cannot measure reuse.
        grids = [(1, 2, 4) if value == 2 else (1, 2, 2) for value in values]
        pixels = torch.cat(
            [torch.full((t * h * w, 4), float(value)) for value, (t, h, w) in zip(values, grids, strict=True)]
        )
        return attach_vision_cache_info("session", {"pixel_values": pixels, "image_grid_thw": torch.tensor(grids)})

    for values in ([1, 1, 2], [1, 2, 2], [2, 3, 3]):
        model(**data(values))
    assert model.vision_model.decoder.rows == [12, 4]
    clear_train_vision_cache(model, reason="offload")
    message, stats = reports[-1]
    assert "reason=offload" in message and "ranks=" in message
    assert stats["calls"] == stats["cache_calls"] == 3
    assert stats["image_refs"] == 9
    assert stats["hits"] == 4 and stats["misses"] == 5
    assert stats["batch_dedup_images"] == 2 and stats["encoded_images"] == 3
    assert stats["patch_refs"] == 52
    assert stats["cache_hit_patches"] == 28 and stats["batch_dedup_patches"] == 8
    assert stats["encoded_patches"] == sum(model.vision_model.decoder.rows) == 16
    assert stats["image_hit_ratio"] == pytest.approx(4 / 9)
    assert stats["image_skip_ratio"] == pytest.approx(6 / 9)
    assert stats["patch_skip_ratio"] == pytest.approx(36 / 52)
    assert stats["evictions"] == stats["oversized_skips"] == stats["bypass_calls"] == 0
    assert stats["entries"] == 3 and stats["bytes"] == stats["peak_bytes"] == 512
    clear_train_vision_cache(model)
    empty = reports[-1][1]
    assert empty["calls"] == empty["image_refs"] == empty["encoded_patches"] == empty["peak_bytes"] == 0
    assert empty["patch_skip_ratio"] == empty["image_skip_ratio"] == empty["image_hit_ratio"] == 0


def test_stats_include_uncached_bypass_work(enabled, monkeypatch):
    reports = _capture_cache_reports(monkeypatch)
    model = Model()
    install_train_vision_cache(model)
    data = inputs([1])
    model(**data)
    model(**data)
    vanilla(model, data)  # No metadata: executes the unchanged decoder.
    clear_train_vision_cache(model)
    stats = reports[-1][1]
    assert model.vision_model.decoder.rows == [4, 4]
    assert stats["calls"] == 3 and stats["cache_calls"] == 2
    assert stats["bypass_calls"] == 1 and stats["bypass_patches"] == 4
    assert stats["image_refs"] == 2 and stats["encoded_images"] == 1
    assert stats["patch_refs"] == 12 and stats["encoded_patches"] == 8
    assert stats["patch_skip_ratio"] == pytest.approx(1 / 3)


def test_stats_record_eviction_oversize_and_clear_reasons(enabled, monkeypatch):
    reports = _capture_cache_reports(monkeypatch)
    monkeypatch.setenv("RELAX_TRAIN_VISION_CACHE_MAX_BYTES", "128")  # One four-patch feature.
    model = Model()
    install_train_vision_cache(model)
    model(**inputs([1, 2, 3]))
    clear_train_vision_cache(model, reason="weight_switch:ref->actor")
    message, stats = reports[-1]
    assert "reason=weight_switch:ref->actor" in message
    assert stats["evictions"] == 2 and stats["entries"] == 1 and stats["peak_bytes"] == 128
    with torch.no_grad():
        model.vision_model.decoder.linear.weight.add_(1)
    model(**inputs([1]))
    assert "reason=encoder_state_changed" in reports[-1][0]
    model.load_state_dict(model.state_dict())
    assert "reason=load_state_dict" in reports[-1][0]
    assert reports[-1][1]["image_refs"] == 1

    monkeypatch.setenv("RELAX_TRAIN_VISION_CACHE_MAX_BYTES", "64")  # Cannot retain even one feature.
    oversized = Model()
    install_train_vision_cache(oversized)
    oversized(**inputs([1, 1]))
    oversized(**inputs([1]))
    clear_train_vision_cache(oversized)
    stats = reports[-1][1]
    assert stats["oversized_skips"] == 2 and stats["evictions"] == 0
    assert stats["encoded_images"] == 2 and stats["batch_dedup_images"] == 1
    assert stats["bytes"] == stats["peak_bytes"] == stats["entries"] == 0
