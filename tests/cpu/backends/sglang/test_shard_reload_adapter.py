# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise native cache semantics on CPU; factory/GPU layout has separate
proof."""

import pytest
import torch

from relax.backends.sglang.shard_p2p_reload import Qwen35ReloadAdapter


def _model():
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = GemmaRMSNorm(4, eps=1e-5)
            self.norm._weight_loader(self.norm.weight, torch.arange(4, dtype=self.norm.weight.dtype))

        def load_weights(self, values):
            raise AssertionError("Cache refresh must not reload unrelated weights")

    return Model()


def test_native_cache_refresh_preserves_storage_and_matches_loaded_weight():
    model = _model()
    adapter = Qwen35ReloadAdapter(type(model))
    state = adapter.bind(model)
    before = model.norm.gemma_weight.data_ptr()
    with torch.no_grad():
        model.norm.weight.add_(2)
    assert state.refresh({"norm.gemma_weight"}) == {"norm.gemma_weight"}
    assert model.norm.gemma_weight.data_ptr() == before
    torch.testing.assert_close(model.norm.gemma_weight, model.norm.weight.float() + 1, rtol=0, atol=0)


def test_native_reload_refuses_extra_cache_or_replaced_loader():
    model = _model()
    adapter = Qwen35ReloadAdapter(type(model))
    model.register_buffer("unexpected_cache", torch.ones(4))
    with pytest.raises(NotImplementedError, match="cache policy"):
        adapter.buffers(model)
    del model.unexpected_cache
    model.load_weights = lambda values: None
    with pytest.raises(NotImplementedError, match="load_weights hook"):
        adapter.buffers(model)


def test_cache_completion_cannot_omit_a_derived_buffer():
    model = _model()
    adapter = Qwen35ReloadAdapter(type(model))
    with pytest.raises(NotImplementedError, match="outside"):
        adapter.validate_written_buffers(model, set())
    adapter.validate_written_buffers(model, {"norm.gemma_weight"})


def test_rotary_cache_can_grow_between_updates_but_not_inside_one_update():
    from sglang.srt.models.qwen2_vl import Qwen2VisionRotaryEmbedding

    model = _model()
    model.rotary = Qwen2VisionRotaryEmbedding(4)
    state = Qwen35ReloadAdapter(type(model)).bind(model)
    model.rotary.update_freqs_cache(4)
    epoch = state.capture_epoch()
    state.refresh({"norm.gemma_weight"}, epoch=epoch)
    model.rotary.update_freqs_cache(16)
    with pytest.raises(RuntimeError, match="after update prepare"):
        state.refresh({"norm.gemma_weight"}, epoch=epoch)


def test_gemma_graph_storage_cannot_change_between_updates():
    model = _model()
    state = Qwen35ReloadAdapter(type(model)).bind(model)
    model.norm.gemma_weight = model.norm.gemma_weight.clone()
    with pytest.raises(RuntimeError, match="allocation epoch changed"):
        state.capture_epoch()


@pytest.mark.parametrize(
    "enabled,paths,supported",
    [(None, None, True), (False, None, True), (True, None, False), (None, {"adapter": "/model/lora"}, False)],
)
def test_receiver_capability_accepts_disabled_lora_default_but_rejects_adapters(
    monkeypatch, enabled, paths, supported
):
    from types import SimpleNamespace

    from sglang.srt.configs.model_config import ModelConfig

    from relax.backends.sglang import shard_p2p_reload

    args = SimpleNamespace(
        pp_size=1,
        dp_size=1,
        ep_size=1,
        quantization=None,
        enable_lora=enabled,
        lora_paths=paths,
        speculative_algorithm=None,
        enable_dp_attention=False,
        enable_memory_saver=False,
    )
    # Only isolate the GPU constructor/layout proof: this test exercises the
    # actual mode gate that previously rejected SGLang's disabled None default.
    monkeypatch.setattr(
        ModelConfig, "from_server_args", lambda args: SimpleNamespace(hf_config=SimpleNamespace(model_type="test"))
    )

    def unsupported_factory(config, model=None):
        raise NotImplementedError("GPU layout proof belongs to the real receiver")

    monkeypatch.setattr(shard_p2p_reload, "reload_adapter", unsupported_factory)
    result = shard_p2p_reload.receiver_capability(torch.nn.Linear(2, 2), args, 0)
    assert result["checks"]["receiver_mode"] is supported
    assert result["observed"]["enable_lora"] is enabled
    assert result["observed"]["lora_paths"] is paths
    assert (result["rejection"] is not None) is supported


def test_persistent_registration_refuses_memory_saver_and_unknown_allocation_policy():
    from types import SimpleNamespace

    from relax.backends.sglang.shard_p2p_reload import receiver_capability

    args = SimpleNamespace(
        pp_size=1,
        dp_size=1,
        ep_size=1,
        quantization=None,
        enable_lora=None,
        lora_paths=None,
        speculative_algorithm=None,
        enable_dp_attention=False,
        enable_memory_saver=True,
    )
    # A negative mode response must precede native factory probing or binding.
    model = torch.nn.Linear(2, 2)
    enabled = receiver_capability(model, args, 0)
    assert enabled["checks"]["receiver_mode"] is False and enabled["adapter"] is None
    assert enabled["observed"]["enable_memory_saver"] is True
    del args.enable_memory_saver
    unknown = receiver_capability(model, args, 0)
    assert unknown["checks"]["receiver_mode"] is False and unknown["adapter"] is None
    assert unknown["observed"]["enable_memory_saver"] is None


def test_same_parameter_alias_refreshes_once_and_hidden_rebinding_fails():
    from relax.backends.sglang.shard_p2p_reload import parameter_bindings, validate_parameter_storage

    model = _model()
    model.alias = model.norm.weight
    model.other = torch.nn.Parameter(torch.zeros_like(model.norm.weight))
    validate_parameter_storage(model)
    bindings = dict(parameter_bindings(model))
    assert bindings["alias"] is bindings["norm.weight"]
    state = Qwen35ReloadAdapter(type(model)).bind(model)
    state.refresh({"norm.gemma_weight"})
    # Canonical iteration still contains the same two physical parameters.
    # Rebind an alias that is hidden by duplicate removal to the other owner.
    model.norm.weight = model.other
    with pytest.raises(RuntimeError, match="alias binding changed"):
        state.capture_epoch()


def test_distinct_parameter_storage_views_are_rejected():
    from relax.backends.sglang.shard_p2p_reload import validate_parameter_storage

    model = torch.nn.Module()
    backing = torch.zeros(8)
    model.left = torch.nn.Parameter(backing[:6])
    model.right = torch.nn.Parameter(backing[4:])
    with pytest.raises(NotImplementedError, match="overlapping storage"):
        validate_parameter_storage(model)


def test_replica_postload_suppression_does_not_relax_receiver_hook_validation():
    model = _model()
    adapter = Qwen35ReloadAdapter(type(model))
    model.post_load_weights = lambda: None
    adapter.validate_written_buffers(model, {"norm.gemma_weight"})
    with pytest.raises(NotImplementedError, match="post_load_weights hook"):
        adapter.bind(model)


def test_replica_detects_rebinding_of_an_alias_to_an_existing_parameter():
    from relax.backends.sglang.shard_p2p_reload import parameter_bindings
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.model_replica import ModelReplica

    model = torch.nn.Module()
    model.left = torch.nn.Parameter(torch.zeros(4))
    model.right = torch.nn.Parameter(torch.ones(4))
    model.alias = model.left
    replica = ModelReplica.__new__(ModelReplica)
    replica._model = model
    replica._parameter_bindings = parameter_bindings(model)
    model.alias = model.right
    with pytest.raises(RuntimeError, match="alias binding changed"):
        replica._assert_params_changed_only_bytes({}, {})


@pytest.mark.parametrize("rebind", [False, True])
def test_probe_and_buffer_loading_preserve_aliases_and_restore_failed_probe(monkeypatch, rebind):
    from types import SimpleNamespace

    from relax.backends.sglang.shard_p2p_reload import parameter_bindings, validate_parameter_bindings
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote import model_replica
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.compat import RankConfig
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.loader_probe import HfTensorSpec

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4))
            self.other = torch.nn.Parameter(torch.zeros(4))
            self.alias = self.weight

        def load_weights(self, values):
            for name, value in values:
                target = self.alias if name == "hf_weight" else self.other
                target.data.copy_(value)
            if rebind:
                self.alias = self.other

    model = Model()
    config = RankConfig((0, 1), SimpleNamespace())
    # The toy loader has no server settings; isolate only global config lookup.
    monkeypatch.setattr(model_replica, "get_server_args", lambda: config.server_args)
    replica = model_replica.ModelReplica(
        model,
        {name: value.shape for name, value in model.named_parameters()},
        config,
        postprocess_device=torch.device("cpu"),
        transfer_buffer_device=torch.device("cpu"),
    )
    original = parameter_bindings(model)
    specs = {name: HfTensorSpec((4,), torch.float32) for name in ("hf_weight", "hf_other")}
    if rebind:
        with pytest.raises(RuntimeError, match="alias binding changed"):
            replica.map_hf_names(specs)
        validate_parameter_bindings(model, original)
        assert model.alias is model.weight
        return
    mapping = replica.map_hf_names(specs)
    assert mapping.hf_names_by_param_name == {"weight": frozenset({"hf_weight"}), "other": frozenset({"hf_other"})}
    buffer = torch.empty(16, dtype=torch.uint8)
    values = torch.arange(4, dtype=torch.float32)
    packed = replica.load_into(buffer, ["weight"], [("hf_weight", values)])
    assert set(packed) == {"weight"} and packed["weight"].numel() == 16
    torch.testing.assert_close(packed["weight"].view(torch.float32), values, rtol=0, atol=0)
    assert model.alias is model.weight and model.weight.numel() == 0
    assert replica._fully_written_param_names == {"weight"}
