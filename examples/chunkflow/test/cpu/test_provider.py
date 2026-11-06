# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Retain provider integration coverage against the actual example patch."""

import sys
import types

import pytest

from examples.chunkflow.setup import prepare
from examples.chunkflow.test.cpu.test_setup import checkout as checkout


pytestmark = pytest.mark.skipif(sys.version_info < (3, 10), reason="Relax model provider requires Python >= 3.10")


@pytest.mark.parametrize("provider_kind", ["bridge", "custom", "raw"])
@pytest.mark.parametrize("enabled", [False, True])
def test_adapter_installs_only_when_enabled_across_providers(monkeypatch, checkout, provider_kind, enabled):
    from tests.cpu.backends.megatron.test_model_provider_vpp import _bridge_args, _load_model_provider

    prepare(checkout)
    module, provider = _load_model_provider(monkeypatch)
    source = checkout / "relax/backends/megatron/model_provider.py"
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    calls = []
    integration = types.ModuleType("examples.chunkflow.integration")
    integration.install = lambda model, size: calls.append((model, size))
    monkeypatch.setitem(sys.modules, integration.__name__, integration if enabled else None)
    # A stale environment variable must not activate a disabled CLI configuration.
    monkeypatch.setenv("RELAX_CHUNKFLOW_CHUNK_SIZE", "4096")
    args = _bridge_args(use_chunkflow=enabled, chunkflow_chunk_size=8192)
    if provider_kind == "custom":
        args.custom_model_provider_path = "test.provider"
        monkeypatch.setattr(module, "load_function", lambda _: provider.provide)
    elif provider_kind == "raw":
        vars(args).update(
            megatron_to_hf_mode="raw",
            transformer_impl="transformer_engine",
            spec="test.spec",
            fp8_param_gather=False,
            padded_vocab_size=32,
            max_position_embeddings=32,
            fp16_lm_cross_entropy=False,
            untie_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            use_rope_scaling=False,
            mtp_num_layers=None,
        )
        monkeypatch.setattr(module, "GPTModel", lambda **kwargs: provider.model)

    built = module.get_model_provider_func(args)(pre_process=True, post_process=False)

    assert built is provider.model
    assert calls == ([(built, 8192)] if enabled else [])
