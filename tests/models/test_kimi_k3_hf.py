# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in, offline CPU checks against KIMI_K3_SOURCE_DIR's official HF code.

The source directory must contain K3's Python/config files and tiktoken.model.
No model weight shards are loaded, and the vision test reduces the architecture
before constructing MoonViT or the projector.
"""

import copy
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def official_k3(tmp_path_factory):
    source_value = os.environ.get("KIMI_K3_SOURCE_DIR")
    if not source_value:
        pytest.skip("Set KIMI_K3_SOURCE_DIR to a local K3 HF source/config/tokenizer directory.")
    source = Path(source_value)
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tiktoken.model")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        pytest.skip(f"KIMI_K3_SOURCE_DIR is missing: {', '.join(missing)}")

    workspace = tmp_path_factory.mktemp("kimi_k3_hf")
    checkpoint = workspace / f"snapshot_{workspace.name}"
    checkpoint.mkdir()
    # Copy only small source/config files and the tokenizer vocabulary. Keeping
    # the snapshot and dynamic-module cache here leaves the supplied repo intact.
    for path in source.glob("*.py"):
        shutil.copy2(path, checkpoint / path.name)
    for name in (*required, "generation_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if (source / name).is_file():
            shutil.copy2(source / name, checkpoint / name)

    with pytest.MonkeyPatch.context() as patch:
        cache = workspace / "hf_cache"
        modules_cache = cache / "modules"
        patch.setenv("HF_HOME", str(cache))
        patch.setenv("HF_HUB_CACHE", str(cache / "hub"))
        patch.setenv("HF_MODULES_CACHE", str(modules_cache))
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        patch.setenv("TIKTOKEN_CACHE_DIR", str(workspace / "tiktoken_cache"))
        patch.syspath_prepend(str(modules_cache))

        torch = pytest.importorskip("torch")
        numpy = pytest.importorskip("numpy")
        image_module = pytest.importorskip("PIL.Image")
        pytest.importorskip("tiktoken")
        transformers = pytest.importorskip("transformers")
        dynamic_modules = pytest.importorskip("transformers.dynamic_module_utils")
        hub_constants = pytest.importorskip("huggingface_hub.constants")
        # Other tests may already have imported HF before these env variables.
        patch.setattr(dynamic_modules, "HF_MODULES_CACHE", str(modules_cache))
        patch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
        dynamic_before = {name for name in sys.modules if name.startswith("transformers_modules")}
        original_threads = torch.get_num_threads()
        torch.set_num_threads(min(original_threads, 2))
        try:
            from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
            from relax.utils.data.kimi_k3 import make_kimi_k3_sft_request, process_kimi_k3_sft_images
            from relax.utils.data.processing_utils import expand_kimi_k25_placeholders, remap_mm_train_inputs

            load_kwargs = dict(trust_remote_code=True, local_files_only=True, cache_dir=str(cache / "hub"))
            tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, use_fast=False, **load_kwargs)
            processor = transformers.AutoProcessor.from_pretrained(checkpoint, use_fast=False, **load_kwargs)
            config = transformers.AutoConfig.from_pretrained(checkpoint, **load_kwargs)
            image = image_module.new("RGB", (56, 28), (25, 125, 210))
            sample = CanonicalSample(
                [
                    CanonicalMessage(
                        "user", [{"type": "text", "text": "Describe this <|open|>literal."}, {"type": "image"}], False
                    ),
                    CanonicalMessage("assistant", "A blue rectangle.", True, reasoning_content="Look at its color."),
                ],
                {"source_dataset": "k3-hf-cpu-test", "row_index": 0},
                images=[image],
            )
            request = make_kimi_k3_sft_request(sample)
            output = process_kimi_k3_sft_images(processor, {"images": [image]}, request)
            official = processor(
                messages=request["messages"],
                tools=request["tools"],
                medias=[{"type": "image", "image": image}],
                **request["kwargs"],
            )
            expected = expand_kimi_k25_placeholders(
                processor,
                official["input_ids"][0].tolist(),
                remap_mm_train_inputs(processor, {"grid_thws": official["grid_thws"]}),
            )
            yield SimpleNamespace(
                torch=torch,
                numpy=numpy,
                tokenizer=tokenizer,
                processor=processor,
                config=config,
                checkpoint=checkpoint,
                image=image,
                request=request,
                output=output,
                expected=expected,
            )
        finally:
            torch.set_num_threads(original_threads)
            for name in set(sys.modules) - dynamic_before:
                if name.startswith("transformers_modules"):
                    sys.modules.pop(name, None)


def test_kimi_k3_official_tokenizer_processor_and_worker_match(official_k3, monkeypatch):
    from relax.utils.data import processor_pool
    from relax.utils.data.kimi_k3 import KIMI_K3_SFT_LOSS_MASK, KIMI_K3_SFT_REQUEST

    case = official_k3
    torch = case.torch
    mask = case.output[KIMI_K3_SFT_LOSS_MASK]
    assert case.output["input_ids"][0] == case.expected
    assert mask.numel() == len(case.expected)
    assert mask.sum() > 0
    assert not mask[torch.tensor(case.expected) == case.config.media_placeholder_token_id].any()
    learned = case.tokenizer.decode([token for token, learn in zip(case.expected, mask.tolist()) if learn])
    assert "A blue rectangle." in learned
    assert "literal" not in learned

    monkeypatch.setattr(processor_pool, "_worker_processor", case.processor)
    monkeypatch.setattr(processor_pool, "_worker_multimodal_config", None)
    worker_ids, worker_mm = processor_pool.process_sample_in_worker(
        "", {"images": [case.numpy.asarray(case.image)]}, {KIMI_K3_SFT_REQUEST: case.request}
    )
    assert worker_ids == case.expected
    assert torch.equal(worker_mm[KIMI_K3_SFT_LOSS_MASK], mask)
    assert worker_mm["pixel_values"].dtype == torch.bfloat16
    assert worker_mm["pixel_values"].device.type == "cpu"


def test_kimi_k3_official_tiny_vision_backpropagates_and_text_batch_has_zero_visual_gradients(
    official_k3, monkeypatch
):
    pytest.importorskip("einops")
    pytest.importorskip("fla")
    pytest.importorskip("megatron.core.transformer.module")
    pytest.importorskip("megatron.bridge.utils.common_utils")
    transformers_utils = pytest.importorskip("transformers.utils")
    from relax.utils.data.kimi_k3 import KIMI_K3_SFT_LOSS_MASK

    case = official_k3
    torch = case.torch
    vision_config = copy.deepcopy(case.config.vision_config)
    reduced = dict(
        vt_hidden_size=32,
        vt_intermediate_size=64,
        vt_num_attention_heads=4,
        qkv_hidden_size=32,
        vt_num_hidden_layers=2,
        mm_hidden_size=32,
        text_hidden_size=16,
        init_pos_emb_height=2,
        init_pos_emb_width=4,
        init_pos_emb_time=1,
    )
    for name, value in reduced.items():
        setattr(vision_config, name, value)
    vision_config._attn_implementation = "eager"
    assert vision_config.vt_hidden_size == vision_config.mm_hidden_size == 32
    assert vision_config.vt_num_hidden_layers == 2
    assert vision_config.text_hidden_size == 16
    assert vision_config.mm_projector_type == "patchmergerv2"

    class TinyLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(64, 16)
            self.head = torch.nn.Linear(16, 1)
            self.pre_process = True
            self.post_process = True
            self.pg_collection = SimpleNamespace(tp=None)
            self.share_embeddings_and_output_weights = False

        def embedding(self, input_ids, position_ids=None):
            return self.table(input_ids.remainder(64)).transpose(0, 1)

        def forward(self, decoder_input, **kwargs):
            # Future text depends on preceding image embeddings, so an
            # assistant-only loss still exercises the real visual gradients.
            return self.head(decoder_input.cumsum(0)).squeeze(-1).transpose(0, 1)

        def shared_embedding_or_output_weight(self):
            return None

    # Leaf loading avoids importing the new K3 Bridge into an older test image.
    model_path = Path(__file__).parents[2] / "relax/models/kimi_k3/model.py"
    spec = importlib.util.spec_from_file_location("_relax_k3_hf_cpu_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(
        provide_language_model=lambda **kwargs: TinyLanguageModel(),
        image_token_id=case.config.media_placeholder_token_id,
        params_dtype=torch.float32,
        hf_model_path=str(case.checkpoint),
        trust_remote_code=True,
        vision_config=vision_config,
        sequence_parallel=False,
    )
    monkeypatch.setattr(transformers_utils, "is_flash_attn_2_available", lambda: False)
    model = module.KimiK3VLModel(config)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
    mask = case.output[KIMI_K3_SFT_LOSS_MASK]
    scores = model(
        torch.tensor([case.expected]),
        pixel_values=case.output["pixel_values"],
        image_grid_thw=case.output["image_grid_thw"],
    )
    (scores * mask.unsqueeze(0)).square().sum().backward()
    for name in ("vision_tower", "mm_projector"):
        parameters = list(getattr(model, name).parameters())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
        assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)

    model.zero_grad(set_to_none=True)
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    for name in ("vision_tower", "mm_projector"):
        assert all(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
            for parameter in getattr(model, name).parameters()
        )


def test_kimi_k3_official_config_selects_visual_bridge_without_loading_language_weights(official_k3):
    pytest.importorskip("megatron.bridge.models.kimi.kimi_k3_bridge")
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    from relax.models.kimi_k3 import KimiK3VLBridge, KimiK3VLModelProvider

    with patch.object(PreTrainedCausalLM, "_load_model", side_effect=AssertionError("HF language model was loaded")):
        auto = AutoBridge.from_hf_pretrained(official_k3.checkpoint, trust_remote_code=True, local_files_only=True)
        provider = auto.to_megatron_provider(load_weights=False)
        assert isinstance(auto._model_bridge, KimiK3VLBridge)
        assert isinstance(provider, KimiK3VLModelProvider)
        hf_config = auto.hf_pretrained.config
        text = hf_config.text_config
        assert provider.num_layers == text.num_hidden_layers
        assert provider.hidden_size == text.hidden_size
        assert provider.q_lora_rank == text.q_lora_rank
        assert provider.kv_lora_rank == text.kv_lora_rank
        assert provider.moe_router_topk == text.num_experts_per_token
        assert provider.moe_router_score_function == text.moe_router_activation_func
        assert provider.moe_shared_expert_intermediate_size == text.moe_intermediate_size * text.num_shared_experts
        assert len(provider.moe_layer_freq) == text.num_hidden_layers
        assert provider.moe_layer_freq[: text.first_k_dense_replace] == [0] * text.first_k_dense_replace
        assert provider.kimi_kda_layers == tuple(text.linear_attn_config["kda_layers"])
        assert provider.vision_config is hf_config.vision_config
        assert provider.image_token_id == hf_config.media_placeholder_token_id
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.variable_seq_lengths is True
        assert auto._model_bridge._HF_PASSTHROUGH_PREFIXES == ()
        registry = auto._model_bridge.mapping_registry()
        expected = {
            "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "language_model.decoder.layers.0.self_attention.q_a_proj.weight": (
                "language_model.model.layers.0.self_attn.q_a_proj.weight"
            ),
            f"language_model.decoder.layers.{text.num_hidden_layers - 1}.output_attn_res_norm.weight": (
                "language_model.model.output_attn_res_norm.weight"
            ),
            "vision_tower.patch_embed.proj.weight": "vision_tower.patch_embed.proj.weight",
            "mm_projector.proj.0.weight": "mm_projector.proj.0.weight",
            "mm_projector.post_norm.weight": "mm_projector.post_norm.weight",
        }
        for megatron_name, hf_name in expected.items():
            mapping = registry.megatron_to_hf_lookup(megatron_name)
            assert mapping is not None
            assert mapping.megatron_param == megatron_name
            assert mapping.hf_param == hf_name


def test_kimi_k3_core_mixed_precision_keeps_kda_state_in_fp32(official_k3):
    core_module = pytest.importorskip("megatron.core.transformer.module")
    convert = getattr(core_module, "convert_module_to_dtype_except_fp32_marked", None)
    if convert is None:
        pytest.skip("Installed MCore does not expose the FP32-preserving dtype conversion helper.")
    torch = official_k3.torch
    language_model = torch.nn.Module()
    language_model.pre_process = False
    language_model.post_process = True
    language_model.pg_collection = SimpleNamespace(tp=None)
    language_model.share_embeddings_and_output_weights = False
    language_model.attention = torch.nn.Module()
    attention = language_model.attention
    attention.A_log = torch.nn.Parameter(torch.tensor([1.001], dtype=torch.float32))
    attention.dt_bias = torch.nn.Parameter(torch.tensor([2.003], dtype=torch.float32))
    attention.weight = torch.nn.Parameter(torch.tensor([3.007], dtype=torch.float32))
    attention._keep_in_float32_parameter_names = ("A_log", "dt_bias")
    expected_a = attention.A_log.detach().clone()
    expected_dt = attention.dt_bias.detach().clone()

    model_path = Path(__file__).parents[2] / "relax/models/kimi_k3/model.py"
    spec = importlib.util.spec_from_file_location("_relax_k3_fp32_cpu_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(provide_language_model=lambda **kwargs: language_model, image_token_id=0)
    model = module.KimiK3VLModel(config)

    convert(model, torch.bfloat16)

    assert attention.A_log.dtype is torch.float32
    assert attention.dt_bias.dtype is torch.float32
    assert attention.weight.dtype is torch.bfloat16
    torch.testing.assert_close(attention.A_log, expected_a, rtol=0, atol=0)
    torch.testing.assert_close(attention.dt_bias, expected_dt, rtol=0, atol=0)
