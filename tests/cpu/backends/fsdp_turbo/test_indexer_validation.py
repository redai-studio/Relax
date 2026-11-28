# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import pytest
import torch

from relax.backends.fsdp_turbo.qwen4_indexer import validate_lightning_mask


def test_prefix_validation_rechecks_mutated_mask():
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    validate_lightning_mask(mask, 4, 4)
    validate_lightning_mask(mask, 4, 4)
    mask[0, 0, 0, 3] = True
    with pytest.raises(ValueError, match="causal-prefix"):
        validate_lightning_mask(mask, 4, 4)


def test_prefix_validation_accepts_padded_queries():
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    mask[..., 3, :] = False
    validate_lightning_mask(mask, 4, 4)


@pytest.fixture
def mask_builder_config():
    from transformers import masking_utils

    if not hasattr(masking_utils, "create_recurrent_attention_mask"):
        pytest.skip("Requires the pinned Transformers mask builders")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig()
    config._attn_implementation = "eager"
    config.use_cache = False
    return config


def test_sft_mask_is_certified_at_its_real_construction_boundary(monkeypatch, mask_builder_config):
    from types import SimpleNamespace

    from relax.backends.fsdp_turbo import qwen4_indexer
    from relax.backends.fsdp_turbo.model_utils import _pad_batch

    _, mask = _pad_batch([torch.arange(5), torch.arange(3)], torch.device("cpu"))

    def unexpected_scan(*args):
        raise AssertionError("Trusted SFT input must not scan the device mask")

    monkeypatch.setattr(qwen4_indexer, "validate_lightning_mask", unexpected_scan)
    _, kwargs = qwen4_indexer._prepare_lightning_mask(
        SimpleNamespace(config=mask_builder_config),
        (),
        {"attention_mask": mask, "inputs_embeds": torch.zeros(2, 5, 4)},
    )
    causal = kwargs["attention_mask"]["indexed_attention"]
    allowed = causal if causal.dtype == torch.bool else causal >= 0
    expected = torch.ones(5, 5, dtype=torch.bool).tril()[None, None] & mask[:, None, None].bool()
    torch.testing.assert_close(allowed, expected)
    # The actual validator consumes the certification; mutated masks cannot reuse it.
    validate_lightning_mask(causal, 5, 5)
    causal[0, 0, 0, 4] = True if causal.dtype == torch.bool else 0
    with pytest.raises(ValueError, match="causal-prefix"):
        validate_lightning_mask(causal, 5, 5)


def test_external_mapping_is_not_certified(mask_builder_config):
    from types import SimpleNamespace

    from relax.backends.fsdp_turbo.qwen4_indexer import _prepare_lightning_mask

    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="causal-prefix"):
        _prepare_lightning_mask(
            SimpleNamespace(config=mask_builder_config),
            (),
            {"attention_mask": {"indexed_attention": mask}, "inputs_embeds": torch.zeros(1, 5, 4)},
        )
