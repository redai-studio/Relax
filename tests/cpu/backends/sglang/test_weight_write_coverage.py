# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
from torch import nn

from relax.backends.sglang.weight_write_coverage import record_weight_write_coverage


def _freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


class _FusedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Parameter(torch.zeros((3, 2), dtype=torch.bfloat16), requires_grad=False)


def test_fused_qkv_partial_copy_and_missing_visual_parameter_fail_coverage() -> None:
    model = _FusedModel()
    model.visual = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    _freeze(model)
    q_source = torch.ones((1, 2), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="uncovered byte range"):
        with record_weight_write_coverage(
            model,
            required_hf_names={"model.layers.0.self_attn.q_proj.weight"},
            excluded_expert_names=set(),
        ) as coverage:
            for name, tensor in coverage.tag_weights([("model.layers.0.self_attn.q_proj.weight", q_source)]):
                assert name.endswith("q_proj.weight")
                model.qkv[:1].copy_(tensor)
            coverage.assert_complete()


def test_required_vision_hf_name_must_be_consumed() -> None:
    class TextAndVision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text = nn.Parameter(torch.zeros((2, 2)), requires_grad=False)
            self.visual = nn.Parameter(torch.zeros((2, 2)), requires_grad=False)

    model = TextAndVision()
    required = {"model.layers.0.weight", "visual.patch_embed.weight"}

    with pytest.raises(ValueError, match="did not consume required HF names"):
        with record_weight_write_coverage(
            model,
            required_hf_names=required,
            excluded_expert_names=set(),
        ) as coverage:
            for _name, tensor in coverage.tag_weights([("model.layers.0.weight", torch.ones_like(model.text))]):
                model.text.copy_(tensor)


def test_live_visual_parameter_is_included_in_coverage_denominator() -> None:
    class TextAndVision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text = nn.Parameter(torch.zeros((2, 2)), requires_grad=False)
            self.visual = nn.Parameter(torch.zeros((2, 2)), requires_grad=False)

    model = TextAndVision()
    with pytest.raises(ValueError, match="uncovered byte range: visual"):
        with record_weight_write_coverage(
            model,
            required_hf_names={"model.layers.0.weight"},
            excluded_expert_names=set(),
        ) as coverage:
            for _name, tensor in coverage.tag_weights([("model.layers.0.weight", torch.ones_like(model.text))]):
                model.text.copy_(tensor)


def test_noncontiguous_source_and_bf16_to_fp32_copy_are_covered() -> None:
    model = nn.Linear(3, 4, bias=False, dtype=torch.float32)
    _freeze(model)
    base = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    source = base.transpose(0, 1)
    assert not source.is_contiguous()

    with record_weight_write_coverage(
        model,
        required_hf_names={"dense.weight"},
        excluded_expert_names=set(),
    ) as coverage:
        for _name, tensor in coverage.tag_weights([("dense.weight", source)]):
            model.weight.copy_(tensor)
        coverage.assert_complete()

    torch.testing.assert_close(model.weight, source.float())


def test_loader_cannot_mutate_a_previously_yielded_hf_source() -> None:
    model = nn.Linear(2, 2, bias=False)
    _freeze(model)
    source = torch.full_like(model.weight, 7)
    before = source.clone()

    with pytest.raises(ValueError, match="read-only HF source storage"):
        with record_weight_write_coverage(
            model,
            required_hf_names={"dense.weight"},
            excluded_expert_names=set(),
        ) as coverage:
            for _name, tensor in coverage.tag_weights([("dense.weight", source)]):
                model.weight.copy_(tensor)
                tensor.zero_()

    torch.testing.assert_close(source, before)


def test_unknown_inplace_write_to_protected_parameter_is_rejected() -> None:
    model = nn.Linear(2, 2, bias=False)
    _freeze(model)

    with pytest.raises(ValueError, match="Unsupported in-place write"):
        with record_weight_write_coverage(
            model,
            required_hf_names={"dense.weight"},
            excluded_expert_names=set(),
        ) as coverage:
            for _name, _tensor in coverage.tag_weights([("dense.weight", torch.ones_like(model.weight))]):
                with torch.no_grad():
                    model.weight.add_(1)


@pytest.mark.parametrize("operation", ["copy", "add", "foreach_add"])
def test_ordinary_loader_cannot_write_excluded_expert_storage(operation: str) -> None:
    class ExpertModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Parameter(torch.zeros((2, 2)), requires_grad=False)
            self.mlp = nn.Module()
            self.mlp.experts = nn.Module()
            self.mlp.experts.w13_weight = nn.Parameter(torch.zeros((2, 2)), requires_grad=False)

    model = ExpertModel()
    expert_name = "mlp.experts.w13_weight"
    source = torch.ones_like(model.dense)

    with pytest.raises(ValueError, match="write excluded expert storage"):
        with record_weight_write_coverage(
            model,
            required_hf_names={"dense.weight"},
            excluded_expert_names={expert_name},
        ) as coverage:
            for _name, tensor in coverage.tag_weights([("dense.weight", source)]):
                if operation == "copy":
                    model.mlp.experts.w13_weight.copy_(tensor)
                elif operation == "foreach_add":
                    torch._foreach_add_([model.mlp.experts.w13_weight], 1)
                else:
                    model.mlp.experts.w13_weight.add_(1)


def test_tied_parameter_storage_is_covered_once() -> None:
    class TiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(2, 2, bias=False)
            self.right = nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight

    model = TiedModel()
    _freeze(model)
    source = torch.full_like(model.left.weight, 3)

    with record_weight_write_coverage(
        model,
        required_hf_names={"shared.weight"},
        excluded_expert_names=set(),
    ) as coverage:
        for _name, tensor in coverage.tag_weights([("shared.weight", source)]):
            model.left.weight.copy_(tensor)
        coverage.assert_complete()


def test_overlapping_copy_requires_an_explicit_hf_alias_contract() -> None:
    model = nn.Linear(2, 2, bias=False)
    _freeze(model)
    source = torch.ones_like(model.weight)
    names = {"first.weight", "second.weight"}

    with pytest.raises(ValueError, match="Overlapping parameter writes"):
        with record_weight_write_coverage(
            model,
            required_hf_names=names,
            excluded_expert_names=set(),
        ) as coverage:
            for _name, tensor in coverage.tag_weights([("first.weight", source), ("second.weight", source)]):
                model.weight.copy_(tensor)


def test_explicit_hf_alias_contract_allows_identical_tied_target_write() -> None:
    class TiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(3, 2)
            self.lm_head = nn.Linear(2, 3, bias=False)
            self.lm_head.weight = self.embedding.weight

    model = TiedModel()
    _freeze(model)
    first = torch.ones_like(model.embedding.weight)
    second = torch.full_like(model.embedding.weight, 2)
    names = {"model.embed_tokens.weight", "lm_head.weight"}

    with record_weight_write_coverage(
        model,
        required_hf_names=names,
        excluded_expert_names=set(),
        allowed_hf_aliases={"embedding.weight": names},
    ) as coverage:
        for name, tensor in coverage.tag_weights([("model.embed_tokens.weight", first), ("lm_head.weight", second)]):
            model.embedding.weight.copy_(tensor)
        coverage.assert_complete()


def test_vocab_padding_fill_requires_declared_parameter_relative_range() -> None:
    model = nn.Embedding(5, 2, dtype=torch.bfloat16)
    _freeze(model)
    source = torch.ones((3, 2), dtype=torch.bfloat16)

    with record_weight_write_coverage(
        model,
        required_hf_names={"model.embed_tokens.weight"},
        excluded_expert_names=set(),
        padding_fill_ranges={"weight": [(12, 20)]},
    ) as coverage:
        for _name, tensor in coverage.tag_weights([("model.embed_tokens.weight", source)]):
            model.weight[:3].copy_(tensor)
            model.weight[3:].fill_(0)
        coverage.assert_complete()


def test_zero_length_copy_and_zero_fill_at_parameter_end_are_ignored() -> None:
    model = nn.Linear(2, 2, bias=False)
    _freeze(model)
    source = torch.ones_like(model.weight)

    with record_weight_write_coverage(
        model,
        required_hf_names={"dense.weight"},
        excluded_expert_names=set(),
    ) as coverage:
        for _name, tensor in coverage.tag_weights([("dense.weight", source)]):
            model.weight.copy_(tensor)
            empty_tail = model.weight.reshape(-1)[model.weight.numel() :]
            empty_tail.copy_(torch.empty_like(empty_tail))
            empty_tail.fill_(0)
        coverage.assert_complete()


def test_undeclared_fill_and_out_of_bounds_padding_range_fail() -> None:
    model = nn.Embedding(5, 2, dtype=torch.bfloat16)
    _freeze(model)
    with pytest.raises(ValueError, match="Invalid parameter-relative padding"):
        with record_weight_write_coverage(
            model,
            required_hf_names={"model.embed_tokens.weight"},
            excluded_expert_names=set(),
            padding_fill_ranges={"weight": [(18, 22)]},
        ):
            pass

    with pytest.raises(ValueError, match="Undeclared padding fill"):
        with record_weight_write_coverage(
            model,
            required_hf_names={"model.embed_tokens.weight"},
            excluded_expert_names=set(),
            padding_fill_ranges={"weight": [(12, 20)]},
        ) as coverage:
            for _name, tensor in coverage.tag_weights(
                [("model.embed_tokens.weight", torch.ones((3, 2), dtype=torch.bfloat16))]
            ):
                model.weight[:3].copy_(tensor)
                model.weight[:1].fill_(0)
