# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils.training import eval_config
from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


@pytest.fixture
def eval_parser(arguments_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ArgumentParser:
    router = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    monkeypatch.setattr(arguments_module, "RouterArgs", router)
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)
    monkeypatch.setattr(arguments_module, "build_eval_dataset_configs", eval_config.build_eval_dataset_configs)
    monkeypatch.setattr(arguments_module, "ensure_dataset_list", eval_config.ensure_dataset_list)
    return arguments_module.get_slime_extra_args_provider()(ArgumentParser())


@pytest.mark.parametrize("parsed_eval_args", [False, True], ids=["public-compat", "parsed-relax"])
def test_eval_prompt_data_inherits_declared_options(
    arguments_module: ModuleType, eval_parser: ArgumentParser, parsed_eval_args: bool
) -> None:
    args = eval_parser.parse_args(
        ["--eval-prompt-data", "validation.jsonl", "--eval-temperature", "0", "--n-samples-per-prompt", "4"]
    )

    del args.rollout_temperature
    (dataset,) = arguments_module._resolve_eval_datasets(args, _parsed_eval_args=parsed_eval_args)

    assert (dataset.name, dataset.path) == ("aime", "validation.jsonl")
    assert args.eval_prompt_data == ["aime", "validation.jsonl"]
    assert dataset.temperature == 0.0
    assert dataset.top_p == args.rollout_top_p
    assert dataset.top_k == args.rollout_top_k
    assert dataset.max_response_len == args.rollout_max_response_len
    assert dataset.n_samples_per_eval_prompt == args.n_samples_per_eval_prompt == 1
    assert (dataset.input_key, dataset.label_key, dataset.tool_key, dataset.metadata_key) == (
        args.input_key,
        args.label_key,
        args.tool_key,
        args.metadata_key,
    )


@pytest.mark.parametrize("parsed_eval_args", [False, True], ids=["public-compat", "parsed-relax"])
def test_eval_yaml_preserves_dataset_and_default_precedence(
    arguments_module: ModuleType, eval_parser: ArgumentParser, tmp_path: Path, parsed_eval_args: bool
) -> None:
    config = tmp_path / "eval.yaml"
    config.write_text(
        """\
eval:
  defaults:
    temperature: 0.25
    top_p: 0
    top_k: false
    label_key: null
    stop: [ignored]
  datasets:
    first:
      path: first.jsonl
      temperature: null
      top_p: 0.5
      stop: []
    second:
      path: second.jsonl
""",
        encoding="utf-8",
    )
    args = eval_parser.parse_args(["--eval-config", str(config), "--eval-temperature", "0.75", "--eval-top-k", "10"])

    del args.eval_temperature, args.rollout_temperature
    first, second = arguments_module._resolve_eval_datasets(args, _parsed_eval_args=parsed_eval_args)

    assert args.eval_prompt_data == ["first", "first.jsonl", "second", "second.jsonl"]
    assert first.temperature is None
    assert first.top_p == 0.5
    assert first.stop == []
    assert second.temperature == 0.25
    assert second.top_p == 0
    assert second.top_k is False
    assert second.label_key is None
    assert second.stop is None


class _SparseArgs(Namespace):
    @property
    def eval_temperature(self) -> float:
        raise AssertionError("dataset defaults supplied temperature")

    @property
    def rollout_top_p(self) -> float:
        raise AssertionError("the explicit eval option supplied top_p")


def test_public_eval_builder_accepts_sparse_args_and_reads_lazily() -> None:
    args = _SparseArgs(eval_top_p=0.0, input_key="prompt")

    (dataset,) = eval_config.build_eval_dataset_configs(
        args, [{"name": "validation", "path": "data.jsonl", "top_k": False}], {"temperature": None}
    )

    assert dataset.temperature is None
    assert dataset.top_p == 0.0
    assert dataset.top_k is False
    assert dataset.input_key == "prompt"
    assert dataset.max_response_len is None
    assert eval_config.pick_from_args(args, ("missing", "eval_top_p", "rollout_top_p")) == 0.0


def test_empty_eval_builder_does_not_read_args() -> None:
    class UnreadableArgs:
        def __getattr__(self, name: str) -> None:
            raise AssertionError(f"unexpected argument access: {name}")

    assert eval_config.build_eval_dataset_configs(UnreadableArgs(), iter(()), {}) == []


def test_parsed_eval_requires_initialized_unshadowed_field(
    arguments_module: ModuleType, eval_parser: ArgumentParser
) -> None:
    args = eval_parser.parse_args(["--eval-prompt-data", "validation.jsonl"])
    del args.rollout_top_p

    with pytest.raises(AttributeError, match="rollout_top_p"):
        arguments_module._resolve_eval_datasets(args, _parsed_eval_args=True)

    (legacy_dataset,) = arguments_module._resolve_eval_datasets(args)
    assert legacy_dataset.top_p is None


@pytest.mark.usefixtures("eval_parser")
def test_parsed_eval_without_datasets_never_reads_inherited_fields(arguments_module: ModuleType) -> None:
    args = Namespace(eval_config=None, eval_prompt_data=None)

    assert arguments_module._resolve_eval_datasets(args, _parsed_eval_args=True) == []
    assert args.eval_prompt_data is None
