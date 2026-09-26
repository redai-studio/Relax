#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for DAPO math reward utility functions.

The target module is loaded directly from its file path so these pure string
utility tests do not import the reward package entrypoint and its heavy
dependencies.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_math_dapo_utils():
    module_path = Path(__file__).resolve().parents[3] / "relax" / "engine" / "rewards" / "math_dapo_utils.py"
    spec = importlib.util.spec_from_file_location("math_dapo_utils_under_test", module_path)
    assert spec is not None, f"failed to create module spec for {module_path}"
    assert spec.loader is not None, f"failed to create module loader for {module_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


math_dapo_utils = _load_math_dapo_utils()


class TestLastBoxedOnlyString:
    def test_returns_last_complete_boxed_expression(self):
        response = "first answer is \\boxed{1}, final answer is \\boxed{\\frac{3}{4}}"

        result = math_dapo_utils.last_boxed_only_string(response)

        assert result == "\\boxed{\\frac{3}{4}}"

    def test_returns_none_without_boxed_expression(self):
        result = math_dapo_utils.last_boxed_only_string("final answer is 9")

        assert result is None

    def test_returns_none_for_unclosed_boxed_expression(self):
        result = math_dapo_utils.last_boxed_only_string("final answer is \\boxed{9")

        assert result is None


class TestRemoveBoxed:
    def test_returns_inner_content(self):
        result = math_dapo_utils.remove_boxed("\\boxed{9}")

        assert result == "9"

    def test_preserves_inner_latex_content(self):
        result = math_dapo_utils.remove_boxed("\\boxed{\\frac{3}{4}}")

        assert result == "\\frac{3}{4}"

    def test_rejects_invalid_prefix(self):
        with pytest.raises(AssertionError, match="box error: 9"):
            math_dapo_utils.remove_boxed("9")

    def test_rejects_missing_closing_brace(self):
        with pytest.raises(AssertionError, match=r"box error: \\boxed\{9"):
            math_dapo_utils.remove_boxed("\\boxed{9")


class TestNormalizeFinalAnswer:
    def test_removes_units_spaces_math_delimiters_and_commas(self):
        result = math_dapo_utils.normalize_final_answer("an $1,234$ dollars")

        assert result == "1234"

    def test_uses_answer_after_equals_sign(self):
        result = math_dapo_utils.normalize_final_answer("x = 42")

        assert result == "42"

    def test_unwraps_text_and_boxed(self):
        result = math_dapo_utils.normalize_final_answer("\\boxed{\\text{9}}")

        assert result == "9"

    def test_normalizes_sqrt_shorthand(self):
        result = math_dapo_utils.normalize_final_answer("sqrt9")

        assert result == "sqrt{9}"


class TestIsCorrectStrictBox:
    def test_scores_correct_final_boxed_answer(self):
        response = "reasoning steps... final answer is \\boxed{9}"

        result = math_dapo_utils.is_correct_strict_box(response, "9")

        assert result == (1, "9")

    def test_scores_wrong_final_boxed_answer(self):
        response = "reasoning steps... final answer is \\boxed{8}"

        result = math_dapo_utils.is_correct_strict_box(response, "9")

        assert result == (-1, "8")

    def test_scores_missing_boxed_answer_as_incorrect(self):
        result = math_dapo_utils.is_correct_strict_box("final answer is 9", "9")

        assert result == (-1, None)

    def test_uses_last_boxed_expression(self):
        response = "intermediate answer is \\boxed{8}; final answer is \\boxed{9}"

        result = math_dapo_utils.is_correct_strict_box(response, "9")

        assert result == (1, "9")

    def test_does_not_normalize_strict_boxed_content(self):
        result = math_dapo_utils.is_correct_strict_box("final answer is \\boxed{09}", "9")

        assert result == (-1, "09")

    def test_rejects_invalid_pause_tokens_index_length(self):
        with pytest.raises(AssertionError):
            math_dapo_utils.is_correct_strict_box(
                "final answer is \\boxed{9}",
                "9",
                pause_tokens_index=[1, 2, 3],
            )


class TestComputeScore:
    def test_returns_positive_dict_for_strict_boxed_answer(self):
        response = "reasoning steps... final answer is \\boxed{9}"

        result = math_dapo_utils.compute_score(response, "9", strict_box_verify=True)

        assert result == {"score": 1.0, "acc": True, "pred": "9"}

    def test_returns_negative_dict_for_strict_boxed_wrong_answer(self):
        response = "reasoning steps... final answer is \\boxed{8}"

        result = math_dapo_utils.compute_score(response, "9", strict_box_verify=True)

        assert result == {"score": -1.0, "acc": False, "pred": "8"}

    def test_default_path_uses_answer_prefix(self):
        response = "Reasoning steps... Answer: \\boxed{9}"

        result = math_dapo_utils.compute_score(response, "9")

        assert result == {"score": 1.0, "acc": True, "pred": "9"}

    def test_default_path_rejects_non_answer_prefixed_boxed_text(self):
        response = "reasoning steps... final answer is \\boxed{9}"

        result = math_dapo_utils.compute_score(response, "9")

        assert result == {"score": -1.0, "acc": False, "pred": "[INVALID]"}

    def test_only_scans_last_300_characters(self):
        response = ("x" * 350) + " Answer: 9"

        result = math_dapo_utils.compute_score(response, "9")

        assert result == {"score": 1.0, "acc": True, "pred": "9"}
