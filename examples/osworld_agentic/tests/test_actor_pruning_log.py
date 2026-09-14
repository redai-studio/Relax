# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Execute the real actor's pruning branch without importing GPU
dependencies."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class _ReachedDataIterator(Exception):
    """Stop at the first model-data boundary, before any forward or update."""


class TestActorPruningLog(unittest.TestCase):
    def setUp(self) -> None:
        configured = os.environ.get("OSWORLD_TEST_ACTOR_SOURCE")
        source = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[3] / "relax/backends/megatron/actor.py"
        )
        if not source.is_file():
            if configured:
                self.fail(f"OSWORLD_TEST_ACTOR_SOURCE does not exist: {source}")
            self.skipTest("Real actor source is absent; set OSWORLD_TEST_ACTOR_SOURCE to run this regression")

        tree = ast.parse(source.read_text(), filename=str(source))
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_actor"]
        self.assertEqual(len(methods), 1, "Expected exactly one real train_actor method")
        self.assertFalse(methods[0].decorator_list, "Review new decorators before bypassing actor initialization")
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), methods[0]],
            type_ignores=[],
        )

        self.boundaries = Mock()
        self.filter_rows = Mock()
        self.logger = Mock()
        self.data_iterator = Mock(side_effect=_ReachedDataIterator)
        self.boundaries.attach_mock(self.filter_rows, "filter_rows")
        self.boundaries.attach_mock(self.logger.info, "log_info")
        self.boundaries.attach_mock(self.data_iterator, "data_iterator")
        namespace = {
            "filter_zero_advantage_rows": self.filter_rows,
            "logger": self.logger,
            "get_data_iterator": self.data_iterator,
        }
        # Compile the complete production method unchanged. Only external
        # boundaries are replaced; the iterator sentinel prevents GPU work.
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        self.train_actor = namespace["train_actor"]
        self.actor = SimpleNamespace(
            args=SimpleNamespace(advantage_estimator="grpo"),
            model=object(),
            _rank=7,
            _weight_snapshot_versions=SimpleNamespace(same_snapshot=Mock(return_value=True)),
        )
        self.assertFalse(hasattr(self.actor, "rank"), "Fixture must preserve the real actor's private rank contract")

    def _execute_to_iterator(self, original, filtered, dropped) -> None:
        self.filter_rows.return_value = (filtered, dropped)
        with self.assertRaises(_ReachedDataIterator):
            self.train_actor(self.actor, 1, original)
        self.data_iterator.assert_called_once_with(self.actor.args, self.actor.model, filtered)

    def test_pruned_rows_log_private_rank_before_model_data_iteration(self) -> None:
        original = {"total_lengths": [10, 20, 30]}
        filtered = {"total_lengths": [20, 30]}

        self._execute_to_iterator(original, filtered, 1)

        self.filter_rows.assert_called_once_with(self.actor.args, original, reference_matches_actor=True)
        self.actor._weight_snapshot_versions.same_snapshot.assert_called_once_with("actor", "ref")
        self.logger.info.assert_called_once_with(
            "Pruned %s/%s zero-advantage rows before model forwards (rank=%s)",
            1,
            3,
            7,
        )
        self.assertEqual(
            [entry[0] for entry in self.boundaries.mock_calls], ["filter_rows", "log_info", "data_iterator"]
        )

    def test_no_pruned_rows_reaches_iterator_without_pruning_log(self) -> None:
        data = {"total_lengths": [10, 20]}

        self._execute_to_iterator(data, data, 0)

        self.filter_rows.assert_called_once_with(self.actor.args, data, reference_matches_actor=True)
        self.logger.info.assert_not_called()
        self.assertEqual([entry[0] for entry in self.boundaries.mock_calls], ["filter_rows", "data_iterator"])

    def test_distinct_reference_snapshot_is_passed_to_filter(self) -> None:
        data = {"total_lengths": [10, 20]}
        self.actor._weight_snapshot_versions.same_snapshot.return_value = False

        self._execute_to_iterator(data, data, 0)

        self.filter_rows.assert_called_once_with(self.actor.args, data, reference_matches_actor=False)
        self.logger.info.assert_not_called()

    def test_missing_snapshot_ledger_is_passed_to_filter_as_unmatched(self) -> None:
        data = {"total_lengths": [10, 20]}
        del self.actor._weight_snapshot_versions

        self._execute_to_iterator(data, data, 0)

        self.filter_rows.assert_called_once_with(self.actor.args, data, reference_matches_actor=False)
        self.logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
