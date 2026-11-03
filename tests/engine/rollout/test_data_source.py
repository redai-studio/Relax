# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Integration tests for data_source.py and eager Dataset global-slice
semantics.

Extracted from test_streaming_dataset.py during tests/ directory
restructuring.

Run with: pytest tests/engine/rollout/test_data_source.py -v
"""

import json
import os
import sys
import tempfile
from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def data_source_module(monkeypatch):
    processing_utils = ModuleType("relax.utils.data.processing_utils")
    processing_utils.load_processor = MagicMock(return_value=None)
    processing_utils.load_tokenizer = MagicMock()
    monkeypatch.setitem(sys.modules, "relax.utils.data.processing_utils", processing_utils)
    monkeypatch.delitem(sys.modules, "relax.engine.rollout.data_source", raising=False)

    from relax.engine.rollout import data_source

    yield data_source
    monkeypatch.delitem(sys.modules, "relax.engine.rollout.data_source", raising=False)


class TestEagerDataset:
    """Tests for eager Dataset global-slice semantics."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        return tokenizer

    def test_dataset_multi_file_global_slice(self, mock_tokenizer):
        from relax.utils.data.data import Dataset

        data1 = [{"text": f"A{i}", "label": f"a{i}"} for i in range(3)]
        data2 = [{"text": f"B{i}", "label": f"b{i}"} for i in range(3)]
        files = []
        try:
            for data in (data1, data2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                    files.append(f.name)

            path = f"[{files[0]},{files[1]}]@[1:5]"
            dataset = Dataset(
                path=path,
                tokenizer=mock_tokenizer,
                processor=None,
                max_length=None,
                prompt_key="text",
                label_key="label",
            )

            assert len(dataset) == 4
            prompts = [dataset[i].prompt for i in range(len(dataset))]
            assert prompts == ["A1", "A2", "B0", "B1"]
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)


class TestDataSourceIntegration:
    """Integration tests for data_source.py with StreamingDataset."""

    @pytest.fixture
    def jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        data = [{"text": f"Sample {i}", "label": f"label_{i}"} for i in range(10)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            filepath = f.name

        yield filepath, data
        os.unlink(filepath)

    def test_factory_function_streaming(self, jsonl_file, data_source_module):
        """Test _create_dataset factory with streaming enabled."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file

        args = MagicMock()
        args.use_streaming_dataset = True
        args.streaming_buffer_size = 100
        args.prompt_data = filepath
        args.rollout_max_prompt_len = None
        args.input_key = "text"
        args.multimodal_keys = None
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.system_prompt = None
        args.tool_key = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = None
        args.rollout_seed = 42
        args.custom_prompt_path = None

        tokenizer = MagicMock()

        dataset = data_source_module._create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, StreamingDataset)
        assert len(dataset) == len(data)

    def test_factory_function_traditional(self, jsonl_file, data_source_module):
        """Test _create_dataset factory with streaming disabled."""
        from relax.utils.data.data import Dataset

        filepath, data = jsonl_file

        args = MagicMock()
        args.use_streaming_dataset = False
        args.prompt_data = filepath
        args.rollout_max_prompt_len = None
        args.input_key = "text"
        args.multimodal_keys = None
        args.label_key = "label"
        args.metadata_key = "metadata"
        args.system_prompt = None
        args.tool_key = None
        args.apply_chat_template = False
        args.apply_chat_template_kwargs = None
        args.rollout_seed = 42
        args.custom_prompt_path = None

        tokenizer = MagicMock()

        dataset = data_source_module._create_dataset(args, tokenizer, processor=None)

        assert isinstance(dataset, Dataset)

    def test_factory_function_streaming_multi_file_slice(self, data_source_module):
        """Test _create_dataset factory with streaming dataset over multiple
        files and outer slice."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        data1 = [{"text": f"A{i}", "label": f"a{i}"} for i in range(3)]
        data2 = [{"text": f"B{i}", "label": f"b{i}"} for i in range(3)]
        files = []
        try:
            for data in (data1, data2):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                    files.append(f.name)

            args = MagicMock()
            args.use_streaming_dataset = True
            args.streaming_buffer_size = 100
            args.prompt_data = f"[{files[0]},{files[1]}]@[2:6]"
            args.rollout_max_prompt_len = None
            args.input_key = "text"
            args.multimodal_keys = None
            args.label_key = "label"
            args.metadata_key = "metadata"
            args.system_prompt = None
            args.tool_key = None
            args.apply_chat_template = False
            args.apply_chat_template_kwargs = None
            args.rollout_seed = 42
            args.custom_prompt_path = None

            tokenizer = MagicMock()
            dataset = data_source_module._create_dataset(args, tokenizer, processor=None)

            assert isinstance(dataset, StreamingDataset)
            assert len(dataset) == 4
            prompts = [dataset[i].prompt for i in range(len(dataset))]
            assert prompts == ["A2", "B0", "B1", "B2"]
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)

    def test_shallow_copy_drops_generation_export_state(self, data_source_module):
        from relax.utils.types import Sample

        source = Sample(spec_generations=[{"version": 1, "generation_id": "old"}])
        copied = data_source_module._shallow_copy_sample(source)

        assert copied.spec_generations is None

    @pytest.mark.parametrize("samples_per_prompt", [1, 3])
    @pytest.mark.parametrize("max_length", [None, 5])
    def test_streaming_buffer_resume_preserves_groups_and_prompt_order(
        self,
        jsonl_file: tuple[str, list[dict[str, str]]],
        data_source_module,
        monkeypatch: pytest.MonkeyPatch,
        samples_per_prompt: int,
        max_length: int | None,
    ) -> None:
        from relax.utils.data.streaming_dataset import StreamingDataset
        from relax.utils.types import Sample

        data_source = data_source_module
        filepath, rows = jsonl_file
        tokenizer = MagicMock()
        dataset = StreamingDataset(filepath, tokenizer, processor=None, max_length=max_length)
        dataset.shuffle(0)
        order = list(dataset.index_manager.indices)
        filtered = {order[2], order[5]} if max_length is not None else set()
        filtered_prompts = {rows[idx]["text"] for idx in filtered}
        tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [1] * (6 if text in filtered_prompts else 5)}
        valid_indices = [idx for idx in order if idx not in filtered]
        # Exercise the real buffer/data-source path with a local dataset and no model loading.
        monkeypatch.setattr(data_source, "_create_dataset", lambda *args: dataset)
        args = Namespace(
            use_streaming_dataset=True,
            rollout_global_dataset=True,
            hf_checkpoint=None,
            dump_details=None,
            rollout_shuffle=True,
            n_samples_per_prompt=samples_per_prompt,
            buffer_filter_path=None,
        )
        source = data_source.RolloutDataSourceWithBuffer(args)
        initial = source.get_samples(4)
        partial = initial[0]
        partial[0].status = Sample.Status.ABORTED
        partial[0].tokens = [10, 11]
        partial[0].response = "unfinished"
        partial[0].response_length = 1
        partial[0].abort_count = 1
        partial[0].metadata["start_rollout_id"] = 0
        source.add_samples([partial, initial[1]])

        mixed = source.get_samples(len(valid_indices) - len(initial) + 2)
        assert mixed[0] is partial
        assert mixed[1] is initial[1]
        assert partial[0].status == Sample.Status.ABORTED
        assert partial[0].tokens == [10, 11]
        assert partial[0].response == "unfinished"
        assert partial[0].response_length == 1
        assert partial[0].abort_count == 1
        assert partial[0].metadata["start_rollout_id"] == 0

        fresh_groups = initial + mixed[2:]
        assert [group[0].prompt for group in fresh_groups] == [rows[idx]["text"] for idx in valid_indices]
        assert all(len(group) == samples_per_prompt for group in fresh_groups)
        assert [group[0].group_index for group in fresh_groups] == list(range(len(valid_indices)))
        assert [sample.index for group in fresh_groups for sample in group] == list(
            range(len(valid_indices) * samples_per_prompt)
        )
        assert dataset.index_manager.position == len(rows)
        assert dataset.index_manager.current_epoch == source.epoch_id == 0
        assert dataset._filter_count == len(filtered)
        assert source.get_buffer_length() == 0

        # A request entirely satisfied by resumed groups must not read new prompts.
        source.add_samples(mixed[:2])
        state = dataset.get_state()
        group_index, sample_index = source.sample_group_index, source.sample_index
        resumed = source.get_samples(2)
        assert all(actual is expected for actual, expected in zip(resumed, mixed[:2], strict=True))
        assert dataset.get_state() == state
        assert (source.sample_group_index, source.sample_index) == (group_index, sample_index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
