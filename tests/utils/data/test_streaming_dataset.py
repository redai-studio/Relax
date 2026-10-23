# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for StreamingDataset.

Run with: pytest tests/utils/data/test_streaming_dataset.py -v
"""

import json
import os
import tempfile
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestStreamingReader:
    """Tests for StreamingReader class."""

    @pytest.fixture
    def jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        data = [
            {"text": "Hello world", "label": "positive"},
            {"text": "This is a test", "label": "neutral"},
            {"text": "Another sample", "label": "negative"},
            {"text": "Fourth line here", "label": "positive"},
            {"text": "Fifth and final", "label": "neutral"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            filepath = f.name

        yield filepath, data
        os.unlink(filepath)

    def test_reader_len(self, jsonl_file):
        """Test that StreamingReader returns correct length."""
        from relax.utils.data.streaming_dataset import StreamingReader

        filepath, data = jsonl_file
        reader = StreamingReader(filepath)

        assert len(reader) == len(data)

    def test_reader_getitem(self, jsonl_file):
        """Test random access to lines."""
        from relax.utils.data.streaming_dataset import StreamingReader

        filepath, data = jsonl_file
        reader = StreamingReader(filepath)

        # Test accessing each item
        for i, expected in enumerate(data):
            result = reader[i]
            assert result == expected

    def test_reader_getitem_random_order(self, jsonl_file):
        """Test random access in non-sequential order."""
        from relax.utils.data.streaming_dataset import StreamingReader

        filepath, data = jsonl_file
        reader = StreamingReader(filepath)

        # Access in random order
        indices = [4, 1, 3, 0, 2]
        for i in indices:
            result = reader[i]
            assert result == data[i]

    def test_reader_index_error(self, jsonl_file):
        """Test that out-of-bounds access raises IndexError."""
        from relax.utils.data.streaming_dataset import StreamingReader

        filepath, data = jsonl_file
        reader = StreamingReader(filepath)

        with pytest.raises(IndexError):
            _ = reader[len(data)]

        with pytest.raises(IndexError):
            _ = reader[-1]

    def test_reader_file_not_found(self):
        """Test that non-existent file raises FileNotFoundError."""
        from relax.utils.data.streaming_dataset import StreamingReader

        with pytest.raises(FileNotFoundError):
            StreamingReader("/nonexistent/path/file.jsonl")

    def test_reader_unsupported_format(self):
        """Test that unsupported file format raises ValueError."""
        from relax.utils.data.streaming_dataset import StreamingReader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test\n")
            filepath = f.name

        try:
            with pytest.raises(ValueError):
                StreamingReader(filepath)
        finally:
            os.unlink(filepath)

    def test_reader_with_slice_notation(self, jsonl_file):
        """Test path with slice notation like 'file.jsonl@[1:3]'."""
        from relax.utils.data.streaming_dataset import StreamingReader

        filepath, data = jsonl_file
        sliced_path = f"{filepath}@[1:3]"
        reader = StreamingReader(sliced_path)

        assert len(reader) == 2
        assert reader[0] == data[1]
        assert reader[1] == data[2]

    @pytest.fixture
    def multi_jsonl_files(self):
        data1 = [
            {"text": "a0", "label": "l0"},
            {"text": "a1", "label": "l1"},
            {"text": "a2", "label": "l2"},
        ]
        data2 = [
            {"text": "b0", "label": "l3"},
            {"text": "b1", "label": "l4"},
            {"text": "b2", "label": "l5"},
        ]

        paths = []
        for data in (data1, data2):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
                paths.append(f.name)

        yield paths, data1 + data2

        for path in paths:
            os.unlink(path)

    @pytest.fixture
    def multi_parquet_files(self):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        data1 = [
            {"text": "pa0", "label": "l0"},
            {"text": "pa1", "label": "l1"},
            {"text": "pa2", "label": "l2"},
        ]
        data2 = [
            {"text": "pb0", "label": "l3"},
            {"text": "pb1", "label": "l4"},
            {"text": "pb2", "label": "l5"},
        ]

        paths = []
        for data in (data1, data2):
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                pq.write_table(pa.Table.from_pylist(data), f.name)
                paths.append(f.name)

        yield paths, data1 + data2

        for path in paths:
            os.unlink(path)

    def test_composite_reader_global_slice(self, multi_jsonl_files):
        """Test concatenated multi-file reader with an outer/global slice."""
        from relax.utils.data.streaming_dataset import CompositeStreamingReader

        paths, combined = multi_jsonl_files
        reader = CompositeStreamingReader(paths, slice(2, 5))

        assert len(reader) == 3
        assert reader[0] == combined[2]
        assert reader[1] == combined[3]
        assert reader[2] == combined[4]

    def test_composite_reader_iter_batch(self, multi_jsonl_files):
        """Test batch access across file boundaries preserves requested
        order."""
        from relax.utils.data.streaming_dataset import CompositeStreamingReader

        paths, combined = multi_jsonl_files
        reader = CompositeStreamingReader(paths, slice(1, 6))
        indices = [3, 0, 4]

        batch = list(reader.iter_batch(indices))

        assert [idx for idx, _ in batch] == indices
        assert [data for _, data in batch] == [combined[4], combined[1], combined[5]]

    def test_composite_reader_global_slice_parquet(self, multi_parquet_files):
        """Test concatenated parquet reader with an outer/global slice."""
        from relax.utils.data.streaming_dataset import CompositeStreamingReader

        paths, combined = multi_parquet_files
        reader = CompositeStreamingReader(paths, slice(1, 5))

        assert len(reader) == 4
        assert [reader[i] for i in range(len(reader))] == combined[1:5]

    def test_composite_reader_parallel_index_preserves_order(self, multi_jsonl_files, monkeypatch):
        from relax.utils.data.streaming_dataset import CompositeStreamingReader, StreamingReader

        paths, combined = multi_jsonl_files
        second_done = threading.Event()
        build_index = StreamingReader._build_index

        def build_out_of_order(reader):
            if reader.path == paths[0]:
                assert second_done.wait(timeout=5), "Indexing did not run concurrently"
            build_index(reader)
            if reader.path == paths[1]:
                second_done.set()

        monkeypatch.setattr(StreamingReader, "_build_index", build_out_of_order)
        reader = CompositeStreamingReader(paths)

        assert [reader[i] for i in range(len(reader))] == combined

    def test_composite_reader_parallel_index_propagates_error(self, multi_jsonl_files, monkeypatch):
        from relax.utils.data.streaming_dataset import CompositeStreamingReader, StreamingReader

        paths, _ = multi_jsonl_files
        build_index = StreamingReader._build_index

        def fail_index(reader):
            if reader.path == paths[1]:
                raise OSError("index read failed")
            build_index(reader)

        monkeypatch.setattr(StreamingReader, "_build_index", fail_index)
        with pytest.raises(OSError, match="index read failed"):
            CompositeStreamingReader(paths)

    def test_composite_reader_parallel_index_empty_shard_and_slice(self, tmp_path):
        from relax.utils.data.streaming_dataset import CompositeStreamingReader

        paths = [tmp_path / f"{i}.jsonl" for i in range(3)]
        paths[0].write_text('\n{"id": 0}\n{"id": 1}\n', encoding="utf-8")
        paths[1].write_text("\n  \n", encoding="utf-8")
        paths[2].write_text('{"id": 2}\n{"id": 3}', encoding="utf-8")
        reader = CompositeStreamingReader([str(path) for path in paths], slice(None, None, -2))

        assert len(reader) == 2
        assert list(reader.iter_batch([1, 0, 1])) == [(1, {"id": 1}), (0, {"id": 3}), (1, {"id": 1})]


class TestSampleBuffer:
    """Tests for SampleBuffer class."""

    def test_buffer_put_get(self):
        """Test basic put and get operations."""
        from relax.utils.data.streaming_dataset import SampleBuffer
        from relax.utils.types import Sample

        buffer = SampleBuffer(max_size=100)
        sample = Sample(prompt="test")

        buffer.put(0, sample)
        result = buffer.get(0)

        assert result is sample

    def test_buffer_cache_miss(self):
        """Test that cache miss returns None."""
        from relax.utils.data.streaming_dataset import SampleBuffer

        buffer = SampleBuffer(max_size=100)
        result = buffer.get(999)

        assert result is None

    def test_buffer_lru_eviction(self):
        """Test LRU eviction when buffer is full."""
        from relax.utils.data.streaming_dataset import SampleBuffer
        from relax.utils.types import Sample

        buffer = SampleBuffer(max_size=3)

        # Fill the buffer
        for i in range(3):
            buffer.put(i, Sample(prompt=f"sample_{i}"))

        # Access sample 0 to make it recently used
        buffer.get(0)

        # Add a new sample, should evict sample 1 (LRU)
        buffer.put(3, Sample(prompt="sample_3"))

        # Sample 1 should be evicted
        assert buffer.get(1) is None
        # Sample 0 and 2 should still be there
        assert buffer.get(0) is not None
        assert buffer.get(2) is not None
        assert buffer.get(3) is not None

    def test_buffer_hit_rate(self):
        """Test hit rate calculation."""
        from relax.utils.data.streaming_dataset import SampleBuffer
        from relax.utils.types import Sample

        buffer = SampleBuffer(max_size=100)
        buffer.put(0, Sample(prompt="test"))

        # 2 hits
        buffer.get(0)
        buffer.get(0)
        # 1 miss
        buffer.get(1)

        assert buffer.hit_rate == pytest.approx(2 / 3)

    def test_buffer_clear(self):
        """Test buffer clear operation."""
        from relax.utils.data.streaming_dataset import SampleBuffer
        from relax.utils.types import Sample

        buffer = SampleBuffer(max_size=100)
        buffer.put(0, Sample(prompt="test"))

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.get(0) is None


class TestPrefetchBuffer:
    """Tests for PrefetchBuffer race behavior."""

    def test_chunk_size_is_clamped_to_cache_capacity(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        def process_fn(idx: int) -> str:
            return f"sample-{idx}"

        buffer = PrefetchBuffer(process_fn, chunk_size=8, max_cached=1, num_workers=1)
        buffer.set_index_order([0])
        try:
            assert buffer.wait_for(0, timeout=2)
            found, sample = buffer.get_cached(0)
            assert found is True
            assert sample == "sample-0"
        finally:
            buffer.stop()

    def test_missed_inflight_index_is_not_stored_as_stale_cache(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        background_started = threading.Event()
        release_background = threading.Event()
        calls = []
        calls_lock = threading.Lock()

        def process_fn(idx: int) -> str:
            thread_name = threading.current_thread().name
            with calls_lock:
                calls.append((idx, thread_name))
            if idx == 0 and thread_name.startswith("pf"):
                background_started.set()
                assert release_background.wait(timeout=2)
            return f"sample-{idx}-{thread_name}"

        buffer = PrefetchBuffer(process_fn, chunk_size=1, max_cached=4, num_workers=1)
        buffer.set_index_order([0, 1])
        try:
            assert background_started.wait(timeout=2)

            sample = buffer.get(0)
            assert sample.startswith("sample-0-")

            release_background.set()
            for _ in range(200):
                with buffer._lock:
                    cached_keys = set(buffer._cache)
                    stale_drops = buffer._prefetch_stale_drops
                if 1 in cached_keys and stale_drops:
                    break
                time.sleep(0.01)

            with buffer._lock:
                assert 0 not in buffer._cache
                assert 1 in buffer._cache
                assert buffer._prefetch_stale_drops == 1

            assert any(idx == 0 and name.startswith("pf") for idx, name in calls)
            assert any(idx == 0 and not name.startswith("pf") for idx, name in calls)
        finally:
            release_background.set()
            buffer.stop()

    def test_wait_for_returns_false_when_no_thread_started(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        buffer = PrefetchBuffer(lambda idx: f"sample-{idx}", chunk_size=1, max_cached=1, num_workers=1)
        result = []

        waiter = threading.Thread(target=lambda: result.append(buffer.wait_for(0, timeout=None)), daemon=True)
        waiter.start()
        try:
            waiter.join(timeout=0.5)
            assert not waiter.is_alive()
            assert result == [False]
        finally:
            buffer.stop()

    def test_completed_samples_publish_and_refill_before_slow_sample(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        release = threading.Event()
        refill_started = threading.Event()
        calls = []

        def process(idx):
            calls.append(idx)
            if idx == 0:
                assert release.wait(3)
            if idx == 2:
                refill_started.set()
            return None if idx == 1 else idx

        buffer = PrefetchBuffer(process, chunk_size=2, max_cached=4, num_workers=2)
        buffer.set_index_order(list(range(6)))
        try:
            assert buffer.wait_for(1, timeout=2)
            assert refill_started.wait(2), "idle worker must refill before sample 0 completes"
            with buffer._lock:
                assert 0 not in buffer._cache
                assert buffer._cache[1] is None
            release.set()
            result = []
            for idx in range(6):
                assert buffer.wait_for(idx, timeout=2)
                found, sample = buffer.get_cached(idx)
                assert found
                result.append(sample)
            assert result == [0, None, 2, 3, 4, 5]
            assert sorted(calls) == list(range(6))
        finally:
            release.set()
            buffer.stop()

    def test_cache_and_outstanding_work_share_capacity(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        release = threading.Event()
        fourth_started = threading.Event()
        calls = []

        def process(idx):
            calls.append(idx)
            if idx == 0:
                assert release.wait(3)
            if idx == 3:
                fourth_started.set()
            return idx

        buffer = PrefetchBuffer(process, chunk_size=3, max_cached=3, num_workers=3)
        buffer.set_index_order(list(range(8)))
        try:
            assert buffer.wait_for(1, timeout=2)
            assert buffer.wait_for(2, timeout=2)
            assert not fourth_started.wait(0.15)
            assert sorted(calls) == [0, 1, 2]
            assert buffer.get_cached(1) == (True, 1)
            assert fourth_started.wait(2)
            assert buffer.wait_for(3, timeout=2)
            with buffer._lock:
                assert set(buffer._cache) == {2, 3}
        finally:
            release.set()
            buffer.stop()

    def test_timed_out_running_task_keeps_slot_and_discards_late_result(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        release = threading.Event()
        next_started = threading.Event()

        def process(idx):
            if idx == 0:
                assert release.wait(3)
            else:
                next_started.set()
            return idx

        buffer = PrefetchBuffer(process, chunk_size=1, max_cached=1, num_workers=1)
        buffer._PROCESS_TIMEOUT_SECONDS = 0.05
        buffer.set_index_order([0, 1])
        try:
            assert buffer.wait_for(0, timeout=2)
            assert buffer.get_cached(0) == (True, None)
            assert not next_started.wait(0.15)
            release.set()
            assert buffer.wait_for(1, timeout=2)
            assert buffer.get_cached(1) == (True, 1)
            assert buffer.get_cached(0) == (False, None)
        finally:
            release.set()
            buffer.stop()

    def test_epoch_reset_cancels_queued_work_and_ignores_old_result(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        old_started = threading.Event()
        old_release = threading.Event()
        old_done = threading.Event()
        new_started = threading.Event()
        new_release = threading.Event()

        def process(idx):
            if idx == 0:
                old_started.set()
                assert old_release.wait(3)
                old_done.set()
            if idx == 2:
                new_started.set()
                assert new_release.wait(3)
            return idx

        buffer = PrefetchBuffer(process, chunk_size=2, max_cached=2, num_workers=1)
        buffer.set_index_order([0, 1])
        try:
            assert old_started.wait(2)
            buffer.set_index_order([2])
            assert new_started.wait(2)
            old_release.set()
            assert old_done.wait(2)
            assert not buffer._finished
            new_release.set()
            assert buffer.wait_for(2, timeout=2)
            assert buffer.get_cached(2) == (True, 2)
            with buffer._lock:
                assert not buffer._cache
        finally:
            old_release.set()
            new_release.set()
            buffer.stop()

    def test_executor_initialization_failure_finishes_generation(self, monkeypatch):
        from relax.utils.data import streaming_dataset

        def fail_executor(**kwargs):
            raise RuntimeError("executor unavailable")

        buffer = streaming_dataset.PrefetchBuffer(lambda idx: idx)
        buffer._finished = False
        monkeypatch.setattr(streaming_dataset, "ThreadPoolExecutor", fail_executor)
        with pytest.raises(RuntimeError, match="executor unavailable"):
            buffer._run(buffer._stop)
        assert buffer._finished
        assert buffer.wait_for(0) is False

    def test_waiter_wakes_when_stream_finishes_without_requested_index(self):
        from relax.utils.data.streaming_dataset import PrefetchBuffer

        release = threading.Event()
        buffer = PrefetchBuffer(lambda idx: release.wait(2) and idx, max_cached=2, num_workers=1)
        buffer.set_index_order([0])
        result = []
        waiter = threading.Thread(target=lambda: result.append(buffer.wait_for(99)), daemon=True)
        waiter.start()
        try:
            release.set()
            waiter.join(2)
            assert not waiter.is_alive()
            assert result == [False]
        finally:
            release.set()
            buffer.stop()

    def test_submission_retry_does_not_duplicate_already_submitted_samples(self, monkeypatch):
        import relax.utils.data.streaming_dataset as module

        original = module.ThreadPoolExecutor
        calls = []

        class FlakyExecutor(original):
            attempts = 0

            def submit(self, fn, *args, **kwargs):
                self.attempts += 1
                if self.attempts == 2:
                    raise RuntimeError("temporary submit failure")
                return super().submit(fn, *args, **kwargs)

        monkeypatch.setattr(module, "ThreadPoolExecutor", FlakyExecutor)

        def process(idx):
            calls.append(idx)
            if idx == 1:
                raise ValueError("bad sample")
            return idx

        buffer = module.PrefetchBuffer(process, chunk_size=3, max_cached=3, num_workers=2)
        buffer.set_index_order([0, 1, 2])
        try:
            for idx, expected in [(0, 0), (1, None), (2, 2)]:
                assert buffer.wait_for(idx, timeout=2)
                assert buffer.get_cached(idx) == (True, expected)
            assert sorted(calls) == [0, 1, 2]
        finally:
            buffer.stop()


class TestIndexManager:
    """Tests for IndexManager class."""

    def test_shuffle_reproducibility(self):
        """Test that shuffle is reproducible with same seed."""
        from relax.utils.data.streaming_dataset import IndexManager

        manager1 = IndexManager(total_size=100, seed=42)
        manager2 = IndexManager(total_size=100, seed=42)

        manager1.shuffle(0)
        manager2.shuffle(0)

        assert manager1.indices == manager2.indices

    def test_shuffle_matches_huggingface_dataset(self):
        """Match ms-swift's dataset shuffle followed by Megatron's sampler."""
        from relax.utils.data.streaming_dataset import IndexManager

        manager = IndexManager(total_size=100, seed=42)
        manager.shuffle(0)

        import torch

        dataset_seed = int(np.random.RandomState(42).randint(0, np.iinfo(np.int32).max))
        dataset_indices = np.random.default_rng(dataset_seed).permutation(100)
        sampler_indices = torch.randperm(100, generator=torch.Generator().manual_seed(0)).numpy()
        assert manager.indices == dataset_indices[sampler_indices].tolist()

    def test_shuffle_different_epochs(self):
        """Test that different epochs produce different shuffles."""
        from relax.utils.data.streaming_dataset import IndexManager

        manager = IndexManager(total_size=100, seed=42)

        manager.shuffle(0)
        indices_epoch0 = list(manager.indices)

        manager.shuffle(1)
        indices_epoch1 = list(manager.indices)

        assert indices_epoch0 != indices_epoch1

    def test_get_next_indices(self):
        """Test getting next indices."""
        from relax.utils.data.streaming_dataset import IndexManager

        manager = IndexManager(total_size=10, seed=42)

        indices1, crossed1 = manager.get_next_indices(3)
        indices2, crossed2 = manager.get_next_indices(3)

        assert len(indices1) == 3
        assert len(indices2) == 3
        assert crossed1 is False
        assert crossed2 is False

        # No overlap between consecutive calls
        assert set(indices1).isdisjoint(set(indices2))

    def test_epoch_boundary_crossing(self):
        """Test epoch boundary is detected correctly."""
        from relax.utils.data.streaming_dataset import IndexManager

        manager = IndexManager(total_size=5, seed=42)

        # Get 3 samples, then 3 more (should cross boundary)
        indices1, crossed1 = manager.get_next_indices(3)
        indices2, crossed2 = manager.get_next_indices(3)

        assert crossed1 is False
        assert crossed2 is True  # Should have crossed epoch boundary
        assert len(indices2) == 3

    def test_state_save_load(self):
        """Test state checkpoint and restore."""
        from relax.utils.data.streaming_dataset import IndexManager

        manager1 = IndexManager(total_size=100, seed=42)
        manager1.shuffle(5)
        manager1.get_next_indices(25)  # Move position

        state = manager1.get_state()

        manager2 = IndexManager(total_size=100, seed=42)
        manager2.load_state(state)

        assert manager2.current_epoch == manager1.current_epoch
        assert manager2.position == manager1.position


class TestStreamingDataset:
    """Tests for StreamingDataset class."""

    @pytest.fixture
    def jsonl_file(self):
        """Create a temporary JSONL file for testing."""
        data = [{"text": f"Sample {i} with some content", "label": f"label_{i}"} for i in range(20)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            filepath = f.name

        yield filepath, data
        os.unlink(filepath)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        return tokenizer

    def test_dataset_len(self, jsonl_file, mock_tokenizer):
        """Test that dataset returns correct length."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
        )

        assert len(dataset) == len(data)

    def test_get_batch(self, jsonl_file, mock_tokenizer):
        """Test getting a batch of samples."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
        )

        samples, crossed = dataset.get_batch(5)

        assert len(samples) == 5
        assert all(s is not None for s in samples)

    @pytest.mark.parametrize("batch_size", [8, 64, 100, 128, 150, 200])
    def test_streaming_dataset_no_prefetch_preserves_epoch_order(self, tmp_path, mock_tokenizer, batch_size):
        from relax.utils.data.streaming_dataset import IndexManager, StreamingDataset

        data = [{"text": f"Sample {i}"} for i in range(batch_size * 3)]
        path = tmp_path / "data.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in data))
        dataset = StreamingDataset(str(path), mock_tokenizer, processor=None, max_length=None)
        dataset.shuffle(0)
        order = list(dataset.index_manager.indices)

        prompts = []
        for batch_id in range(3):
            samples, crossed = dataset.get_batch(batch_size)
            prompts.extend(sample.prompt for sample in samples)
            assert crossed is False
            assert dataset.index_manager.position == (batch_id + 1) * batch_size
            assert dataset.index_manager.current_epoch == 0

        assert prompts == [data[idx]["text"] for idx in order]
        assert len(set(prompts)) == len(data)

        expected_manager = IndexManager(len(data))
        expected_manager.shuffle(1)
        samples, crossed = dataset.get_batch(1)
        assert crossed is True
        assert samples[0].prompt == data[expected_manager.indices[0]]["text"]
        assert dataset.index_manager.position == 1

    def test_streaming_dataset_no_prefetch_refills_filtered_samples(self, jsonl_file, mock_tokenizer):
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(filepath, mock_tokenizer, processor=None, max_length=5)
        dataset.shuffle(0)
        order = list(dataset.index_manager.indices)
        filtered = {order[pos] for pos in (1, 4, 9, 13, 18)}
        filtered_prompts = {data[idx]["text"] for idx in filtered}
        mock_tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [1] * (6 if text in filtered_prompts else 5)}
        valid_indices = [idx for idx in order if idx not in filtered]

        first, crossed = dataset.get_batch(8)
        assert crossed is False
        assert [sample.prompt for sample in first] == [data[idx]["text"] for idx in valid_indices[:8]]
        assert dataset.index_manager.position == order.index(valid_indices[7]) + 1

        second, crossed = dataset.get_batch(len(valid_indices) - 8)
        assert crossed is False
        assert [sample.prompt for sample in first + second] == [data[idx]["text"] for idx in valid_indices]
        # The final row is valid, so every row in this epoch has now been examined.
        assert dataset.index_manager.position == len(data)
        assert dataset._filter_count == len(filtered)
        assert dataset._total_processed == len(data)

    def test_shuffle_epoch(self, jsonl_file, mock_tokenizer):
        """Test that shuffle produces different order."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
            seed=42,
        )

        # Get first batch
        samples1, _ = dataset.get_batch(10)
        prompts1 = [s.prompt for s in samples1]

        # Reset and shuffle with different epoch
        dataset.index_manager.reset(epoch_id=1)
        samples2, _ = dataset.get_batch(10)
        prompts2 = [s.prompt for s in samples2]

        assert prompts1 != prompts2

    def test_buffer_caching(self, jsonl_file, mock_tokenizer):
        """Test that samples are cached in buffer."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
            buffer_size=100,
        )

        # Access some samples
        _ = dataset[0]
        _ = dataset[1]

        # Check buffer
        assert len(dataset.buffer) == 2
        assert dataset.buffer.get(0) is not None
        assert dataset.buffer.get(1) is not None

    def test_get_stats(self, jsonl_file, mock_tokenizer):
        """Test statistics collection."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
        )

        _ = dataset.get_batch(5)
        stats = dataset.get_stats()

        assert "total_size" in stats
        assert "buffer_size" in stats
        assert "buffer_hit_rate" in stats
        assert stats["total_size"] == len(data)

    def test_state_checkpoint(self, jsonl_file, mock_tokenizer):
        """Test state save and restore."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
            seed=42,
        )

        # Advance state
        _ = dataset.get_batch(7)
        state = dataset.get_state()
        assert state["position"] == 7
        assert state["epoch_id"] == 0
        expected = [data[idx]["text"] for idx in dataset.index_manager.indices[7:15]]

        # Create new dataset and restore
        dataset2 = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
            seed=42,
        )
        dataset2.load_state(state)

        assert dataset2.index_manager.position == dataset.index_manager.position
        assert dataset2.index_manager.current_epoch == dataset.index_manager.current_epoch
        for current in (dataset, dataset2):
            samples, crossed = current.get_batch(8)
            assert [sample.prompt for sample in samples] == expected
            assert crossed is False

    def test_samples_proxy(self, jsonl_file, mock_tokenizer):
        """Test samples proxy for compatibility."""
        from relax.utils.data.streaming_dataset import StreamingDataset

        filepath, data = jsonl_file
        dataset = StreamingDataset(
            path=filepath,
            tokenizer=mock_tokenizer,
            processor=None,
            max_length=None,
            prompt_key="text",
        )

        # Test slice access
        samples = dataset.samples[0:3]

        assert len(samples) == 3
        assert all(s is not None for s in samples)

    def test_dataset_multi_file_global_slice(self, mock_tokenizer):
        """Test StreamingDataset with multi-file path and outer slice
        semantics."""
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

            path = f"[{files[0]},{files[1]}]@[1:5]"
            dataset = StreamingDataset(
                path=path,
                tokenizer=mock_tokenizer,
                processor=None,
                max_length=None,
                prompt_key="text",
            )

            assert len(dataset) == 4
            prompts = [dataset[i].prompt for i in range(len(dataset))]
            assert prompts == ["A1", "A2", "B0", "B1"]
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)

    def test_dataset_multi_file_global_slice_get_batch_across_epoch(self, mock_tokenizer):
        """Test get_batch preserves the sliced multi-file domain across epoch
        wraparound."""
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

            path = f"[{files[0]},{files[1]}]@[1:5]"
            dataset = StreamingDataset(
                path=path,
                tokenizer=mock_tokenizer,
                processor=None,
                max_length=None,
                prompt_key="text",
            )

            dataset.shuffle(0)
            first_epoch_order = list(dataset.index_manager.indices)
            samples1, crossed1 = dataset.get_batch(3)
            samples2, crossed2 = dataset.get_batch(3)

            prompts1 = [sample.prompt for sample in samples1]
            assert crossed1 is False
            assert prompts1 == [dataset.reader[idx]["text"] for idx in first_epoch_order[:3]]
            prompts2 = [sample.prompt for sample in samples2]
            assert crossed2 is True
            expected_indices = first_epoch_order[3:] + dataset.index_manager.indices[:2]
            assert prompts2 == [dataset.reader[idx]["text"] for idx in expected_indices]
            assert dataset.index_manager.position == 2
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
