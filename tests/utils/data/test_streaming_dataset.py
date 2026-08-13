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

    def test_dataset_multi_file_global_slice_get_batch_across_epoch(self, mock_tokenizer, monkeypatch):
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

            monkeypatch.setattr("random.shuffle", lambda seq: None)

            path = f"[{files[0]},{files[1]}]@[1:5]"
            dataset = StreamingDataset(
                path=path,
                tokenizer=mock_tokenizer,
                processor=None,
                max_length=None,
                prompt_key="text",
            )

            samples1, crossed1 = dataset.get_batch(3)
            samples2, crossed2 = dataset.get_batch(3)

            assert [sample.prompt for sample in samples1] == ["A1", "A2", "B0"]
            prompts2 = [sample.prompt for sample in samples2]
            # The public contract is that batching stays within the sliced
            # multi-file domain and wraps across epochs when needed. The
            # exact second-batch ordering depends on internal over-fetch/cursor
            # strategy and should not be treated as externally stable.
            assert len(prompts2) == 3
            assert set(prompts2).issubset({"A1", "A2", "B0", "B1"})
            assert crossed2 is True
        finally:
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
