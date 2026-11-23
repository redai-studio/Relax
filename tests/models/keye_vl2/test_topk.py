# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye FlashInfer DSA selection requires CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 73, 2048, 2049, 4096, 32768])
def test_dsa_topk_prefers_earlier_tokens_at_tied_boundary(dtype: torch.dtype, width: int) -> None:
    from relax.models.keye_vl2.topk import dsa_topk_indices

    # Include all-tied and mixed-score rows, with many ties across the 2048th boundary.
    positions = torch.arange(width, device="cuda")
    scores = torch.stack([positions % 7, torch.zeros_like(positions), positions.flip(0) % 11]).to(dtype)
    expected = scores.argsort(dim=-1, descending=True, stable=True)[:, : min(2048, width)].sort(-1).values
    actual = dsa_topk_indices(scores)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual.sort(-1).values, expected, atol=0, rtol=0)
    torch.testing.assert_close(dsa_topk_indices(scores), actual, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dsa_topk_prefix_and_padded_decode_select_same_keys(dtype: torch.dtype) -> None:
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    width = 4096
    for length in (73, 2048, 2049, 3073):
        prefix = (torch.arange(length, device="cuda") % 3).to(dtype)[None]
        expected = prefix.argsort(dim=-1, descending=True, stable=True)[:, : min(2048, length)].sort(-1).values
        # Training uses the dtype's minimum; paged decode uses -inf for invalid keys.
        for padding in (torch.finfo(dtype).min, float("-inf")):
            padded = torch.full((1, width), padding, dtype=dtype, device="cuda")
            padded[:, :length] = prefix
            selected = dsa_topk_indices(padded)
            ordered, counts = compact_indices(selected, selected < length)
            torch.testing.assert_close(counts, torch.tensor([min(2048, length)], device="cuda", dtype=torch.int32))
            torch.testing.assert_close(ordered[:, : min(2048, length)].long(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dsa_topk_cuda_graph_replays_changed_scores_and_lengths(dtype: torch.dtype) -> None:
    from relax.models.keye_vl2.sparse_attention import compact_indices
    from relax.models.keye_vl2.topk import dsa_topk_indices

    width = 4096
    scores = torch.zeros((4, width), device="cuda", dtype=dtype)
    lengths = torch.full((4,), width, device="cuda", dtype=torch.int32)

    def select() -> tuple[torch.Tensor, torch.Tensor]:
        selected = dsa_topk_indices(scores)
        return compact_indices(selected, selected < lengths[:, None])

    for _ in range(3):
        select()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ordered, counts = select()
    positions = torch.arange(width, device="cuda")
    for iteration, new_lengths in enumerate(((73, 2048, 2051, 3073), (4096, 2100, 125, 2049))):
        lengths.copy_(torch.tensor(new_lengths, device="cuda", dtype=torch.int32))
        changed = ((positions[None] + torch.arange(4, device="cuda")[:, None]) % (3 + iteration)).to(dtype)
        changed.masked_fill_(positions[None] >= lengths[:, None], float("-inf"))
        scores.copy_(changed)
        graph.replay()
        for row, length in enumerate(new_lengths):
            count = min(2048, length)
            expected = changed[row, :length].argsort(descending=True, stable=True)[:count].sort().values
            torch.testing.assert_close(ordered[row, :count].long(), expected, atol=0, rtol=0)
        torch.testing.assert_close(counts, lengths.clamp_max(2048), atol=0, rtol=0)


@pytest.mark.parametrize("length", [17, 2048, 2049, 8192, 10240, 32768])
def test_bitmap_replay_roundtrip_and_row_selection(length: int) -> None:
    from relax.models.keye_vl2.replay_codec import decode_into, encode_into

    ids = torch.full((4, 2048), -1, device="cuda", dtype=torch.int32)
    count = min(length, 2048)
    ids[:, :count] = torch.rand(4, length, device="cuda").topk(count).indices.int()
    # A strided per-layer destination catches capture scratch indexing errors.
    scratch = torch.empty((4, 3, 1025), device="cuda", dtype=torch.int32)
    encode_into(ids, scratch[:, 1])
    packed = scratch[:, 1, : 1 + (length + 31) // 32].contiguous()
    rows = torch.tensor([3, 0], device="cuda")
    destinations = torch.tensor([1, 4], device="cuda")
    output = torch.full((6, 2048), -1, device="cuda", dtype=torch.int32)
    decode_into(packed, rows, destinations, output)
    torch.testing.assert_close(output[destinations].sort().values, ids[rows].sort().values, atol=0, rtol=0)
    assert (output[[0, 2, 3, 5]] == -1).all()


@pytest.mark.parametrize("codec_name,topk", [("keye-bitmap-v1", 2048), ("qsa-r4-bitmap-v1", 512)])
def test_bitmap_consumer_cpu_and_gpu_bodies_share_cache_and_row_semantics(codec_name, topk):
    import numpy as np

    from relax.distributed.replay_manager.config import ReplayConfig
    from relax.distributed.replay_manager.consumer import ConsumerBatchLayout, ConsumerLayerReader, ConsumerTensorCache
    from relax.distributed.replay_manager.core import ReplayInventory
    from relax.distributed.replay_manager.storage import TQReplayStorage
    from relax.models.keye_vl2.replay_codec import encode_into

    class Bodies:
        def __init__(self):
            self.values = {}
            self.reads = 0

        def write_many(self, keys, values):
            self.values.update((key, value.copy()) for key, value in zip(keys, values, strict=True))

        def read(self, keys):
            self.reads += 1
            return [self.values[key] for key in keys]

        def delete(self, keys):
            for key in keys:
                self.values.pop(key, None)

        def close(self):
            pass

    ids = torch.arange(topk, device="cuda", dtype=torch.int32).expand(2056, -1).clone()
    ids.masked_fill_(ids > torch.arange(2056, device="cuda")[:, None], -1)
    ids[-2:] += torch.tensor([[1], [2]], device="cuda", dtype=torch.int32)
    encoded = torch.empty((2056, 1025), device="cuda", dtype=torch.int32)
    if topk == 2048:
        encode_into(ids, encoded)
    else:
        from relax.distributed.replay_manager._bitmap import encode_qsa_tokens_into

        tokens = torch.where(ids[:, :, None] >= 0, ids[:, :, None] * 4 + torch.arange(4, device="cuda"), -1).flatten(1)
        tokens = torch.cat((tokens, torch.full((len(ids), 3), -1, dtype=torch.int32, device="cuda")), dim=1).int()
        encode_qsa_tokens_into(tokens, torch.full((len(ids),), 4 * 2056, dtype=torch.int32, device="cuda"), encoded)
    host = encoded.cpu().numpy()
    coordinates = np.array([[0, 16], [0, 0], [0, 2054], [0, 2055], [1, 17], [-1, 0]])
    layout = ConsumerBatchLayout.from_coordinates(coordinates)
    first_body_bytes = 1024 * 4
    for gpu_body, cache_bytes in ((False, 16 << 20), (True, 16 << 20), (True, first_body_bytes)):
        body = Bodies()
        config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=1024, publication_bytes=64 << 20)
        inventory = ReplayInventory(config, body.delete)
        store = TQReplayStorage(config, inventory=inventory)
        store._backend = body
        publication = store.begin_publication(kind="indexer", layers=1, topk=topk, codec=codec_name)
        for start, stop in [(0, 1024), (1024, 2048), (2048, 2056)]:
            if topk == 512:
                from relax.distributed.replay_manager.backend.qwen38 import QSABitmapCapturer

                width = QSABitmapCapturer.encoded_width(stop)
            else:
                width = 1 if stop <= topk else 1 + (stop + 31) // 32
            publication.append(stop - start, lambda layer: host[start:stop, :width].copy(), encoded_width=width)
        ref = publication.finish(list(range(2057)))
        # Preload actual GPU bodies to exercise device reception semantics;
        # this test does not claim to test an RDMA transport.
        if gpu_body:
            body.values = {key: torch.from_numpy(value).cuda() for key, value in body.values.items()}
            store.encoded_batch_limit = lambda **kwargs: 16 << 20
        cache = ConsumerTensorCache(cache_bytes)
        with store.reader([ref, ref]) as reader:
            source = ConsumerLayerReader(reader, layout, layer=0, kind="indexer", device=ids.device, cache=cache)
        try:
            first, mask = source.read()
            assert torch.equal(first[:5].sort().values, ids[[16, 0, 2054, 2055, 17]].sort().values)
            assert (first[5] == -1).all() and torch.equal(mask.cpu(), torch.tensor([True] * 5 + [False]))
            reads = body.reads
            second, _ = source.read()
            torch.testing.assert_close(second, first, atol=0, rtol=0)
            if cache_bytes >= 16 << 20:
                assert body.reads == reads
            else:
                assert cache.evictions > 0 and cache.gpu_bytes <= cache_bytes
            assert cache.encoded_h2d_bytes == (0 if gpu_body else cache.encoded_read_bytes)
            assert cache.encoded_rdma_bytes == (cache.encoded_read_bytes if gpu_body else 0)
            assert cache.gpu_peak_bytes > 0
            if gpu_body:
                assert cache.encoded_read_bodies > cache.encoded_read_batches
                assert cache.encoded_batch_peak_bytes <= 16 << 20
            if gpu_body and cache_bytes == 16 << 20:
                for key in list(cache.entries):
                    cache.discard(key)
                read_bodies = body.read

                def malformed_batch(keys):
                    values = read_bodies(keys)
                    values[-1] = values[-1].flatten()
                    return values

                body.read = malformed_batch
                with pytest.raises(ValueError, match="encoded replay body shape"):
                    source.read()
                assert not cache.entries
                body.read = read_bodies
                recovered, _ = source.read()
                torch.testing.assert_close(recovered, first, atol=0, rtol=0)
        finally:
            source.release()
            store.discard([ref])
            store.close_run()
        assert not body.values and not inventory.state["objects"]


def test_compacted_capture_matches_raw_codec_and_preserves_graph_offsets():
    from types import SimpleNamespace

    from relax.models.keye_vl2.replay_codec import encode_into
    from relax.models.keye_vl2.sglang.replay import KeyeBitmapCapturer

    ids = torch.zeros((4, 2048), device="cuda", dtype=torch.int32)
    counts = torch.zeros(4, device="cuda", dtype=torch.int32)
    scratch = torch.full((10, 2, 1025), 123456, device="cuda", dtype=torch.int32)
    capturer = KeyeBitmapCapturer.__new__(KeyeBitmapCapturer)
    capturer.device_cache = SimpleNamespace(buffer=scratch)
    for _ in range(3):
        capturer.capture_compacted(1, ids, counts, row_offset=3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capturer.capture_compacted(1, ids, counts, row_offset=3)
    positions = torch.arange(2048, device="cuda", dtype=torch.int32)
    for sizes, offsets in [([17, 2048, 2048, 0], [0, 0, 1, 0]), ([2048, 5, 100, 2048], [2, 0, 11, 4])]:
        counts.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
        ids.copy_(positions[None] + torch.tensor(offsets, device="cuda", dtype=torch.int32)[:, None])
        ids.masked_fill_(positions[None] >= counts[:, None], 0)
        graph.replay()
        normalized = ids.masked_fill(positions[None] >= counts[:, None], -1)
        expected = torch.empty((4, 1025), device="cuda", dtype=torch.int32)
        encode_into(normalized, expected)
        torch.testing.assert_close(scratch[3:7, 1], expected, atol=0, rtol=0)
        assert (scratch[:3] == 123456).all() and (scratch[7:] == 123456).all()
        assert (scratch[:, 0] == 123456).all()
