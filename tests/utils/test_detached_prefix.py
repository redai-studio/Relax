# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU float64 semantic prototype; NOT Qwen3.5/Megatron validation."""

import importlib.util
import sys
from pathlib import Path

import torch


path = Path(__file__).resolve().parents[2] / "relax/utils/training/detached_prefix.py"
spec = importlib.util.spec_from_file_location("detached_prefix_under_test", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
Cache, Key = mod.DetachedPrefixCache, mod.PrefixKey


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32, 8, dtype=torch.float64)
        self.qkv = torch.nn.Linear(8, 24, dtype=torch.float64)
        self.head = torch.nn.Linear(8, 32, dtype=torch.float64)

    def forward(self, ids, past=None):
        q, k, v = self.qkv(self.emb(ids)).chunk(3, -1)
        offset = 0 if past is None else past[0].shape[0]
        if past is not None:
            k, v = torch.cat((past[0], k)), torch.cat((past[1], v))
        mask = torch.arange(k.shape[0])[None, :] <= (torch.arange(q.shape[0]) + offset)[:, None]
        scores = (q @ k.T / 8**0.5).masked_fill(~mask, float("-inf"))
        return self.head(scores.softmax(-1) @ v), (k, v)


def run(mode, initial):
    model = TinyAttention()
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    cache = Cache(mode, 100000)
    outputs = []
    for prefix in (4, 8):
        ids = torch.arange(prefix + 3)
        if mode == "baseline":
            output, _ = model(ids)
            output = output[prefix:]
        else:
            key = Key("episode", ("causal", "text", "adapter-none"), tuple(range(prefix)), tuple(range(prefix)))
            state = cache.prepare(key, lambda start, past: model(ids[start:prefix], past)[1])
            assert all(t.grad_fn is None and not t.requires_grad for t in state)
            output, _ = model(ids[prefix:], state)
        outputs.append(output.detach())
        output.square().mean().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    assert sum(g.abs().sum() for g in grads) > 0
    optimizer.step()
    cache.bump_version()
    assert not cache.entries
    return outputs, grads, list(model.parameters()), cache


def test_forward_gradients_update_and_extension():
    torch.manual_seed(17)
    initial = TinyAttention().state_dict()
    b, d, r = [run(mode, initial) for mode in ("baseline", "detached_prefix", "detached_prefix_reuse")]
    for xs, ys in ((b[0], d[0]), (d[0], r[0]), (d[1], r[1]), (d[2], r[2])):
        for x, y in zip(xs, ys):
            torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)
    assert any(not torch.allclose(x, y) for x, y in zip(b[1], d[1]))
    assert d[3].computed_tokens == 12
    assert r[3].computed_tokens == 8 and r[3].extensions == 1


def test_identity_mutation_eviction_and_invalidation():
    cache = Cache("detached_prefix_reuse", 32)
    key = Key("a", ("text",), (1, 2), (0, 1))

    def build(start, state):
        assert not torch.is_grad_enabled()
        return (torch.ones(4),)

    state = cache.prepare(key, build)
    state[0].zero_()
    assert cache.prepare(key, build)[0].sum() == 4
    for changed in (
        Key("b", key.semantics, key.tokens, key.positions),
        Key("a", ("image",), key.tokens, key.positions),
        Key("a", key.semantics, (1, 3), key.positions),
        Key("a", key.semantics, key.tokens, (1, 2)),
    ):
        before = cache.builds
        cache.prepare(changed, build)
        assert cache.builds == before + 1
        assert cache.bytes <= 32
    cache.bump_version()
    before = cache.builds
    cache.prepare(key, build)
    assert cache.builds == before + 1
