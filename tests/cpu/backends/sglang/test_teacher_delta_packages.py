# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import struct

import numpy as np
import pytest
import torch

from examples.on_policy_distillation.mopd import pack_teacher_pair as pack
from relax.backends.sglang.adts import package_teacher_ids
from relax.backends.sglang.adts.bitpack import encode_frame
from relax.backends.sglang.adts_teacher_loader import _verify_base


def test_initial_teacher_route_is_independent_of_route_order(tmp_path):
    routes = {"code": str(tmp_path / "Teacher2"), "math": str(tmp_path / "Teacher1")}
    teachers = {name: {"checkpoint_path": name} for name in ("Teacher1", "Teacher2")}
    for name in teachers:
        (tmp_path / name).mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"initial_teacher": "Teacher1", "teachers": teachers}))

    expected = {"math": "Teacher1", "code": "Teacher2"}
    assert package_teacher_ids(str(tmp_path), routes) == (expected, "math")
    assert package_teacher_ids(str(tmp_path), dict(reversed(list(routes.items())))) == (expected, "math")

    alias = tmp_path / "alias"
    alias.symlink_to(routes["math"], target_is_directory=True)
    aliased = {"math_extra": str(alias), **routes}
    assert package_teacher_ids(str(tmp_path), aliased) == ({**expected, "math_extra": "Teacher1"}, "math")


def test_changed_checkpoint_at_same_path_fails_base_verification():
    weights = {"w": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    header = {
        "provenance": {"source": "A", "target": "B"},
        "tensors": {"w": {"shape": [2], "dtype": "torch.bfloat16", "a_sha256": pack._hash_tensor(weights["w"])}},
    }
    _verify_base(weights, weights, header, "A")
    weights["w"][0] = 3
    with pytest.raises(ValueError, match="base weight mismatch: w"):
        _verify_base(weights, weights, header, "A")


@pytest.fixture
def checkpoint_captures(tmp_path, monkeypatch):
    from unittest.mock import Mock

    inputs = {}
    for i, domain in enumerate(("math", "code", "if")):
        path = tmp_path / domain / "checkpoint"
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        inputs[path] = torch.tensor([0xFFFE + i & 65535, 3 - i, 100 + i], dtype=torch.uint16).view(torch.bfloat16)

    monkeypatch.setenv("RELAX_ADTS_CAPTURE_FILE", "")

    def spawn(fn, args, nprocs):
        checkpoint, _, tp_size = args
        engine = Mock()

        def update(named_tensors, load_format):
            for rank in range(tp_size):
                model = torch.nn.Module()
                words = inputs[checkpoint].view(torch.uint16).int() * (-1 if rank % 2 else 1) + 16 * rank
                model.register_parameter("w", torch.nn.Parameter(words.to(torch.uint16).view(torch.bfloat16)))
                model.register_parameter("unchanged", torch.nn.Parameter(torch.ones(1)))
                model.register_buffer("aux", torch.tensor([1]))
                monkeypatch.setattr(
                    "sglang.srt.distributed.get_tp_group", Mock(return_value=Mock(rank_in_group=rank)), raising=False
                )
                pack.capture_teacher(model, named_tensors)
            return True, ""

        engine.update_weights_from_tensor.side_effect = update
        factory = Mock(return_value=engine)
        monkeypatch.setattr("sglang.Engine", factory)
        fn(0, *args)
        assert factory.call_args.kwargs["tp_size"] == tp_size

    monkeypatch.setattr(torch.multiprocessing, "spawn", spawn)
    return inputs


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_three_checkpoints_publish_three_pairs_for_all_six_switches(tmp_path, checkpoint_captures, tp_size):
    weights = checkpoint_captures
    bank = tmp_path / "bank"
    pack.main(["--checkpoints", *map(str, weights), "--output", str(bank), "--tp-size", str(tp_size)])
    manifest = json.loads((bank / "manifest.json").read_text())
    routes = {path.parent.name: str(path) for path in weights}
    ids, initial_route = package_teacher_ids(str(bank), routes, tp_size)
    assert ids[initial_route] == manifest["initial_teacher"]
    targets = {ids[path.parent.name]: value for path, value in weights.items()}
    assert len(manifest["package_integrity"]) == 3 * tp_size
    decoded = set()
    for filename, record in manifest["package_integrity"].items():
        path = bank / filename
        assert pack.sha256_file(path) == record["sha256"]
        with path.open("rb") as stream:
            header = pack.bitpack.read_package_header(stream)
            (packet,) = header["packets"]
            (mode,) = struct.unpack("<I", stream.read(4))
            assert mode == 1
            b0, b1 = stream.read(2)
        source, target = header["provenance"]["source"], header["provenance"]["target"]
        assert set(header["tensors"]) == {"w", "unchanged"}
        assert packet["name"] == "w" and packet["n"] == 6
        # Decode the fixture's small signed changes independently of the encoder.
        codes = np.array([b0 & 15, b0 >> 4, b1 & 15], dtype=np.int32)
        delta = (codes >> 1) ^ -(codes & 1)
        for src, dst, sign in ((source, target, 1), (target, source, -1)):
            rank = record["rank"]
            before = targets[src].view(torch.uint16).numpy().astype(np.int32) * (-1 if rank % 2 else 1) + 16 * rank
            actual = ((before + sign * delta) & 65535).astype("<u2")
            expected = (
                targets[dst].view(torch.uint16).numpy().astype(np.int32) * (-1 if rank % 2 else 1) + 16 * rank
            ) & 65535
            np.testing.assert_array_equal(actual, expected)
            decoded.add((rank, src, dst))
    assert decoded == {(rank, src, dst) for rank in range(tp_size) for src in targets for dst in targets if src != dst}


def test_encode_numpy_teacher_weights():
    rng = np.random.default_rng(42)
    teacher_a = rng.normal(0, 0.02, size=(512, 512)).astype(np.float32)
    teacher_a[1, :2] = [0.3, 0.5]
    teacher_b = teacher_a.copy()
    # One row is one 512-word tile. Give the first four rows predictable modes.
    teacher_b[1] *= 1.01
    teacher_b[1, :2] = [0.3, 0.6]
    teacher_b[2] *= 1.5
    teacher_b[3] *= -1
    teacher_b[4:] += rng.normal(0, 2e-5, size=teacher_b[4:].shape).astype(np.float32)

    def bf16_words(weights):
        # Round finite float32 values to BF16, ties to even, then keep the high 16 bits.
        bits = weights.view("<u4")
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        return (rounded >> 16).astype("<u2")

    words_a, words_b = bf16_words(teacher_a), bf16_words(teacher_b)
    left = words_a.ravel().view(np.uint8)
    right = words_b.ravel().view(np.uint8)
    payload = encode_frame(left, right)

    table = np.frombuffer(payload, dtype="<u4", count=teacher_a.shape[0])
    modes = table & 3
    np.testing.assert_array_equal(modes[:4], [0, 1, 2, 3])
    # Row 1 begins with codes 0 (unchanged) and 52 (escaped from a 4-bit slot).
    row_start = table.nbytes + (int(table[1]) >> 2)
    assert payload[row_start] == 0xF0
    escape_pos = row_start + pack.bitpack.TILE // 2
    (code,) = struct.unpack_from("<H", payload, escape_pos)
    assert code == 52
    delta = (code >> 1) ^ -(code & 1)
    assert int(words_a[1, 1]) + delta == int(words_b[1, 1])
    assert int(words_b[1, 1]) - delta == int(words_a[1, 1])
