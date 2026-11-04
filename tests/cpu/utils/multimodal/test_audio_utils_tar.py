# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for tar-offset audio loading in load_audio_from_path."""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from relax.utils.multimodal.audio_utils import (
    _load_audio_from_tar_offset,
    load_audio_from_path,
)


def _make_wav_bytes(sample_rate: int = 16000, duration_s: float = 0.1) -> bytes:
    n_samples = int(sample_rate * duration_s)
    samples = (np.sin(np.linspace(0, 6.28, n_samples)) * 1000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_tar_with_payload(tmp_path: Path, payload: bytes) -> tuple[Path, int, int]:
    pad = b"X" * 1024
    tar_path = tmp_path / "shard.tar"
    tar_path.write_bytes(pad + payload + b"Y" * 32)
    return tar_path, len(pad), len(payload)


def test_load_audio_from_tar_offset_reads_bytes_at_offset(tmp_path):
    wav = _make_wav_bytes()
    tar_path, off, sz = _make_tar_with_payload(tmp_path, wav)
    spec = f"{tar_path}:clip.wav:{off}:{sz}"
    audio = _load_audio_from_tar_offset(spec, sample_rate=16000)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.shape[0] > 0


def test_load_audio_from_path_dispatches_tar_schema(tmp_path):
    wav = _make_wav_bytes()
    tar_path, off, sz = _make_tar_with_payload(tmp_path, wav)
    spec = f"{tar_path}:clip.wav:{off}:{sz}"
    audio = load_audio_from_path(spec)
    assert audio.ndim == 1
    assert audio.shape[0] > 0


def test_load_audio_from_path_plain_path_unchanged(tmp_path):
    wav = _make_wav_bytes()
    plain = tmp_path / "plain.wav"
    plain.write_bytes(wav)
    audio = load_audio_from_path(str(plain))
    assert audio.ndim == 1
    assert audio.shape[0] > 0


def test_load_audio_from_tar_offset_rejects_short_read(tmp_path):
    wav = _make_wav_bytes()
    tar_path, off, sz = _make_tar_with_payload(tmp_path, wav)
    spec = f"{tar_path}:clip.wav:{off}:{sz + 10_000_000}"
    with pytest.raises(OSError, match="short read"):
        _load_audio_from_tar_offset(spec, sample_rate=16000)
