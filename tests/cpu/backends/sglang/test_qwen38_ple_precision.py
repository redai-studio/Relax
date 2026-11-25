# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real SGLang precision dispatch for mixed Qwen FP8 exports on CPU.

The existing PLE copy tests do not check table allocation precision. Importing
the installed SGLang adds a few seconds but needs no model or CUDA allocation.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from relax.utils.quant_cast.fp8 import build_quantization_config


@pytest.fixture
def ple_precision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[type, type, Callable]:
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path))
    config_mod = pytest.importorskip("sglang.srt.configs.qwen4_exp")
    fp8_mod = pytest.importorskip("sglang.srt.layers.quantization.fp8")
    model_mod = pytest.importorskip("sglang.srt.models.qwen4_exp")
    return config_mod.Qwen4ExpTextConfig, fp8_mod.Fp8Config, model_mod._ple_table_is_fp8


@pytest.mark.parametrize(
    "dtype,excluded,quantized,expected",
    [
        ("bfloat16", True, True, False),
        ("bfloat16", False, True, True),
        ("float8_e4m3fn", True, True, True),
        ("bfloat16", False, False, False),
    ],
)
def test_qwen38_ple_precision_respects_fp8_exclusions(
    ple_precision: tuple[type, type, Callable], dtype: str, excluded: bool, quantized: bool, expected: bool
) -> None:
    config_class, quant_class, table_is_fp8 = ple_precision
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    config = config_class(ple_embedding_dtype=dtype)
    quant_config = (
        quant_class.from_config(build_quantization_config("block", [128, 128], [prefix] if excluded else []))
        if quantized
        else None
    )

    assert table_is_fp8(config, quant_config, prefix) is expected
