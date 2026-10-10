# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
from types import SimpleNamespace

import pytest


def _skip_if_server_args_stubbed(se):
    """Skip under the dependency-stub environment: suite-wide stub tests can
    replace ``sglang.srt.server_args`` while ``sglang_engine`` is imported,
    leaving its module-level ``ServerArgs`` bound to a non-dataclass stub that
    ``_compute_genrm_server_args`` cannot iterate.

    Check what the engine module actually sees, not a fresh import.
    """
    if not dataclasses.is_dataclass(getattr(se, "ServerArgs", None)):
        pytest.skip("sglang_engine.ServerArgs is stubbed in this suite run")


@pytest.mark.parametrize(
    ("enable_mtp_training", "speculative_algorithm", "overrides", "expected"),
    [
        (True, "EAGLE", None, False),
        (False, None, None, False),
        (False, "EAGLE", None, True),
        (False, None, {"speculative_algorithm": "EAGLE"}, True),
        (False, "EAGLE", {"speculative_algorithm": None}, False),
    ],
)
def test_draft_weights_cpu_backup_follows_mtp_and_speculative_config(
    enable_mtp_training, speculative_algorithm, overrides, expected
):
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.backends.sglang.sglang_engine import _enable_draft_weights_cpu_backup

    args = SimpleNamespace(
        enable_mtp_training=enable_mtp_training,
        sglang_speculative_algorithm=speculative_algorithm,
    )

    assert _enable_draft_weights_cpu_backup(args, overrides) is expected


def _genrm_server_args_namespace(engine_config=None):
    """Minimal args shape for _compute_genrm_server_args (single-GPU
    engine)."""
    return SimpleNamespace(
        genrm_model_path="/tmp/fake/Qwen3-0.6B",
        genrm_num_gpus=1,
        genrm_num_gpus_per_engine=1,
        genrm_engine_config=engine_config or {},
        num_gpus_per_node=4,
        rollout_num_gpus=0,
        rollout_num_gpus_per_engine=1,
        seed=42,
        offload_rollout=False,
        colocate=False,
        debug_rollout_only=True,
        actor_num_gpus_per_node=0,
        actor_num_nodes=1,
        use_rollout_routing_replay=False,
        fp16=False,
    )


def test_genrm_deterministic_attention_backend_keeps_radix_capable_fa3(monkeypatch, tmp_path):
    """Regression (final-code autoscaler acceptance): deterministic inference
    must not land on the generic flashinfer default on pre-Blackwell GPUs.

    SGLang applies the flashinfer default *before* its deterministic handler,
    which then force-disables the radix cache (flashinfer is not in SGLang's
    radix-supported deterministic set). Without prefix reuse, a 48-way
    concurrent scoring load re-prefills every long shared prompt until the
    scheduler wedges. The fix mirrors SGLang's own deterministic fallback: fa3
    on pre-Blackwell (radix stays available), flashinfer on Blackwell+.
    """
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.backends.sglang import sglang_engine as se

    _skip_if_server_args_stubbed(se)

    args = _genrm_server_args_namespace()
    monkeypatch.setattr(se, "_genrm_deterministic_attention_backend", lambda: "fa3")

    kwargs, _ = se._compute_genrm_server_args(args, 0, "tcp://127.0.0.1:1", 16001, "127.0.0.1", 16000)

    assert kwargs["enable_deterministic_inference"] is True
    assert kwargs["attention_backend"] == "fa3"
    assert kwargs["random_seed"] == 42


def test_genrm_attention_backend_engine_config_still_overrides(monkeypatch):
    """The per-instance --genrm-engine-config override keeps the highest
    priority: an operator can still pin a different attention backend."""
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.backends.sglang import sglang_engine as se

    _skip_if_server_args_stubbed(se)

    args = _genrm_server_args_namespace(engine_config={"attention_backend": "triton"})
    monkeypatch.setattr(se, "_genrm_deterministic_attention_backend", lambda: "fa3")

    kwargs, _ = se._compute_genrm_server_args(args, 0, "tcp://127.0.0.1:1", 16001, "127.0.0.1", 16000)

    assert kwargs["attention_backend"] == "triton"
    assert kwargs["enable_deterministic_inference"] is True


def test_genrm_attention_backend_respects_global_user_choice(monkeypatch):
    """Re-review finding: an explicit ``--sglang-attention-backend`` must keep
    winning over the deterministic probe.

    The sglang_* inheritance loop backfills only fields absent from ``kwargs``,
    so pinning the probe result unconditionally (including the probe's
    ``None``) silently disabled the user's global choice.
    """
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.backends.sglang import sglang_engine as se

    _skip_if_server_args_stubbed(se)

    args = _genrm_server_args_namespace()
    args.sglang_attention_backend = "triton"  # the user's explicit global choice
    monkeypatch.setattr(se, "_genrm_deterministic_attention_backend", lambda: "fa3")

    kwargs, _ = se._compute_genrm_server_args(args, 0, "tcp://127.0.0.1:1", 16001, "127.0.0.1", 16000)

    assert kwargs["attention_backend"] == "triton"
    assert kwargs["enable_deterministic_inference"] is True


def test_genrm_attention_backend_probe_failure_defers_to_global_and_default(monkeypatch):
    """A failed probe must not pin ``None`` over the inheritance: nothing is
    pinned (SGLang's default applies) and the user's global choice still flows
    through the sglang_* loop."""
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.backends.sglang import sglang_engine as se

    _skip_if_server_args_stubbed(se)

    monkeypatch.setattr(se, "_genrm_deterministic_attention_backend", lambda: None)

    # No user global and a failed probe: nothing is pinned.
    args = _genrm_server_args_namespace()
    kwargs, _ = se._compute_genrm_server_args(args, 0, "tcp://127.0.0.1:1", 16001, "127.0.0.1", 16000)
    assert "attention_backend" not in kwargs

    # The user's global choice survives the failed probe via the loop.
    args_user = _genrm_server_args_namespace()
    args_user.sglang_attention_backend = "flashinfer"
    kwargs_user, _ = se._compute_genrm_server_args(args_user, 0, "tcp://127.0.0.1:1", 16001, "127.0.0.1", 16000)
    assert kwargs_user["attention_backend"] == "flashinfer"


def test_genrm_deterministic_attention_backend_arch_table(monkeypatch):
    """The arch probe maps Blackwell+ to flashinfer (SGLang's own deterministic
    choice there) and everything else to fa3; a failed probe defers to SGLang's
    default (None)."""
    pytest.importorskip("sglang.srt.utils", exc_type=ImportError)

    from relax.backends.sglang import sglang_engine as se

    def _probe(sm100, sm120, expected):
        monkeypatch.setattr("sglang.srt.utils.is_sm100_supported", lambda: sm100)
        monkeypatch.setattr("sglang.srt.utils.is_sm120_supported", lambda: sm120)
        assert se._genrm_deterministic_attention_backend() == expected

    _probe(True, False, "flashinfer")
    _probe(False, True, "flashinfer")
    _probe(False, False, "fa3")


def test_genrm_deterministic_attention_backend_probe_failure_defers(monkeypatch):
    """When the architecture probe cannot run, the helper defers to SGLang's
    default instead of guessing."""
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    import builtins

    from relax.backends.sglang import sglang_engine as se

    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name == "sglang.srt.utils":
            raise ImportError("probe unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert se._genrm_deterministic_attention_backend() is None
