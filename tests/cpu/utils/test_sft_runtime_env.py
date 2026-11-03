# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.sft.runtime import sft_tq_num_shards
from relax.utils.env import Envs, known_env_names
from relax.utils.utils import post_process_env


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        fully_async=False,
        use_dynamic_batch_size=False,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        over_sampling_batch_size=1,
    )


def test_sft_tq_shards_is_registered_and_typed(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "3")
    args = SimpleNamespace(loss_type="sft", sft_async_prepack=True)

    assert "RELAX_SFT_TQ_SHARDS" in known_env_names()
    assert Envs.RELAX_SFT_TQ_SHARDS == 3
    assert sft_tq_num_shards(args) == 3


def test_post_process_env_propagates_sft_tq_shards(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "4")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")

    runtime_env = post_process_env(_make_args(), {"env_vars": {}})

    assert runtime_env["env_vars"]["RELAX_SFT_TQ_SHARDS"] == "4"


def test_post_process_env_preserves_configured_sft_tq_shards(monkeypatch):
    monkeypatch.setenv("RELAX_SFT_TQ_SHARDS", "4")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")

    runtime_env = post_process_env(_make_args(), {"env_vars": {"RELAX_SFT_TQ_SHARDS": "2"}})

    assert runtime_env["env_vars"]["RELAX_SFT_TQ_SHARDS"] == "2"


def test_post_process_env_propagates_kernel_cache_session(monkeypatch):
    monkeypatch.setenv("RELAX_KERNEL_CACHE_DIR", "/shared/cache")
    monkeypatch.setenv("RELAX_KERNEL_CACHE_LOCAL_DIR", "/local/cache")
    monkeypatch.setenv("RELAX_KERNEL_CACHE_SESSION_ID", "session-a")
    monkeypatch.setenv("RELAX_KERNEL_CACHE_KEY", "profile-a")
    monkeypatch.setenv("RELAX_KERNEL_CACHE_BUILD_FINGERPRINT", "build-a")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")

    runtime_env = post_process_env(_make_args(), {"env_vars": {}})

    assert runtime_env["env_vars"]["RELAX_KERNEL_CACHE_DIR"] == "/shared/cache"
    assert runtime_env["env_vars"]["RELAX_KERNEL_CACHE_LOCAL_DIR"] == "/local/cache"
    assert runtime_env["env_vars"]["RELAX_KERNEL_CACHE_SESSION_ID"] == "session-a"
    assert runtime_env["env_vars"]["RELAX_KERNEL_CACHE_KEY"] == "profile-a"
    assert runtime_env["env_vars"]["RELAX_KERNEL_CACHE_BUILD_FINGERPRINT"] == "build-a"


def test_post_process_env_preserves_configured_kernel_cache_session(monkeypatch):
    monkeypatch.setenv("RELAX_KERNEL_CACHE_LOCAL_DIR", "/driver/cache")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")

    runtime_env = post_process_env(_make_args(), {"env_vars": {"RELAX_KERNEL_CACHE_LOCAL_DIR": "/configured/cache"}})

    assert runtime_env["env_vars"]["RELAX_KERNEL_CACHE_LOCAL_DIR"] == "/configured/cache"


def test_post_process_env_propagates_jsonl_index_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_JSONL_INDEX_WORKERS", "8")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")
    runtime_env = post_process_env(_make_args(), {"env_vars": {}})
    assert runtime_env["env_vars"]["RELAX_JSONL_INDEX_WORKERS"] == "8"


def test_post_process_env_preserves_jsonl_index_settings(monkeypatch):
    monkeypatch.setenv("RELAX_JSONL_INDEX_WORKERS", "8")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")
    configured = {"RELAX_JSONL_INDEX_WORKERS": "4"}
    runtime_env = post_process_env(_make_args(), {"env_vars": configured.copy()})
    assert all(runtime_env["env_vars"][key] == value for key, value in configured.items())


@pytest.mark.parametrize("configured,expected", [({}, "false"), ({"TOKENIZERS_PARALLELISM": "true"}, "true")])
def test_post_process_env_tokenizer_passthrough(monkeypatch, configured, expected):
    monkeypatch.setenv("RELAX_PROPAGATE_ENV_VARS", "TOKENIZERS_PARALLELISM,OMP_NUM_THREADS,RELAX_PROCESSOR_POOL_SIZE")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("RELAX_PROCESSOR_POOL_SIZE", "96")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")
    env = post_process_env(_make_args(), {"env_vars": configured.copy()})["env_vars"]
    assert env["TOKENIZERS_PARALLELISM"] == expected
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["RELAX_PROCESSOR_POOL_SIZE"] == "96"


def test_post_process_env_keeps_tokenizer_default(monkeypatch):
    monkeypatch.delenv("RELAX_PROPAGATE_ENV_VARS", raising=False)
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")
    assert post_process_env(_make_args(), {"env_vars": {}})["env_vars"]["TOKENIZERS_PARALLELISM"] == "true"


def test_nonfinite_delivery_policy_reaches_training_workers(monkeypatch):
    monkeypatch.setenv("RELAX_SKIP_NONFINITE_GRAD", "1")
    # Qwen38 has its own passthrough list; it must not erase this policy.
    monkeypatch.setenv("RELAX_PROPAGATE_ENV_VARS", "QWEN38_WORKSPACE,SAVE_DIR")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _addr: "127.0.0.1")
    env = post_process_env(_make_args(), {"env_vars": {"OMP_NUM_THREADS": "24"}})["env_vars"]
    assert env["RELAX_SKIP_NONFINITE_GRAD"] == "1"
    assert env["OMP_NUM_THREADS"] == "24"
