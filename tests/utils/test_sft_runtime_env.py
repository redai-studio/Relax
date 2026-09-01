# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

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
