# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint save-gating (train()-owned) and fail-fast config validation.

Covers the two framework gaps: the sync-colocate path never routes through
``components.Actor._maybe_save_model``, so the FSDP actor must own the save
decision; and rollout TP (multi-GPU engine) is accepted as long as the engine
GPUs tile the actor world (``actor_world % rollout_num_gpus_per_engine == 0``),
otherwise rejected fail-fast.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
import requests
import torch

from relax.backends.fsdp.actor import FSDPTrainRayActor
from relax.backends.fsdp.arguments import validate_generative_config


class _SaveSpy:
    """Minimal stand-in exposing just what ``_maybe_save`` touches."""

    def __init__(self, **args):
        self.args = SimpleNamespace(**args)
        self.saved: list = []
        self._checkpoint_failures = 0

    def save_model(self, rollout_id, force_sync=False):
        self.saved.append((rollout_id, force_sync))


def _spy(**over):
    base = dict(save="/tmp/ckpt", save_interval=20, num_rollout=100, rotate_ckpt=False)
    base.update(over)
    return _SaveSpy(**base)


def test_maybe_save_fires_on_interval():
    spy = _spy()
    FSDPTrainRayActor._maybe_save(spy, 19)  # (19 + 1) % 20 == 0
    assert spy.saved == [(19, False)]


def test_maybe_save_skips_between_intervals():
    spy = _spy()
    FSDPTrainRayActor._maybe_save(spy, 5)
    assert spy.saved == []


def test_maybe_save_forces_sync_on_final_step():
    spy = _spy(save_interval=20, num_rollout=100)
    FSDPTrainRayActor._maybe_save(spy, 99)  # final step, not an interval multiple
    assert spy.saved == [(99, True)]


def test_maybe_save_final_step_does_not_require_save_interval():
    spy = _spy(save_interval=None)
    FSDPTrainRayActor._maybe_save(spy, 99)
    assert spy.saved == [(99, True)]


def test_maybe_save_noop_without_save_dir():
    spy = _spy(save=None)
    FSDPTrainRayActor._maybe_save(spy, 19)
    assert spy.saved == []


def test_maybe_save_rotate_ckpt_saves_every_step():
    spy = _spy(rotate_ckpt=True, save_interval=None)
    FSDPTrainRayActor._maybe_save(spy, 3)
    assert spy.saved == [(3, False)]


def test_periodic_checkpoint_failure_is_best_effort():
    spy = _spy()

    def _fail(*_args, **_kwargs):
        raise OSError("disk full")

    spy.save_model = _fail
    FSDPTrainRayActor._maybe_save(spy, 19)
    assert spy._checkpoint_failures == 1


def test_final_checkpoint_failure_is_propagated():
    spy = _spy()
    error = OSError("disk full")

    def _fail(*_args, **_kwargs):
        raise error

    spy.save_model = _fail
    with pytest.raises(OSError) as exc_info:
        FSDPTrainRayActor._maybe_save(spy, 99)
    assert exc_info.value is error
    assert spy._checkpoint_failures == 1


class _SyncSaveSpy:
    def __init__(self, *, rollout_manager=True, offload_engine=True):
        self.rollout_manager = object() if rollout_manager else None
        self.offload_engine = offload_engine
        self.events: list = []

    def update_weights(self, *, offload_after_sync=True):
        self.events.append(("sync", offload_after_sync))
        return self.offload_engine

    def _maybe_save(self, rollout_id):
        self.events.append(("save", rollout_id))

    def _finish_weight_sync_memory_state(self, offload_engine):
        self.events.append(("release", offload_engine))


def test_sync_save_and_release_keeps_checkpoint_before_actor_sleep():
    spy = _SyncSaveSpy(offload_engine=True)
    FSDPTrainRayActor._sync_save_and_release(spy, 19)
    assert spy.events == [("sync", False), ("save", 19), ("release", True)]


def test_sync_save_and_release_saves_without_rollout_manager():
    spy = _SyncSaveSpy(rollout_manager=False)
    FSDPTrainRayActor._sync_save_and_release(spy, 19)
    assert spy.events == [("save", 19), ("release", False)]


def test_sync_save_and_release_restores_memory_state_when_final_save_fails():
    spy = _SyncSaveSpy(offload_engine=True)

    def _fail(_rollout_id):
        spy.events.append(("save", _rollout_id))
        raise OSError("disk full")

    spy._maybe_save = _fail
    with pytest.raises(OSError, match="disk full"):
        FSDPTrainRayActor._sync_save_and_release(spy, 99)
    assert spy.events == [("sync", False), ("save", 99), ("release", True)]


def test_empty_training_batch_does_not_advance_or_sync():
    from relax.utils.timer import Timer

    shell = object.__new__(FSDPTrainRayActor)
    shell.args = SimpleNamespace()
    shell.policy_version = 7
    shell._load_train_batches = lambda *_args: []
    shell._assert_microbatch_count_aligned = lambda *_args: None
    shell._sync_save_and_release = lambda *_args: pytest.fail("empty training data must not sync weights")
    try:
        with pytest.raises(RuntimeError, match="no training micro-batches"):
            FSDPTrainRayActor.train(shell, rollout_id=3)
        assert shell.policy_version == 7
    finally:
        if "train_wait" in Timer().start_time:
            Timer().end("train_wait")


@pytest.mark.parametrize(
    ("engine_offset", "expected"),
    [
        (0.0, ("mean_diff=+0.00000", "ratio=1.00000", "k3=0.00000")),
        (0.1, ("mean_diff=-0.10000", "ratio=0.90484", "k3=0.00484")),
    ],
)
def test_logp_parity_compares_engine_and_replay_means_directly(monkeypatch, engine_offset, expected):
    messages = []
    monkeypatch.setattr(
        "relax.backends.fsdp.actor.logger",
        SimpleNamespace(info=messages.append, warning=messages.append),
    )
    actor_logp = torch.tensor([[-1.25, -0.75]])
    batch = {
        # SGLang returns a per-element mean for every full-schedule step.
        "rollout_old_logp": torch.tensor([[0.0, -1.25 + engine_offset, 0.0, -0.75 + engine_offset]]),
        "sde_indices": torch.tensor([1, 3]),
        "sigmas": torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1]),
        # More than one latent element makes the former duplicate division observable.
        "image_x_t": torch.zeros(1, 2, 2, 3),
    }

    FSDPTrainRayActor._log_logp_parity(object(), batch, actor_logp, eta=0.7, sigma_max=0.99)

    parity = next(message for message in messages if message.startswith("[logp parity] n="))
    assert all(fragment in parity for fragment in expected)


def test_record_update_metrics_logs_global_advantage_moments():
    shell = object.__new__(FSDPTrainRayActor)
    shell.device = torch.device("cpu")
    shell._ratio_max_local = float("-inf")
    shell._ratio_min_local = float("inf")
    agg = {key: [] for key in ("loss", "ratio_mean", "ratio_std", "clip_fraction", "approx_kl", "adv_mean", "adv_std")}
    result = {
        "loss": torch.tensor(1.0),
        "metrics": {
            "ratio_mean": torch.tensor(1.0),
            "ratio_std": torch.tensor(0.0),
            "clip_fraction": torch.tensor(0.0),
            "approx_kl": torch.tensor(0.0),
            "ratio_max": torch.tensor(1.0),
            "ratio_min": torch.tensor(1.0),
        },
    }
    advantages = torch.tensor([-1.0, 0.0, 1.0])

    FSDPTrainRayActor._record_update_metrics(shell, agg, result, advantages, rollout_id=0, update_index=0)

    assert agg["adv_mean"] == pytest.approx([0.0])
    assert agg["adv_std"] == pytest.approx([float(advantages.std(unbiased=False))])


class _Resp:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _eval_shell(**over):
    args = dict(
        eval_interval=10,
        num_rollout=100,
        final_eval_poll_interval=0.0,
        final_eval_wait_timeout=1.0,
    )
    args.update(over)
    shell = object.__new__(FSDPTrainRayActor)
    shell._rank = 0
    shell.rollout_manager = object()
    shell.args = SimpleNamespace(**args)
    return shell


def test_run_step_evaluation_waits_for_final_eval(monkeypatch):
    calls = []
    done_checks = 0

    def _get(url, *args, **kwargs):
        del args, kwargs
        nonlocal done_checks
        calls.append(url)
        if url.endswith("/evaluate"):
            return _Resp()
        if url.endswith("/is_eval_done"):
            done_checks += 1
            return _Resp({"done": done_checks >= 2})
        raise AssertionError(url)

    monkeypatch.setattr("relax.utils.utils.get_serve_url", lambda _role: "http://rollout")
    monkeypatch.setattr(requests, "get", _get)

    FSDPTrainRayActor._run_step_evaluation(_eval_shell(), 99)

    assert calls == ["http://rollout/evaluate", "http://rollout/is_eval_done", "http://rollout/is_eval_done"]


def test_run_step_evaluation_does_not_wait_before_final_step(monkeypatch):
    calls = []

    def _get(url, *args, **kwargs):
        del args, kwargs
        calls.append(url)
        return _Resp()

    monkeypatch.setattr("relax.utils.utils.get_serve_url", lambda _role: "http://rollout")
    monkeypatch.setattr(requests, "get", _get)

    FSDPTrainRayActor._run_step_evaluation(_eval_shell(), 89)

    assert calls == ["http://rollout/evaluate"]


def _valid_args(**over):
    base = dict(
        train_backend="fsdp",
        generation_task="t2i",
        model_adapter_path="relax.models.qwen_image.adapter.QwenImageAdapter",
        model_path="/models/Qwen-Image",
        fsdp_trainable_mode="full",
        weight_sync_wire_dtype="bf16",
        rollout_num_gpus_per_engine=1,
        multimodal_keys={},
        reward_required_components=None,
        reward_component_weights=None,
        reward_runtime="cpu",
        generative_advantage_std_mode=None,
        # `eta` is required (both the rollout request and the actor replay read
        # it, and they must not disagree); step 0 is excluded because sigma==1
        # makes the sde-type diffusion coefficient singular.
        sampling_config={"num_inference_steps": 12, "eta": 0.7, "sde_indices": [1, 3, 5]},
        # Synchronous colocate is the only supported generative deployment mode.
        colocate=True,
        fully_async=False,
        hybrid=False,
        kl_coef=0.0,
        use_kl_loss=False,
    )
    base.update(over)
    return Namespace(**base)


def test_validate_requires_eta_in_sampling_config():
    """A missing `eta` must fail preflight rather than default silently.

    The rollout request and the actor replay both read it; they used to carry
    different defaults (0.7 vs 1.0), which anchored the PPO ratio to a Gaussian
    the sample was never drawn from with no error anywhere.
    """
    with pytest.raises(ValueError, match="must set 'eta'"):
        validate_generative_config(_valid_args(sampling_config={"num_inference_steps": 12, "sde_indices": [1, 3, 5]}))


@pytest.mark.parametrize("eta", [0, -0.1, float("nan"), float("inf"), float("-inf"), "bad"])
def test_validate_rejects_invalid_eta(eta):
    with pytest.raises(ValueError, match="eta must be finite and > 0"):
        validate_generative_config(
            _valid_args(sampling_config={"num_inference_steps": 12, "eta": eta, "sde_indices": [1, 3, 5]})
        )


@pytest.mark.parametrize("num_steps", [0, -1, 1.5, True, "bad"])
def test_validate_rejects_invalid_num_inference_steps(num_steps):
    with pytest.raises(ValueError, match="num_inference_steps"):
        validate_generative_config(
            _valid_args(sampling_config={"num_inference_steps": num_steps, "eta": 0.7, "sde_indices": [1]})
        )


@pytest.mark.parametrize("field", ["sde_indices", "sde_pool"])
@pytest.mark.parametrize("index", [-1, 12])
def test_validate_rejects_out_of_range_sde_steps(field, index):
    sampling = {"num_inference_steps": 12, "eta": 0.7, "sde_indices": [1]}
    sampling[field] = [index]
    with pytest.raises(ValueError, match="outside"):
        validate_generative_config(_valid_args(sampling_config=sampling))


def test_validate_rejects_unknown_sde_type():
    with pytest.raises(ValueError, match="sde_type must be"):
        validate_generative_config(
            _valid_args(
                sampling_config={"num_inference_steps": 12, "eta": 0.7, "sde_type": "unknown", "sde_indices": [1]}
            )
        )


@pytest.mark.parametrize("fraction", [[-0.1, 0.5], [0.5, 1.1], [0.5, 0.5], [float("nan"), 1.0], [0.0]])
def test_validate_rejects_invalid_sde_timestep_fraction(fraction):
    with pytest.raises(ValueError, match="sde_timestep_fraction"):
        validate_generative_config(
            _valid_args(
                sampling_config={
                    "num_inference_steps": 12,
                    "eta": 0.7,
                    "sde_type": "dance",
                    "num_sde_steps": 2,
                    "sde_timestep_fraction": fraction,
                }
            )
        )


def test_validate_rejects_sde_step_zero_but_allows_dance():
    """sigma==1 at step 0 makes the sde-type diffusion coefficient singular."""
    with pytest.raises(ValueError, match="SDE step 0"):
        validate_generative_config(
            _valid_args(sampling_config={"num_inference_steps": 12, "eta": 0.7, "sde_indices": [0, 2, 4]})
        )
    # 'dance' uses a constant coefficient, so step 0 is fine there.
    validate_generative_config(
        _valid_args(
            sampling_config={"num_inference_steps": 12, "eta": 0.7, "sde_type": "dance", "sde_indices": [0, 2]}
        )
    )


def test_validate_rejects_unknown_reward_runtime():
    """Yamls reach this validator before argparse, so the enum is checked here.

    `runtime: dedicated` used to be shipped in both task yamls; after that value
    was removed the run passed preflight and then died in argparse.
    """
    with pytest.raises(ValueError, match="reward_runtime must be one of"):
        validate_generative_config(_valid_args(reward_runtime="dedicated"))


def test_validate_rejects_unknown_generative_advantage_std_mode():
    with pytest.raises(ValueError, match="generative_advantage_std_mode"):
        validate_generative_config(_valid_args(generative_advantage_std_mode="global"))


def test_validate_accepts_batch_generative_advantage_std_mode():
    validate_generative_config(_valid_args(generative_advantage_std_mode="batch"))


def test_validate_accepts_single_gpu_engine():
    validate_generative_config(_valid_args())  # must not raise


def test_validate_accepts_tp_dividing_actor_world():
    # Rollout TP>1 is supported when the engine GPUs tile the actor world.
    validate_generative_config(
        _valid_args(rollout_num_gpus_per_engine=2, actor_num_gpus_per_node=8, actor_num_nodes=1)
    )


def test_validate_accepts_tp_when_actor_world_unknown():
    # Without actor world info the divisibility check is deferred to the first sync.
    validate_generative_config(_valid_args(rollout_num_gpus_per_engine=4))


def test_validate_rejects_tp_not_dividing_actor_world():
    with pytest.raises(ValueError, match="divisible by rollout_num_gpus_per_engine"):
        validate_generative_config(
            _valid_args(rollout_num_gpus_per_engine=3, actor_num_gpus_per_node=8, actor_num_nodes=1)
        )


def test_validate_rejects_tp_below_one():
    with pytest.raises(ValueError, match="must be >= 1"):
        validate_generative_config(_valid_args(rollout_num_gpus_per_engine=0))


def test_validate_skips_non_fsdp_backend():
    # Megatron path must be untouched by the generative fail-fast checks.
    validate_generative_config(_valid_args(train_backend="megatron", rollout_num_gpus_per_engine=8))


def test_validate_rejects_fsdp_critic_and_ppo_but_not_megatron():
    with pytest.raises(ValueError, match="critic-free FlowGRPO"):
        validate_generative_config(_valid_args(use_critic=True, advantage_estimator="grpo"))
    with pytest.raises(ValueError, match="critic-free FlowGRPO"):
        validate_generative_config(_valid_args(use_critic=False, advantage_estimator="ppo"))
    validate_generative_config(_valid_args(use_critic=False, advantage_estimator="grpo"))
    validate_generative_config(_valid_args(train_backend="megatron", use_critic=True, advantage_estimator="ppo"))


def test_validate_requires_colocate():
    with pytest.raises(ValueError, match="requires --colocate"):
        validate_generative_config(_valid_args(colocate=False))


def test_validate_rejects_fully_async():
    with pytest.raises(ValueError, match="fully_async is not supported"):
        validate_generative_config(_valid_args(fully_async=True))


def test_validate_rejects_hybrid():
    with pytest.raises(ValueError, match="hybrid mode is not supported"):
        validate_generative_config(_valid_args(hybrid=True))


def test_validate_rejects_kl():
    with pytest.raises(ValueError, match="kl_coef/use_kl_loss are not supported"):
        validate_generative_config(_valid_args(kl_coef=0.1))
    with pytest.raises(ValueError, match="kl_coef/use_kl_loss are not supported"):
        validate_generative_config(_valid_args(use_kl_loss=True))


def test_validate_rejects_unconsumed_flags():
    # Flags the shared parser accepts but this backend never reads must fail fast
    # rather than silently doing nothing.
    with pytest.raises(ValueError, match="use_dynamic_batch_size is not supported"):
        validate_generative_config(_valid_args(use_dynamic_batch_size=True))
    with pytest.raises(ValueError, match="max_tokens_per_gpu is not supported"):
        validate_generative_config(_valid_args(max_tokens_per_gpu=4096))
    with pytest.raises(ValueError, match="autoscaler_config is not supported"):
        validate_generative_config(_valid_args(autoscaler_config={"min": 1}))
    with pytest.raises(ValueError, match="save_hf .* is not supported"):
        validate_generative_config(_valid_args(save_hf="/tmp/hf"))
    with pytest.raises(ValueError, match="load_debug_rollout_data .* is not implemented"):
        validate_generative_config(_valid_args(load_debug_rollout_data="/tmp/dump"))


def test_validate_warns_but_accepts_inert_flags(caplog):
    # async_save is forced by slime_validate_args alongside --rotate-ckpt, so it
    # must warn rather than fail.
    validate_generative_config(_valid_args(async_save=True))


def test_lr_at_step_constant_by_default():
    from relax.backends.fsdp.actor import lr_at_step

    args = Namespace(lr_warmup_iters=0, fsdp_lr_scheduler="constant", lr_decay_iters=None, min_lr=0.0)
    assert lr_at_step(args, 0, 1e-5) == 1e-5
    assert lr_at_step(args, 500, 1e-5) == 1e-5


def test_lr_at_step_warmup_then_cosine_decays_to_min():
    from relax.backends.fsdp.actor import lr_at_step

    args = Namespace(lr_warmup_iters=10, fsdp_lr_scheduler="cosine", lr_decay_iters=110, min_lr=1e-6)
    assert lr_at_step(args, 0, 1e-4) == pytest.approx(1e-5)  # first warmup step
    assert lr_at_step(args, 9, 1e-4) == pytest.approx(1e-4)  # warmup complete
    assert lr_at_step(args, 110, 1e-4) == pytest.approx(1e-6)  # decayed to the floor
    mid = lr_at_step(args, 60, 1e-4)
    assert 1e-6 < mid < 1e-4


# ---------------------------------------------------------------------------
# LoRA boundary
# ---------------------------------------------------------------------------


def _lora_args(**over):
    base = dict(
        fsdp_trainable_mode="lora",
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.0,
        lora_merge_mode=True,
        lora_adapter_mode=False,
        fsdp_master_dtype="fp32",
        fsdp_cpu_offload=False,
    )
    base.update(over)
    return _valid_args(**base)


def test_validate_accepts_lora_merge_mode():
    validate_generative_config(_lora_args())


def test_validate_accepts_lora_adapter_mode_and_derives_transport():
    args = _lora_args(lora_merge_mode=False, lora_adapter_mode=True)
    validate_generative_config(args)
    # The transport follows the rollout path rather than being configured twice.
    assert args.weight_sync_mode == "adapter"


def test_validate_rejects_lora_mode_without_rank():
    with pytest.raises(ValueError, match="requires --lora-rank"):
        validate_generative_config(_lora_args(lora_rank=0))


def test_validate_rejects_rank_without_lora_mode():
    # Otherwise --lora-rank parses fine and trains nothing but the full model.
    with pytest.raises(ValueError, match="requires --fsdp-trainable-mode lora"):
        validate_generative_config(_valid_args(lora_rank=64))


def test_validate_rejects_lora_dropout():
    # A stochastic forward desynchronizes the FlowGRPO pi_old anchor from the
    # update, which silently collapses the clipped loss instead of erroring.
    with pytest.raises(ValueError, match="lora_dropout must be 0.0"):
        validate_generative_config(_lora_args(lora_dropout=0.1))


def test_validate_rejects_lora_with_cpu_offload():
    with pytest.raises(ValueError, match="fsdp-cpu-offload is not supported"):
        validate_generative_config(_lora_args(fsdp_cpu_offload=True))


def test_validate_rejects_adapter_transport_without_adapter_mode():
    with pytest.raises(ValueError, match="requires --lora-adapter-mode"):
        validate_generative_config(_valid_args(weight_sync_mode="adapter"))


def test_validate_rejects_weight_sync_mode_contradicting_lora_adapter_mode():
    """The transport is DERIVED from --lora-adapter-mode.

    It used to be silently overwritten, which made --weight-sync-mode a knob
    whose value was discarded; now a contradicting explicit value is an error.
    """
    args = _lora_args(lora_adapter_mode=True)
    args.weight_sync_mode = "full"  # parser default == "unset"
    validate_generative_config(args)
    assert args.weight_sync_mode == "adapter"

    args = _lora_args(lora_adapter_mode=False)
    args.weight_sync_mode = "adapter"
    with pytest.raises(ValueError, match="requires --lora-adapter-mode"):
        validate_generative_config(args)


def test_validate_warns_when_lora_lacks_fp32_masters(caplog):
    # Not a hard error (an operator may want the memory back), but the failure
    # mode -- a flat reward curve that reads as a too-low lr -- must be loud.
    with caplog.at_level("WARNING"):
        validate_generative_config(_lora_args(fsdp_master_dtype=None))
    assert any("fsdp-master-dtype fp32" in r.message for r in caplog.records)
