# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint save-gating (train()-owned) and fail-fast config validation.

Covers the two framework gaps: the sync-colocate path never routes through
``components.Actor._maybe_save_model``, so the FSDP actor must own the save
decision; and rollout TP (multi-GPU engine) is accepted as long as the engine
GPUs tile the actor world (``actor_world % rollout_num_gpus_per_engine == 0``),
otherwise rejected fail-fast.
"""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import requests
import torch

from relax.backends.fsdp.actor import FSDPTrainRayActor
from relax.backends.fsdp.arguments import add_fsdp_arguments, add_generative_arguments, validate_generative_config
from tests.cpu.relax_parser import parser_for_flags
from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


class _SaveSpy:
    """Minimal stand-in exposing just what ``_maybe_save`` touches."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.saved: list = []
        self._checkpoint_failures = 0

    def save_model(self, rollout_id, force_sync=False):
        self.saved.append((rollout_id, force_sync))


@pytest.fixture
def checkpoint_parser(arguments_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ArgumentParser:
    router = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    monkeypatch.setattr(arguments_module, "RouterArgs", router)
    return arguments_module.get_slime_extra_args_provider()(ArgumentParser())


def _spy(checkpoint_parser: ArgumentParser) -> _SaveSpy:
    args = checkpoint_parser.parse_args(["--save", "/tmp/ckpt", "--save-interval", "20", "--num-rollout", "100"])
    return _SaveSpy(args)


def test_maybe_save_fires_on_interval(checkpoint_parser):
    spy = _spy(checkpoint_parser)
    FSDPTrainRayActor._maybe_save(spy, 19)  # (19 + 1) % 20 == 0
    assert spy.saved == [(19, False)]


def test_maybe_save_skips_between_intervals(checkpoint_parser):
    spy = _spy(checkpoint_parser)
    FSDPTrainRayActor._maybe_save(spy, 5)
    assert spy.saved == []


def test_maybe_save_forces_sync_on_final_step(checkpoint_parser):
    spy = _spy(checkpoint_parser)
    FSDPTrainRayActor._maybe_save(spy, 99)  # final step, not an interval multiple
    assert spy.saved == [(99, True)]


def test_maybe_save_final_step_does_not_require_save_interval(checkpoint_parser):
    spy = _SaveSpy(checkpoint_parser.parse_args(["--save", "/tmp/ckpt", "--num-rollout", "100"]))
    FSDPTrainRayActor._maybe_save(spy, 99)
    assert spy.saved == [(99, True)]


def test_maybe_save_final_step_keeps_zero_interval_short_circuit(checkpoint_parser: ArgumentParser) -> None:
    args = checkpoint_parser.parse_args(["--save", "/tmp/ckpt", "--num-rollout", "100", "--save-interval", "0"])
    spy = _SaveSpy(args)
    FSDPTrainRayActor._maybe_save(spy, 99)
    assert spy.saved == [(99, True)]


def test_maybe_save_noop_without_save_dir(checkpoint_parser):
    spy = _SaveSpy(checkpoint_parser.parse_args([]))
    FSDPTrainRayActor._maybe_save(spy, 19)
    assert spy.saved == []


def test_maybe_save_rejects_missing_declared_save_field() -> None:
    spy = _SaveSpy(Namespace())
    with pytest.raises(AttributeError, match="save"):
        FSDPTrainRayActor._maybe_save(spy, 19)
    assert spy.saved == []


@pytest.mark.parametrize("num_rollout", [None, 0], ids=["omitted", "zero"])
def test_maybe_save_periodic_without_final_count(checkpoint_parser: ArgumentParser, num_rollout: int | None) -> None:
    cli = ["--save", "/tmp/ckpt", "--save-interval", "20"]
    if num_rollout is not None:
        cli += ["--num-rollout", str(num_rollout)]
    spy = _SaveSpy(checkpoint_parser.parse_args(cli))
    FSDPTrainRayActor._maybe_save(spy, 19)
    assert spy.saved == [(19, False)]


def test_maybe_save_rotate_ckpt_saves_every_step(checkpoint_parser):
    spy = _SaveSpy(checkpoint_parser.parse_args(["--save", "/tmp/ckpt", "--num-rollout", "100", "--rotate-ckpt"]))
    FSDPTrainRayActor._maybe_save(spy, 3)
    assert spy.saved == [(3, False)]


def test_periodic_checkpoint_failure_is_best_effort(checkpoint_parser):
    spy = _spy(checkpoint_parser)

    def _fail(*_args, **_kwargs):
        raise OSError("disk full")

    spy.save_model = _fail
    FSDPTrainRayActor._maybe_save(spy, 19)
    assert spy._checkpoint_failures == 1


def test_final_checkpoint_failure_is_propagated(checkpoint_parser):
    spy = _spy(checkpoint_parser)
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
    shell.args = parser_for_flags("--loss-type").parse_args([])
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


@pytest.mark.parametrize("configured", [False, True], ids=["declared-defaults", "two-updates-zero-clip"])
def test_fsdp_training_uses_declared_update_count_and_clip(configured: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from relax.backends.fsdp import actor as actor_module
    from relax.utils.timer import Timer

    train_parser = parser_for_flags("--clip-grad", "--eps-clip", "--loss-type", "--offload-train")
    parser = add_generative_arguments(ArgumentParser(parents=[train_parser], add_help=False))
    cli = ["--generation-task", "t2i", "--sampling-config", '{"eta": 0.7}', "--no-offload-train"]
    if configured:
        cli += ["--num-updates-per-batch", "2", "--eps-clip", "0"]
    shell = object.__new__(FSDPTrainRayActor)
    shell.args = parser.parse_args(cli)
    shell._rank = 1
    shell._lr_steps = 0
    shell.model = torch.nn.Linear(1, 1)
    shell.adapter = object()
    events = []
    shell.optimizer = SimpleNamespace(zero_grad=lambda **_kwargs: None, step=lambda: events.append(("step",)))
    micro_batches = [{"advantages": torch.ones(1), "step_indices": [index]} for index in range(2)]
    shell._load_train_batches = lambda *_args: micro_batches
    shell._assert_microbatch_count_aligned = lambda *_args: None
    shell._prepare_micro_batch = lambda mb: (mb, mb["advantages"])
    shell._record_update_metrics = lambda *_args: None
    shell._apply_lr = lambda: None
    shell._log_update = lambda *_args: None
    shell._sync_save_and_release = lambda *_args: None
    shell._run_step_evaluation = lambda *_args: None
    shell._log_train_metrics = lambda *_args: None

    def freeze_anchor(_batch: dict, micro_batch: dict, *_args: object) -> torch.Tensor:
        index = micro_batch["step_indices"][0]
        events.append(("freeze", index))
        return torch.tensor(index)

    def update(
        _model: torch.nn.Module,
        _adapter: object,
        _batch: dict,
        *,
        step_indices: list[int],
        clip_eps: float,
        old_logp: torch.Tensor | None,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        anchor = old_logp.item() if old_logp is not None else None
        events.append(("update", step_indices[0], clip_eps, anchor))
        return {"new_logp": torch.zeros(1)}

    shell._freeze_anchor = freeze_anchor
    monkeypatch.setattr(actor_module, "flow_grpo_update", update)

    try:
        shell.train(rollout_id=3)
    finally:
        for name in ("train", "actor_train", "train_wait"):
            if name in Timer().start_time:
                Timer().end(name)
            Timer().reset(name)

    if configured:
        assert events == [
            ("freeze", 0),
            ("freeze", 1),
            ("update", 0, 0.0, 0),
            ("step",),
            ("update", 1, 0.0, 1),
            ("step",),
        ]
        assert shell._lr_steps == 2
    else:
        assert events == [("update", 0, 0.2, None), ("update", 1, 0.2, None), ("step",)]
        assert shell._lr_steps == 1


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


def _eval_shell(**over: object) -> FSDPTrainRayActor:
    args = parser_for_flags("--num-rollout").parse_args(["--num-rollout", "100"])
    vars(args).update(
        eval_interval=10,
        final_eval_poll_interval=0.0,
        final_eval_wait_timeout=1.0,
    )
    vars(args).update(over)
    shell = object.__new__(FSDPTrainRayActor)
    shell._rank = 0
    shell.rollout_manager = object()
    shell.args = args
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


def _valid_args(**over: object) -> Namespace:
    # The generative provider also registers the FSDP training group.
    parser = add_generative_arguments(ArgumentParser())
    initialized = parser_for_flags(
        "--actor-num-gpus-per-node",
        "--actor-num-nodes",
        "--advantage-estimator",
        "--autoscaler-config",
        "--fully-async",
        "--hybrid",
        "--kl-coef",
        "--lora-adapter-mode",
        "--lora-dropout",
        "--lora-rank",
        "--loss-type",
        "--max-tokens-per-gpu",
        "--save-debug-train-data",
        "--save-hf",
        "--use-dynamic-batch-size",
        "--use-kl-loss",
    ).parse_args([])
    args = parser.parse_args(
        [
            "--generation-task",
            "t2i",
            "--model-adapter-path",
            "relax.models.qwen_image.adapter.QwenImageAdapter",
            "--model-path",
            "/models/Qwen-Image",
            # Both rollout and actor replay require eta. Exclude step 0,
            # where the sde diffusion coefficient is singular.
            "--sampling-config",
            '{"num_inference_steps": 12, "eta": 0.7, "sde_indices": [1, 3, 5]}',
        ],
        namespace=initialized,
    )
    vars(args).update(
        train_backend="fsdp",
        rollout_num_gpus_per_engine=1,
        multimodal_keys={},
        # Synchronous colocate is the only supported generative deployment mode.
        colocate=True,
        async_save=False,
        load_debug_rollout_data=None,
        use_critic=False,
    )
    # Some validator cases intentionally bypass argparse choices.
    vars(args).update(over)
    return args


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


def test_validate_aggregates_invalid_generative_options() -> None:
    args = _valid_args(
        generation_task=None,
        model_adapter_path=None,
        model_path=None,
        weight_sync_wire_dtype="fp32",
        reward_required_components=["quality"],
        reward_component_weights={"base": 1.0},
    )
    with pytest.raises(ValueError) as exc_info:
        validate_generative_config(args)

    message = str(exc_info.value)
    assert "generation_task must be one of" in message
    assert "model_adapter_path is required" in message
    assert "model_path is required" in message
    assert "weight_sync_wire_dtype must be bf16" in message
    assert "missing required components: ['quality']" in message


def test_validate_accepts_tp_dividing_actor_world():
    # Rollout TP>1 is supported when the engine GPUs tile the actor world.
    validate_generative_config(
        _valid_args(rollout_num_gpus_per_engine=2, actor_num_gpus_per_node=8, actor_num_nodes=1)
    )


def test_validate_accepts_tp_when_actor_world_unknown():
    # Without actor world info the divisibility check is deferred to the first sync.
    validate_generative_config(_valid_args(rollout_num_gpus_per_engine=4, actor_num_gpus_per_node=None))


def test_validate_rejects_tp_not_dividing_actor_world():
    with pytest.raises(ValueError, match="divisible by rollout_num_gpus_per_engine"):
        validate_generative_config(
            _valid_args(rollout_num_gpus_per_engine=3, actor_num_gpus_per_node=8, actor_num_nodes=1)
        )


def test_validate_rejects_tp_below_one():
    with pytest.raises(ValueError, match="must be >= 1"):
        validate_generative_config(_valid_args(rollout_num_gpus_per_engine=0))


def test_validate_skips_non_fsdp_backend() -> None:
    # Megatron path must be untouched by the generative fail-fast checks.
    validate_generative_config(Namespace())
    validate_generative_config(Namespace(train_backend="megatron", rollout_num_gpus_per_engine=8))


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

    args = add_fsdp_arguments(ArgumentParser()).parse_args([])
    assert lr_at_step(args, 0, 1e-5) == 1e-5
    assert lr_at_step(args, 500, 1e-5) == 1e-5


def test_lr_at_step_keeps_base_lr_for_explicit_none():
    from relax.backends.fsdp.actor import lr_at_step

    args = add_fsdp_arguments(ArgumentParser()).parse_args([])
    args.fsdp_lr_scheduler = None
    args.lr_decay_iters = 10
    args.min_lr = 1e-6
    assert lr_at_step(args, 5, 1e-5) == 1e-5


def test_lr_at_step_warmup_then_cosine_decays_to_min():
    from relax.backends.fsdp.actor import lr_at_step

    args = add_fsdp_arguments(ArgumentParser()).parse_args(["--fsdp-lr-scheduler", "cosine"])
    args.lr_warmup_iters = 10
    args.lr_decay_iters = 110
    args.min_lr = 1e-6
    assert lr_at_step(args, 0, 1e-4) == pytest.approx(1e-5)  # first warmup step
    assert lr_at_step(args, 9, 1e-4) == pytest.approx(1e-4)  # warmup complete
    assert lr_at_step(args, 110, 1e-4) == pytest.approx(1e-6)  # decayed to the floor
    mid = lr_at_step(args, 60, 1e-4)
    assert 1e-6 < mid < 1e-4


# ---------------------------------------------------------------------------
# LoRA boundary
# ---------------------------------------------------------------------------


def _lora_args(**over):
    parsed = add_fsdp_arguments(ArgumentParser()).parse_args(["--fsdp-trainable-mode", "lora"])
    base = dict(
        fsdp_trainable_mode=parsed.fsdp_trainable_mode,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=parser_for_flags("--lora-dropout").parse_args([]).lora_dropout,
        lora_merge_mode=True,
        lora_adapter_mode=False,
        fsdp_master_dtype="fp32",
        fsdp_cpu_offload=parsed.fsdp_cpu_offload,
    )
    base.update(over)
    return _valid_args(**base)


def _actor_config_args(*options: str) -> Namespace:
    lora_parser = parser_for_flags("--lora-rank", "--lora-alpha", "--lora-target-modules", "--lora-dropout")
    parser = add_generative_arguments(ArgumentParser(parents=[lora_parser], add_help=False))
    return parser.parse_args(
        [
            "--generation-task",
            "t2i",
            "--model-path",
            "base-model",
            "--fsdp-trainable-mode",
            "lora",
            "--lora-rank",
            "4",
            "--lora-alpha",
            "8",
            *options,
        ]
    )


@pytest.mark.parametrize("configured", ["adapter-default", "explicit-cli", "empty-yaml"])
def test_lora_actor_checkpoint_contract_uses_initialized_configuration(configured: str, tmp_path: Path) -> None:
    cli = []
    expected_revision = None
    expected_targets = ["adapter-target"]
    adapter = SimpleNamespace(family="test", supported_tasks=("t2i",), lora_target_modules=expected_targets)
    if configured == "explicit-cli":
        cli = ["--lora-target-modules", "cli-target", "--model-revision", "release", "--lora-dropout", "0"]
        expected_revision = "release"
        expected_targets = ["cli-target"]
    args = _actor_config_args(*cli)
    if configured == "empty-yaml":
        args.lora_target_modules = []
        args.lora_dropout = None

    actor = object.__new__(FSDPTrainRayActor)
    actor.args = args
    actor.adapter = adapter
    actor._trainable_mode = args.fsdp_trainable_mode
    actor._lora_target_modules = actor._resolve_lora_target_modules(args)
    actor._base_model_sha256 = actor._hash_base_model(args)

    contract = actor._adapter_contract()

    assert contract["base_model_sha256"] == hashlib.sha256(f"base-model:{expected_revision}".encode()).hexdigest()
    assert contract["lora"]["target_modules"] == expected_targets
    assert contract["lora"]["dropout"] == 0.0
    assert contract["lora"]["scaling"] == 2.0
    (tmp_path / "adapter_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    actor._assert_resumable(str(tmp_path))


def test_lora_actor_requires_targets_if_adapter_does_not_provide_defaults() -> None:
    actor = object.__new__(FSDPTrainRayActor)
    actor.adapter = SimpleNamespace()

    with pytest.raises(ValueError, match="needs target modules"):
        actor._resolve_lora_target_modules(_actor_config_args())


@pytest.mark.parametrize("configured", [False, True], ids=["declared-empty", "explicit-sde"])
def test_fsdp_actor_fingerprint_uses_declared_sampling_config(configured: bool) -> None:
    sampling = {"eta": 0.7, "sde_indices": [1, 3]} if configured else {}
    cli = ["--sampling-config", json.dumps(sampling)] if configured else []
    actor = object.__new__(FSDPTrainRayActor)
    actor.args = _actor_config_args(*cli)

    payload = json.dumps({"sampling": sampling, "task": "t2i"}, sort_keys=True)

    assert actor.sampling_fingerprint() == hashlib.sha256(payload.encode()).hexdigest()


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
        parsed = add_fsdp_arguments(ArgumentParser()).parse_args(["--fsdp-cpu-offload"])
        validate_generative_config(_lora_args(fsdp_cpu_offload=parsed.fsdp_cpu_offload))


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
