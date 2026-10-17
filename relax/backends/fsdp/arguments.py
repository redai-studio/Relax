# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FSDP2 / generative-RL argument group, profile validator and full-FT
boundary.

`add_generative_arguments` is the single entry point wired into the main Relax
parser (`relax/utils/arguments.py`). It registers the engine / adapter / task /
artifact / reward / sampling / weight-sync flags plus the FSDP2 training group.
Every flag defaults to an inert value so the default Megatron token-RL path is
untouched (design doc 8.3, 13).

`validate_generative_config` implements the fail-fast checks of design doc 13.1.
It is intentionally import-light (no torch / model loads) so it can run at
launch preflight and in unit tests.
"""

from __future__ import annotations

import json
import math
from argparse import ArgumentParser, Namespace
from typing import Any, List

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


__all__ = ["add_generative_arguments", "add_fsdp_arguments", "validate_generative_config"]

# Task identifiers the generative pipeline understands. Only `t2i` ships on
# this branch (`relax/models/qwen_image/`,
# `scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh`). Image-edit and
# video / audio-video, plus their Wan / LTX adapters, live on the
# `backup/diffusion-generative-rl-full` branch. A caller bringing their own
# adapter via `--model-adapter-path` must re-add its task here too, so the
# vocabulary can never advertise a task with no working replay path -- `i2i`
# used to be accepted here while `replay_transition` silently ignored the
# source image, i.e. it trained an unconditioned velocity field.
GENERATION_TASKS = ("t2i",)

# Where the reward scorer runs. Declared once so the argparse `choices` and the
# fail-fast validator cannot drift.
REWARD_RUNTIMES = ("colocate", "cpu", "remote")

# How generative reward components are scaled after per-prompt centering.
ADVANTAGE_STD_MODES = ("group", "batch", "none")


def add_fsdp_arguments(parser: ArgumentParser) -> ArgumentParser:
    """FSDP2 full-fine-tune training knobs (design doc 10.1)."""
    group = parser.add_argument_group("fsdp")
    group.add_argument("--fsdp-trainable-mode", type=str, default="full", choices=["full", "lora"])
    group.add_argument(
        "--fsdp-trainable-attr",
        type=str,
        default="transformer",
        help="Attribute on the model bundle that holds the trainable transformer.",
    )
    group.add_argument("--fsdp-param-dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    group.add_argument("--fsdp-reduce-dtype", type=str, default="fp32", choices=["bf16", "fp32"])
    group.add_argument(
        "--fsdp-master-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16"],
        help=(
            "Storage dtype for the TRAINABLE parameters, i.e. the optimizer's master copy. "
            "Unset (the full-FT default) keeps the loaded dtype. LoRA runs should set fp32: "
            "with bf16 masters the ~1e-4 relative RL updates round off the mantissa and the "
            "policy stops moving while grad norm and loss still look healthy. The forward is "
            "unaffected -- --fsdp-param-dtype still governs the all-gathered compute copy."
        ),
    )
    # Resharding after forward is the memory-saving default; only the opt-out is
    # a real knob (a positive --fsdp-reshard-after-forward would be a no-op).
    group.add_argument(
        "--no-fsdp-reshard-after-forward", dest="fsdp_reshard_after_forward", action="store_false", default=True
    )
    group.add_argument("--fsdp-activation-checkpointing", action="store_true", default=False)
    group.add_argument("--fsdp-cpu-offload", action="store_true", default=False)
    group.add_argument(
        "--fsdp-debug-fingerprint",
        action="store_true",
        default=False,
        help=(
            "Log a full-model abs-sum fingerprint of the trainable weights before/after each step "
            "to detect no-op optimizer steps. Off by default: it is a full-parameter reduction + host "
            "sync every step (a GPU-CPU sync in the hot path)."
        ),
    )
    group.add_argument(
        "--fsdp-load-wave-size",
        type=int,
        default=0,
        help=(
            "Gate base-model loading to this many ranks at a time (barrier between waves) to bound peak "
            "host RAM and model-dir read contention at launch. 0 disables gating (all ranks load at once)."
        ),
    )
    group.add_argument(
        "--fsdp-lr-scheduler",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help=(
            "Post-warmup learning-rate shape for the generative backend. Warmup length comes from "
            "--lr-warmup-iters, the decay horizon from --lr-decay-iters and the floor from --min-lr; "
            "all are counted in OPTIMIZER STEPS (the actor takes several per rollout). Defaults to "
            "'constant', which preserves the validated full-FT behaviour."
        ),
    )
    group.add_argument(
        "--num-updates-per-batch",
        type=int,
        default=1,
        help=(
            "FlowGRPO PPO optimizer updates per rollout batch (design doc 6.4). The π_old anchor is frozen once, "
            "then the local train shard is split into this many disjoint equal-size updates. "
            "The validated diffusion recipes use 2. "
            "With 1 the ratio is always 1 so the clipped loss is 0 by construction (group-centered advantages) "
            "and clipping never engages; >1 makes later disjoint updates run after earlier optimizer steps, "
            "so the frozen anchor diverges → nonzero, meaningful loss + real PPO clip."
        ),
    )
    return parser


def add_generative_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Engine / adapter / task / reward / sampling / weight-sync flags."""
    group = parser.add_argument_group("generative")

    # Task & model
    group.add_argument("--generation-task", type=str, default=None, choices=[None, *GENERATION_TASKS, "i2i"])
    group.add_argument(
        "--i2i-sft-config",
        type=json.loads,
        default=None,
        help="Image SFT JSON: data_root, dataset_size, max_pixels, conditioning_images, attention settings.",
    )
    group.add_argument("--model-path", type=str, default=None, help="HF-format model directory for the adapter.")
    group.add_argument("--model-revision", type=str, default=None)
    group.add_argument(
        "--model-adapter-path", type=str, default=None, help="Dotpath to a GenerativeModelAdapter implementation."
    )

    # Rollout engine / driver hooks (resolved in rollout.py / utils.py)
    group.add_argument(
        "--rollout-engine-class-path",
        type=str,
        default=None,
        help="Dotpath to a custom rollout engine class; empty ⇒ default text SGLangEngine.",
    )
    group.add_argument(
        "--artifact-root", type=str, default=None, help="Root directory for media / trajectory sidecars."
    )
    group.add_argument(
        "--artifact-retention-rollouts",
        type=int,
        default=2,
        help=(
            "How many rollouts of trajectory sidecars + generated media to keep under --artifact-root. "
            "Older rollouts are reaped at the start of each rollout. <=0 keeps everything, which grows "
            "without bound (one safetensors sidecar per group plus one image per candidate, every step)."
        ),
    )

    # Sampling geometry (per-task; carried as a JSON dict, see design doc 13).
    group.add_argument("--sampling-config", type=json.loads, default=None, help="JSON sampling geometry dict.")
    group.add_argument("--generation-seed", type=int, default=1234)

    # Reward manager (design doc 7.3 / 11).
    group.add_argument(
        "--reward-runtime",
        type=str,
        default="cpu",
        choices=list(REWARD_RUNTIMES),
        help=(
            "Where the reward scorer runs. 'cpu' (default, the validated recipe) and 'colocate' both score "
            "IN THE ROLLOUT WORKER'S PROCESS -- they differ only in whether the scorer may use CUDA; "
            "'remote' calls --reward-endpoint over HTTP. There is no separate scorer actor pool: the old "
            "'dedicated' value was a synonym for 'colocate' that additionally logged a warning about a Ray "
            "group it never got."
        ),
    )
    group.add_argument("--reward-scorer-path", type=str, default=None, help="Dotpath to a GenerativeRewardScorer.")
    group.add_argument("--reward-model-path", type=str, default=None)
    group.add_argument(
        "--reward-required-components", type=json.loads, default=None, help="JSON list of component names."
    )
    group.add_argument(
        "--reward-component-weights", type=json.loads, default=None, help="JSON {component: weight} mapping."
    )
    group.add_argument("--reward-endpoint", type=str, default=None, help="HTTP endpoint for a remote scorer.")
    group.add_argument(
        "--generative-advantage-std-mode",
        type=str,
        default=None,
        choices=list(ADVANTAGE_STD_MODES),
        help=(
            "Std divisor after per-prompt reward centering for native generative RL. "
            "'group' matches Relax/Text GRPO, 'batch' matches diffusion reference recipes with global std, "
            "and 'none' is Dr.GRPO. Unset preserves the shared --disable-grpo-std-normalization behaviour."
        ),
    )

    # Full-weight sync transaction (design doc 10).
    group.add_argument(
        "--weight-sync-mode",
        type=str,
        default="full",
        choices=["full", "adapter"],
        help=(
            "Weight-sync transport. 'full' streams the whole trainable transformer every step "
            "(full FT, and LoRA merge-mode, which folds B@A into the base first). 'adapter' "
            "streams only the LoRA tensors and is derived automatically from --lora-adapter-mode."
        ),
    )
    group.add_argument("--weight-sync-wire-dtype", type=str, default="bf16", choices=["bf16"])
    group.add_argument("--weight-sync-bucket-size-mb", type=int, default=512)

    parser = add_fsdp_arguments(parser)
    return parser


def _fail(errors: List[str], cond: bool, msg: str) -> None:
    if cond:
        errors.append(msg)


def _parse_sampling_int(value: Any, field: str, errors: List[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"sampling_config {field} must be an integer, got {value!r}.")
        return None
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"sampling_config {field} must be an integer, got {value!r}.")
        return None
    try:
        numeric = float(value)
        exact = math.isfinite(numeric) and numeric == parsed
    except (OverflowError, TypeError, ValueError):
        exact = False
    if not exact:
        errors.append(f"sampling_config {field} must be an integer, got {value!r}.")
        return None
    return parsed


def _validate_sde_schedule(sampling: dict, errors: List[str]) -> None:
    """Reject a trained SDE step 0 under ``sde_type='sde'``.

    At step 0 the schedule has ``sigma == 1``, so the flow-SDE diffusion
    coefficient ``sqrt(sigma / (1 - sigma)) * eta`` is singular. The replay
    only stays finite there because of an arbitrary ``sigma_max=0.99`` clamp,
    which is NOT what the sampler used: measured against the SGLang diffusion
    engine, ``std_dev_t`` came out 7.0 vs ~2.92, which flips the sign of the
    ``sample`` coefficient in ``prev_sample_mean`` and inflates the variance
    2.4x. Every other step agrees to 1e-4. So a run that trains step 0 is
    computing its policy gradient against a distribution the sample was never
    drawn from -- silently, since the ratio==1 invariant hides it on the first
    mini-epoch.

    ``sde_type='dance'`` uses a constant coefficient (``std_dev_t = eta``) with
    no division by ``1 - sigma``, so step 0 is fine there.

    Checked against every route by which step 0 can become trainable: an
    explicit ``sde_indices``, the ``sde_pool`` a per-rollout resample draws
    from, and the derived ``sde_timestep_fraction`` window.
    """
    if str(sampling.get("sde_type", "sde")).lower() != "sde":
        return

    from relax.models.generative import resolve_sde_indices

    reachable = set()
    explicit = sampling.get("sde_indices")
    if explicit:
        reachable.update(int(i) for i in explicit)
    if sampling.get("sde_resample_per_rollout"):
        pool = sampling.get("sde_pool")
        # Without an explicit pool the draw ranges over the whole fraction
        # window, so resolve it the same way the runtime will.
        reachable.update(int(i) for i in pool) if pool else reachable.update(
            resolve_sde_indices({**sampling, "sde_resample_per_rollout": False, "sde_indices": None})
        )
    if not explicit and not sampling.get("sde_resample_per_rollout"):
        reachable.update(resolve_sde_indices(sampling))

    _fail(
        errors,
        0 in reachable,
        "sampling_config trains SDE step 0 under sde_type='sde', where sigma==1 makes the diffusion "
        "coefficient singular and the replayed Gaussian disagree with the sampler's (measured: "
        "std_dev_t 7.0 vs 2.92, sign flip on the prev_sample_mean 'sample' term). Exclude it -- e.g. "
        "sde_indices [1,3,5] with sde_pool [1,2,3,4,5], or sde_timestep_fraction starting above "
        "1/num_inference_steps. Use sde_type='dance' if you need step 0.",
    )


def validate_generative_config(args: Namespace) -> None:
    """Fail-fast validation for a generative RL run (design doc 13.1).

    Only runs when ``train_backend == 'fsdp'``; raises ``ValueError``
    collecting all violations so the operator sees every problem at once. GPU-
    topology and media-shape checks that need live objects are enforced later
    by the runtime.
    """
    if getattr(args, "train_backend", "megatron") != "fsdp":
        return

    if getattr(args, "loss_type", None) == "sft":
        validate_i2i_sft_config(args)
        return

    errors: List[str] = []
    warnings: List[str] = []

    # Keep this check with the rest of generative validation so operators get a
    # single aggregated error report instead of one argparse enum failure at a time.
    reward_runtime = getattr(args, "reward_runtime", "cpu")
    _fail(
        errors,
        reward_runtime not in REWARD_RUNTIMES,
        f"reward_runtime must be one of {REWARD_RUNTIMES}, got {reward_runtime!r}.",
    )
    advantage_std_mode = getattr(args, "generative_advantage_std_mode", None)
    _fail(
        errors,
        advantage_std_mode is not None and advantage_std_mode not in ADVANTAGE_STD_MODES,
        f"generative_advantage_std_mode must be one of {ADVANTAGE_STD_MODES}, got {advantage_std_mode!r}.",
    )

    # The transport follows the LoRA rollout path. It is DERIVED, not configured:
    # an explicit --weight-sync-mode that contradicts --lora-adapter-mode is an
    # error rather than something this function silently overwrites.
    adapter_mode = bool(getattr(args, "lora_adapter_mode", False))
    configured = getattr(args, "weight_sync_mode", "full")
    if adapter_mode and configured != "adapter":
        _fail(
            errors,
            configured != "full",  # "full" is the parser default, i.e. "unset"
            f"--lora-adapter-mode implies weight_sync_mode='adapter', but {configured!r} was requested.",
        )
        args.weight_sync_mode = "adapter"
    elif not adapter_mode and configured == "adapter":
        errors.append("weight_sync_mode='adapter' requires --lora-adapter-mode.")

    task = getattr(args, "generation_task", None)
    _fail(errors, task not in GENERATION_TASKS, f"generation_task must be one of {GENERATION_TASKS}, got {task!r}.")
    _fail(
        errors, not getattr(args, "model_adapter_path", None), "model_adapter_path is required for train_backend=fsdp."
    )
    _fail(errors, not getattr(args, "model_path", None), "model_path is required for train_backend=fsdp.")

    # Trainable-mode boundary. `--lora-rank > 0` is the single switch (shared with
    # the Megatron LoRA path); --fsdp-trainable-mode must agree with it rather than
    # be derived from it, so a half-configured run fails at preflight instead of
    # silently training the wrong parameter set.
    mode = getattr(args, "fsdp_trainable_mode", "full")
    lora_rank = int(getattr(args, "lora_rank", 0) or 0)
    _fail(errors, mode not in ("full", "lora"), f"fsdp_trainable_mode must be 'full' or 'lora', got {mode!r}.")
    _fail(
        errors,
        mode == "lora" and lora_rank <= 0,
        "fsdp_trainable_mode='lora' requires --lora-rank > 0.",
    )
    _fail(
        errors,
        mode == "full" and lora_rank > 0,
        "--lora-rank > 0 requires --fsdp-trainable-mode lora; otherwise the flag is silently ignored.",
    )
    if mode == "lora":
        # The pi_old anchor is produced by REPLAYING the trajectory through the live
        # model under model.train() (actor.py _freeze_anchor / train). Under full FT
        # that replay is deterministic, which is what makes ratio == 1 on the first
        # mini-epoch. Dropout would draw different masks for the anchor and the
        # update, so the ratio drifts off 1 on an on-policy step and -- with the tiny
        # --eps-clip these runs use -- essentially everything clips. No exception, no
        # log line, just a run that trains and goes nowhere.
        _fail(
            errors,
            float(getattr(args, "lora_dropout", 0.0) or 0.0) != 0.0,
            "lora_dropout must be 0.0 for train_backend=fsdp: the FlowGRPO pi_old anchor is a "
            "replay through the live model, so a stochastic forward breaks the ratio==1 invariant "
            "of the first mini-epoch and the clipped loss silently collapses.",
        )
        # CPU-offloaded shards have no CPU process-group backend. The weight-sync
        # gather routes around it via gather_device, but clip_grad_norm_ over
        # offloaded DTensor grads does not.
        _fail(
            errors,
            bool(getattr(args, "fsdp_cpu_offload", False)),
            "--fsdp-cpu-offload is not supported with fsdp_trainable_mode='lora' "
            "(gradient clipping over CPU-offloaded adapter DTensors has no collective backend).",
        )
        # bf16 masters round the ~1e-4 relative RL update off the mantissa. The run
        # still looks healthy (finite grad norm, moving loss) while the policy stops
        # moving -- indistinguishable from an lr that is too low, so say it loudly
        # rather than fail: an operator may deliberately want the memory back.
        if getattr(args, "fsdp_master_dtype", None) != "fp32":
            warnings.append(
                "fsdp_trainable_mode='lora' without --fsdp-master-dtype fp32: the optimizer's master "
                "copy stays bf16, which rounds away the ~1e-4 updates this algorithm produces. Expect "
                "a flat reward curve that looks like a too-low learning rate."
            )
    else:
        _fail(
            errors,
            getattr(args, "weight_sync_mode", "full") != "full",
            "weight_sync_mode must be 'full' unless fsdp_trainable_mode='lora'.",
        )

    sampling = getattr(args, "sampling_config", None) or {}
    if not isinstance(sampling, dict):
        errors.append(f"sampling_config must be a mapping, got {type(sampling).__name__}.")
        sampling = {}
    # `eta` is the sampler's noise level. The rollout request and the actor's
    # replay must use the SAME value or the PPO ratio is anchored to a Gaussian
    # the sample was never drawn from -- silently, because the ratio==1
    # invariant of the first mini-epoch hides it. Neither side carries a
    # default any more (they used to disagree, 0.7 vs 1.0), so require it here.
    if "eta" not in sampling:
        errors.append(
            "sampling_config must set 'eta' (the SDE noise level). It is read by BOTH the rollout request "
            "and the actor replay; leaving it implicit let the two disagree silently."
        )
    else:
        try:
            eta = float(sampling["eta"])
        except (TypeError, ValueError):
            eta = float("nan")
        _fail(
            errors,
            not math.isfinite(eta) or eta <= 0.0,
            f"sampling_config eta must be finite and > 0, got {sampling['eta']!r}.",
        )

    num_steps = _parse_sampling_int(sampling.get("num_inference_steps"), "num_inference_steps", errors)
    if num_steps is not None:
        _fail(errors, num_steps <= 0, f"sampling_config num_inference_steps must be > 0, got {num_steps}.")

    sde_type = str(sampling.get("sde_type", "sde")).lower()
    _fail(
        errors,
        sde_type not in ("sde", "dance"),
        f"sampling_config sde_type must be 'sde' or 'dance', got {sde_type!r}.",
    )

    indices_valid = num_steps is not None and num_steps > 0
    fraction = sampling.get("sde_timestep_fraction")
    if fraction is not None:
        try:
            fraction_values = [float(value) for value in fraction]
        except (TypeError, ValueError):
            fraction_values = []
        if (
            len(fraction_values) != 2
            or not all(math.isfinite(value) for value in fraction_values)
            or not 0.0 <= fraction_values[0] < fraction_values[1] <= 1.0
        ):
            errors.append(
                "sampling_config sde_timestep_fraction must be two finite values satisfying 0 <= low < high <= 1, "
                f"got {fraction!r}."
            )
            indices_valid = False
    for field in ("sde_indices", "sde_pool"):
        values = sampling.get(field)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            errors.append(f"sampling_config {field} must be a list of integers, got {values!r}.")
            indices_valid = False
            continue
        for value in values:
            index = _parse_sampling_int(value, f"{field} entry", errors)
            if index is None:
                indices_valid = False
            elif num_steps is not None and num_steps > 0 and not 0 <= index < num_steps:
                errors.append(
                    f"sampling_config {field} index {index} is outside [0, num_inference_steps={num_steps})."
                )
                indices_valid = False
    if indices_valid and sde_type in ("sde", "dance"):
        from relax.models.generative import resolve_sde_indices

        assert num_steps is not None
        resolved = resolve_sde_indices(sampling)
        for index in resolved:
            _fail(
                errors,
                not 0 <= index < num_steps,
                f"sampling_config resolved SDE index {index} is outside [0, num_inference_steps={num_steps}).",
            )
        _validate_sde_schedule(sampling, errors)

    use_critic = bool(getattr(args, "use_critic", False))
    estimator = getattr(args, "advantage_estimator", "grpo")
    _fail(
        errors,
        use_critic or estimator == "ppo",
        "train_backend=fsdp generative RL supports only the critic-free FlowGRPO path; "
        "PPO/use_critic requires a critic/value-model actor that is not implemented for FSDP.",
    )
    _fail(
        errors,
        getattr(args, "weight_sync_wire_dtype", "bf16") != "bf16",
        "weight_sync_wire_dtype must be bf16 for full sync.",
    )

    # Deployment mode: only synchronous colocate is implemented for the generative
    # FSDP backend. The fully-async / hybrid weight transport is stubbed
    # (FSDPTrainRayActor.update_weights_fully_async raises, and train_async is not
    # implemented), so reject those modes at preflight instead of crashing mid-run
    # with an opaque AttributeError / NotImplementedError.
    _fail(
        errors,
        bool(getattr(args, "fully_async", False)),
        "fully_async is not supported for train_backend=fsdp (generative RL is synchronous colocate only in this release).",
    )
    _fail(
        errors,
        bool(getattr(args, "hybrid", False)),
        "hybrid mode is not supported for train_backend=fsdp (synchronous colocate only in this release).",
    )
    _fail(
        errors,
        not bool(getattr(args, "colocate", False)),
        "train_backend=fsdp requires --colocate (synchronous colocate weight sync).",
    )
    # KL-to-reference is not implemented in the FlowGRPO update (no reference model,
    # no KL term). A nonzero kl_coef / use_kl_loss would be silently ignored, so
    # reject it rather than mislead the operator.
    _fail(
        errors,
        float(getattr(args, "kl_coef", 0.0) or 0.0) != 0.0 or bool(getattr(args, "use_kl_loss", False)),
        "kl_coef/use_kl_loss are not supported for train_backend=fsdp "
        "(FlowGRPO has no reference model / KL-to-ref term in this release).",
    )
    # Rollout TP (rollout_num_gpus_per_engine) is supported: the FSDP weight sync
    # groups the `tp` FSDP ranks co-located on an engine's physical GPUs into a
    # Gloo gather group and hands the engine `tp` CUDA-IPC blobs (one per worker).
    # The only hard requirement is that the engine GPUs tile the actor world, i.e.
    # actor_world % rollout_num_gpus_per_engine == 0 (the exact
    # world == sum(engine_gpu_counts) check runs at the first sync). Guard tp >= 1
    # here and, when the actor world size is already known, the divisibility.
    tp_raw = getattr(args, "rollout_num_gpus_per_engine", 1)
    tp = int(tp_raw) if tp_raw is not None else 1
    _fail(errors, tp < 1, "rollout_num_gpus_per_engine must be >= 1.")
    gpn = getattr(args, "actor_num_gpus_per_node", None)
    nodes = getattr(args, "actor_num_nodes", None)
    if tp >= 1 and gpn and nodes:
        actor_world = int(gpn) * int(nodes)
        _fail(
            errors,
            actor_world % tp != 0,
            f"actor world size ({actor_world}) must be divisible by rollout_num_gpus_per_engine ({tp}) "
            "for train_backend=fsdp (each engine's GPUs must tile the FSDP ranks for CUDA-IPC weight sync).",
        )

    # Condition presence per task.
    mm_keys = getattr(args, "multimodal_keys", None) or {}
    if task in ("i2i", "i2v"):
        _fail(errors, "image" not in mm_keys, f"task {task} requires an 'image' entry in multimodal_keys.")
    if task == "v2v":
        _fail(errors, "video" not in mm_keys, "task v2v requires a 'video' entry in multimodal_keys.")

    # Reward required components + weights consistency.
    required = getattr(args, "reward_required_components", None)
    weights = getattr(args, "reward_component_weights", None)
    if required is not None and weights is not None:
        missing = [c for c in required if c not in weights]
        _fail(errors, bool(missing), f"reward component_weights missing required components: {missing}.")

    # ---------------------------------------------------------------------
    # Flags the shared (Megatron) parser accepts but this backend does NOT
    # consume. Left unchecked they parse fine and silently do nothing, which
    # reads as "configured" on a dashboard while having zero effect. Reject the
    # ones that would change training/serving semantics; warn for the inert ones.
    # ---------------------------------------------------------------------
    _fail(
        errors,
        bool(getattr(args, "use_dynamic_batch_size", False)),
        "use_dynamic_batch_size is not supported for train_backend=fsdp "
        "(diffusion latents are fixed-shape; there is no token-budget batching).",
    )
    _fail(
        errors,
        getattr(args, "max_tokens_per_gpu", None) is not None,
        "max_tokens_per_gpu is not supported for train_backend=fsdp (no token-budget batching).",
    )
    _fail(
        errors,
        getattr(args, "autoscaler_config", None) is not None,
        "autoscaler_config is not supported for train_backend=fsdp "
        "(the diffusion engine does not implement the elastic router/scale-out API).",
    )
    _fail(
        errors,
        getattr(args, "save_hf", None) is not None,
        "save_hf (in-training HF export) is not supported for train_backend=fsdp; "
        "export offline with examples/diffusion/export_checkpoint.py.",
    )
    _fail(
        errors,
        getattr(args, "load_debug_rollout_data", None) is not None,
        "load_debug_rollout_data (offline rollout replay) is not implemented for train_backend=fsdp; "
        "the generative actor reads its batch from the TransferQueue + on-disk trajectory sidecars.",
    )

    if getattr(args, "async_save", False):
        # slime_validate_args REQUIRES --async-save alongside --rotate-ckpt, so this
        # cannot be a hard error; the FSDP DCP save is synchronous regardless.
        warnings.append("async_save has no effect for train_backend=fsdp (the DCP save is synchronous).")
    if getattr(args, "lr_decay_style", None) and getattr(args, "fsdp_lr_scheduler", "constant") == "constant":
        warnings.append(
            f"lr_decay_style={getattr(args, 'lr_decay_style')!r} is ignored; the generative backend uses "
            "--fsdp-lr-scheduler (currently 'constant'). Set --fsdp-lr-scheduler to enable decay."
        )
    if getattr(args, "save_debug_train_data", None):
        warnings.append(
            "save_debug_train_data is not emitted by the generative actor (the train rows are numeric "
            "indices + on-disk trajectory sidecars, not token sequences); rollout-side artifacts are "
            "written under --artifact-root instead."
        )
    for msg in warnings:
        logger.warning(f"[generative config] {msg}")

    if errors:
        raise ValueError("Generative RL config validation failed:\n  - " + "\n  - ".join(errors))


def validate_i2i_sft_config(args: Namespace) -> None:
    """Validate supervision independently of rollout/reward configuration."""
    errors: list[str] = []
    _fail(errors, getattr(args, "generation_task", None) != "i2i", "FSDP SFT currently supports generation_task=i2i.")
    for field in ("model_path", "prompt_data"):
        _fail(errors, not getattr(args, field, None), f"Image SFT requires --{field.replace('_', '-')}.")
    config = getattr(args, "i2i_sft_config", None)
    if not isinstance(config, dict):
        raise ValueError("Image SFT requires --i2i-sft-config with data_root and dataset_size")
    defaults = {
        "max_pixels": 1048576,
        "conditioning_images": 1,
        "attention_implementation": "flash_attention_2",
        "deterministic_attention": False,
        "prefix_gradient_checkpointing": True,
        "fixed_pixels_tiers": None,
        "seed": None,
    }
    unknown = (
        set(config)
        - set(defaults)
        - {
            "data_root",
            "dataset_size",
            "image_prefetch_depth",
            "image_prefetch_workers",
            "dp_tier_micro_balance",
            "dp_tier_rank_rotation",
        }
    )
    _fail(errors, bool(unknown), f"Unknown i2i_sft_config keys: {sorted(unknown)}")
    args.i2i_sft_config = config = {**defaults, **config}
    depth, workers = config.get("image_prefetch_depth", 0), config.get("image_prefetch_workers", 2)
    _fail(errors, type(depth) is not int or depth < 0, "image_prefetch_depth must be a nonnegative integer")
    _fail(errors, type(workers) is not int or workers <= 0, "image_prefetch_workers must be a positive integer")
    _fail(
        errors, type(config.get("dp_tier_micro_balance", False)) is not bool, "dp_tier_micro_balance must be boolean"
    )
    # Explicit false has the same checkpoint contract as historical absence.
    if config.get("dp_tier_micro_balance") is False:
        config.pop("dp_tier_micro_balance")
    _fail(
        errors, type(config.get("dp_tier_rank_rotation", False)) is not bool, "dp_tier_rank_rotation must be boolean"
    )
    _fail(
        errors,
        config.get("dp_tier_rank_rotation", False) and not config.get("dp_tier_micro_balance", False),
        "dp_tier_rank_rotation requires dp_tier_micro_balance",
    )
    if config.get("dp_tier_rank_rotation") is False:
        config.pop("dp_tier_rank_rotation")
    _fail(
        errors,
        not isinstance(config.get("data_root"), str) or not config.get("data_root"),
        "i2i_sft_config.data_root must be a non-empty path",
    )
    for field in ("dataset_size", "max_pixels", "conditioning_images"):
        value = config.get(field)
        _fail(errors, type(value) is not int or value <= 0, f"i2i_sft_config.{field} must be a positive integer")
    _fail(errors, config["conditioning_images"] not in (1, 2), "conditioning_images must be 1 or 2")
    _fail(
        errors,
        config["attention_implementation"] not in ("flash_attention_2", "torch"),
        "Unsupported image attention implementation",
    )
    _fail(errors, type(config["deterministic_attention"]) is not bool, "deterministic_attention must be boolean")
    _fail(
        errors,
        type(config["prefix_gradient_checkpointing"]) is not bool,
        "prefix_gradient_checkpointing must be boolean",
    )
    _fail(
        errors,
        config["deterministic_attention"] and config["attention_implementation"] != "flash_attention_2",
        "Deterministic image SFT currently requires flash_attention_2",
    )
    tiers = config.get("fixed_pixels_tiers")
    _fail(
        errors,
        tiers is not None
        and (not isinstance(tiers, list) or not tiers or not all(type(v) is int and v > 0 for v in tiers)),
        "i2i_sft_config.fixed_pixels_tiers must be a non-empty list of positive integers",
    )
    seed = config.get("seed")
    _fail(
        errors,
        config.get("dp_tier_micro_balance", False) and (not isinstance(tiers, list) or len(tiers) < 2),
        "dp_tier_micro_balance requires at least two fixed_pixels_tiers",
    )
    if isinstance(tiers, list) and len(tiers) > 1 and seed is None:
        seed = config["seed"] = getattr(args, "seed", 42)
    _fail(errors, seed is not None and type(seed) is not int, "i2i_sft_config.seed must be an integer when set")
    args.model_adapter_path = (
        getattr(args, "model_adapter_path", None) or "relax.models.sensenova.adapter.SenseNovaSFTAdapter"
    )
    # These controls have no implementation in the supervised image path.
    for field in (
        "fully_async",
        "hybrid",
        "offload_train",
        "fsdp_cpu_offload",
        "fsdp_activation_checkpointing",
        "use_dynamic_batch_size",
        "balance_data",
        "sft_async_prepack",
        "sft_chunked_logits",
        "sft_predict_interval",
        "eval_prompt_data",
        "eval_size",
        "custom_dataset_class_path",
        "use_routing_replay",
        "use_rollout_routing_replay",
        "use_opd",
        "enable_mtp_training",
        "debug_train_only",
        "debug_rollout_only",
        "lora_merge_mode",
        "lora_adapter_mode",
        "num_critic_only_steps",
        "sft_train_data_prefetch",
        "fsdp_debug_fingerprint",
    ):
        _fail(errors, bool(getattr(args, field, None)), f"Image SFT does not support --{field.replace('_', '-')}.")
    for field in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
    ):
        _fail(errors, getattr(args, field, 1) != 1, f"Image FSDP SFT requires {field}=1.")
    _fail(
        errors,
        getattr(args, "task_type", "causal_lm") != "causal_lm",
        "Image SFT does not support text classification",
    )
    _fail(errors, getattr(args, "micro_batch_size", 1) != 1, "Image SFT currently requires --micro-batch-size 1")
    _fail(errors, getattr(args, "n_samples_per_prompt", 1) != 1, "Image SFT requires --n-samples-per-prompt 1")
    _fail(errors, getattr(args, "num_updates_per_batch", 1) != 1, "Image SFT takes one optimizer update per batch")
    batch = getattr(args, "global_batch_size", 0) or 0
    _fail(
        errors,
        batch <= 0 or batch != getattr(args, "rollout_batch_size", None),
        "Image SFT requires equal positive global_batch_size and rollout_batch_size",
    )
    _fail(
        errors,
        getattr(args, "fsdp_lr_scheduler", "constant") != "constant",
        "Image SFT currently supports linear warmup followed by constant LR",
    )
    _fail(
        errors,
        getattr(args, "fsdp_param_dtype", "bf16") != "bf16" or getattr(args, "fsdp_reduce_dtype", "fp32") != "fp32",
        "Image SFT requires BF16 compute and FP32 reduction",
    )
    if getattr(args, "fsdp_master_dtype", None) is None:
        args.fsdp_master_dtype = "fp32"
    _fail(errors, args.fsdp_master_dtype != "fp32", "Image SFT requires FP32 optimizer masters")
    if getattr(args, "fsdp_trainable_mode", "full") == "lora":
        _fail(errors, (getattr(args, "lora_rank", 0) or 0) <= 0, "LoRA mode requires positive --lora-rank")
    elif getattr(args, "lora_rank", 0):
        errors.append("Full SFT requires --lora-rank 0")
    if errors:
        raise ValueError("Invalid image SFT configuration:\n- " + "\n- ".join(errors))
