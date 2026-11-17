# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SenseNova I2I sampling requests and trajectory validation.

The trainer owns group planning, policy choice and replay. These helpers build
one candidate request and validate the server's actual sampling record; they
never synthesize missing trajectories from requested values.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional


PROTOCOL = "sensenova_i2i_flow_v1"
PROTOCOL_VERSION = 4
ART_PROTOCOL_VERSION = 5


def build_sensenova_rollout_request(
    prompt: str,
    image_paths: List[str],
    *,
    seed: int,
    candidate_id: str,
    group_id: str,
    member_id: int,
    algorithm: str = "art",
    dynamics: Optional[str] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    image_guidance_scale: float = 1.0,
    cfg_norm: str = "none",
    cfg_interval: tuple = (0.0, 1.0),
    timestep_shift: float = 3.0,
    enable_timestep_shift: bool = True,
    t_eps: float = 0.02,
    noise_level: float = 0.8,
    sde_step_indices: Optional[List[int]] = None,
    return_step_indices: Optional[List[int]] = None,
    initial_seed: Optional[int] = None,
    denoise_seeds: Optional[List[int]] = None,
    sde_num_steps: Optional[int] = None,
    sde_seed: Optional[int] = None,
    sigmas: Optional[List[float]] = None,
    timesteps: Optional[List[float]] = None,
    storage_dtype: str = "bfloat16",
    debug: bool = False,
    think_mode: bool = False,
    think_token_ids: Optional[List[int]] = None,
    max_think_tokens: int = 1024,
) -> Dict[str, Any]:
    """Build one candidate; compatible candidates can be packed by the engine.

    Top-level group_id/member_id are Relax-local bookkeeping. Packing removes
    them before HTTP transport; only candidate identity and noise reach SGLang.

    For GRPO, the caller declares eligible stochastic steps. An optional
    ``sde_num_steps`` selects a shared group window using the explicit
    ``sde_seed``. Initial and transition noise remain candidate-local.
    """
    if algorithm not in ("grpo", "art"):
        raise ValueError("algorithm must be 'grpo' or 'art'")
    if not isinstance(prompt, str) or not isinstance(image_paths, list) or not image_paths:
        raise ValueError("I2I rollout requires a prompt and an ordered nonempty image_paths list")
    if any(not isinstance(path, str) or not path for path in image_paths):
        raise ValueError("image_paths entries must be nonempty paths or URLs")
    if (
        not candidate_id
        or not group_id
        or not isinstance(member_id, int)
        or isinstance(member_id, bool)
        or member_id < 0
    ):
        raise ValueError("candidate_id, group_id and a non-negative member_id are required")
    dynamics = dynamics or ("ode" if algorithm == "art" else "cps")
    if dynamics not in ("ode", "cps", "sde") or (algorithm == "grpo" and dynamics == "ode"):
        raise ValueError("GRPO requires CPS or Flow-SDE; ART requires ODE")
    if algorithm == "art":
        if (
            dynamics != "ode"
            or sde_step_indices not in (None, [])
            or sde_num_steps is not None
            or sde_seed is not None
        ):
            raise ValueError("ART requires deterministic ODE with no stochastic steps")
        if (
            type(num_inference_steps) is not int
            or num_inference_steps < 1
            or not isinstance(return_step_indices, list)
            or not return_step_indices
            or any(type(i) is not int or not 0 <= i < num_inference_steps for i in return_step_indices)
            or len(set(return_step_indices)) != len(return_step_indices)
        ):
            raise ValueError("ART requires explicit unique selected input steps in [0, num_inference_steps)")
        return_step_indices = sorted(return_step_indices)
    if storage_dtype not in ("bfloat16", "float16"):
        raise ValueError("storage_dtype must be bfloat16 or float16")
    if storage_dtype == "float16" and (algorithm != "art" or dynamics != "ode"):
        raise ValueError("FP16 storage requires ART ODE capture")
    if algorithm == "grpo" and not sde_step_indices:
        raise ValueError("GRPO requires explicit stochastic step indices")
    if sde_num_steps is not None and sde_seed is None:
        raise ValueError("Random step selection requires an explicit group-shared sde_seed")
    if sde_num_steps is not None:
        import torch

        eligible = list(range(num_inference_steps - 1)) if sde_step_indices is None else sorted(sde_step_indices)
        if (
            type(sde_num_steps) is not int
            or not 1 <= sde_num_steps <= len(eligible)
            or len(set(eligible)) != len(eligible)
            or any(type(i) is not int or not 0 <= i < num_inference_steps for i in eligible)
        ):
            raise ValueError("sde_num_steps must select a nonempty subset of valid, unique stochastic steps")
        if type(sde_seed) is not int or not 0 <= sde_seed < 2**63:
            raise ValueError("sde_seed must be an integer in [0, 2**63)")
        # Group planning belongs to the trainer. Send the chosen steps so the
        # inference engine only executes and records an explicit schedule.
        permutation = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(sde_seed))
        sde_step_indices = sorted(eligible[i] for i in permutation[:sde_num_steps].tolist())
    if type(max_think_tokens) is not int or not 1 <= max_think_tokens <= 1024:
        raise ValueError("max_think_tokens must be in [1, 1024]")
    if think_token_ids is not None and (
        not think_mode
        or not isinstance(think_token_ids, list)
        or len(think_token_ids) > max_think_tokens
        or any(type(token) is not int or token < 0 for token in think_token_ids)
    ):
        raise ValueError("Think replay requires think_mode and a bounded list of token IDs")
    options: Dict[str, Any] = {
        "capture": {"grpo": "transitions", "art": "ode_states"}[algorithm],
        "candidate_id": str(candidate_id),
    }
    if storage_dtype != "bfloat16":
        options["storage_dtype"] = storage_dtype
    for name, value in (
        ("initial_seed", initial_seed),
        ("denoise_seeds", denoise_seeds),
        ("sigmas", sigmas),
        ("timesteps", timesteps),
    ):
        if value is not None:
            options[name] = value
    return {
        "group_id": str(group_id),
        "member_id": member_id,
        "prompt": prompt,
        "image_path": list(image_paths),
        "seed": seed,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "num_outputs_per_prompt": 1,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": image_guidance_scale,
        "rollout": True,
        "rollout_sde_type": dynamics,
        "rollout_noise_level": 0.0 if dynamics == "ode" else noise_level,
        "rollout_log_prob_no_const": dynamics != "sde",
        "rollout_debug_mode": debug,
        "rollout_return_denoising_env": True,
        "rollout_return_dit_trajectory": True,
        "rollout_sde_step_indices": sde_step_indices,
        "rollout_return_step_indices": return_step_indices,
        "extra_sampling_params": {
            "rollout_options": options,
            "cfg_norm": cfg_norm,
            "cfg_interval": list(cfg_interval),
            "timestep_shift": timestep_shift,
            "enable_timestep_shift": enable_timestep_shift,
            "t_eps": t_eps,
            "think_mode": think_mode,
            "think_token_ids": think_token_ids,
            "max_think_tokens": max_think_tokens,
        },
    }


def is_sensenova_rollout_request(request: Mapping[str, Any]) -> bool:
    return isinstance((request.get("extra_sampling_params") or {}).get("rollout_options"), dict)


def validate_sensenova_rollout_response(response: Mapping[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an already-deserialized, per-candidate server result."""
    import torch

    options = (request.get("extra_sampling_params") or {}).get("rollout_options") or {}
    if (
        options.get("capture") == "terminal"
        and request.get("rollout_sde_type") == "ode"
        and request.get("rollout_return_dit_trajectory") is False
        and request.get("rollout_return_denoising_env") is False
        and not request.get("rollout_return_step_indices")
    ):
        if response.get("seed") != request["seed"]:
            raise ValueError("Evaluation response seed mismatch")
        return {
            "request_id": response.get("request_id"),
            "group_id": request.get("group_id"),
            "member_id": request.get("member_id"),
            "seed": response["seed"],
            "generated_output": _validate_reward_image(response, request),
            "height": request["height"],
            "width": request["width"],
            "task": "image-editing",
        }
    environment = response.get("denoising_env") or {}
    metadata = (environment.get("image_kwargs") or {}).get("sensenova_rollout")
    if not isinstance(metadata, dict) or metadata.get("protocol") != PROTOCOL:
        raise ValueError("Missing SenseNova rollout metadata")
    art_capture = options.get("capture") == "ode_states"
    if metadata.get("protocol_version") != (ART_PROTOCOL_VERSION if art_capture else PROTOCOL_VERSION):
        raise ValueError("SenseNova rollout protocol version mismatch")
    if metadata.get("state_layout") != "T,C,H,W" or response.get("seed") != request["seed"]:
        raise ValueError("Rollout state layout or request seed mismatch")
    if metadata.get("candidate_id") != options.get("candidate_id", str(request["seed"])):
        raise ValueError("Candidate identity mismatch")
    if metadata.get("dynamics") != request.get("rollout_sde_type"):
        raise ValueError("Returned rollout dynamics disagree with the request")
    if options.get("capture") is not None and metadata.get("capture") != options["capture"]:
        raise ValueError("Returned capture mode disagrees with the request")
    if metadata.get("initial_seed") != options.get("initial_seed", request["seed"]):
        raise ValueError("Actual initial noise seed mismatch")
    if metadata.get("noise_level") != request["rollout_noise_level"]:
        raise ValueError("Actual rollout noise level mismatch")
    denoise_seeds = metadata.get("denoise_seeds")
    if (
        not isinstance(denoise_seeds, list)
        or len(denoise_seeds) != request["num_inference_steps"]
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in denoise_seeds)
    ):
        raise ValueError("Missing actual per-step noise seeds")
    if "denoise_seeds" in options and options["denoise_seeds"] != denoise_seeds:
        raise ValueError("Actual per-step noise seeds mismatch")

    steps = request["num_inference_steps"]
    trajectory = response.get("dit_trajectory") or {}
    states, times, sigmas = (trajectory.get(key) for key in ("latents", "timesteps", "sigmas"))
    state_indices = metadata.get("state_indices")
    transition_indices = metadata.get("transition_indices")
    sde_indices = metadata.get("sde_indices")
    for name, values, limit in (
        ("state_indices", state_indices, steps + 1),
        ("transition_indices", transition_indices, steps),
        ("sde_indices", sde_indices, steps),
    ):
        if not isinstance(values, list) or any(type(i) is not int or not 0 <= i < limit for i in values):
            raise ValueError(f"Invalid actual {name}")
        if sorted(set(values)) != values:
            raise ValueError(f"Unsorted or duplicate actual {name}")
    if request["rollout_noise_level"] > 0:
        eligible = request.get("rollout_sde_step_indices")
        eligible = list(range(steps - 1)) if eligible is None else sorted(eligible)
        if not set(sde_indices) <= set(eligible) or len(sde_indices) != options.get("sde_num_steps", len(eligible)):
            raise ValueError("Actual stochastic steps disagree with the declared window")
    elif sde_indices:
        raise ValueError("Deterministic rollout cannot report stochastic steps")
    if art_capture:
        _validate_art_ode_cache(
            metadata,
            states,
            trajectory.get("old_velocities"),
            num_steps=steps,
            state_indices=state_indices,
            expected_indices=request.get("rollout_return_step_indices"),
        )
    elif metadata.get("capture") == "terminal":
        if state_indices != [steps] or transition_indices:
            raise ValueError("Terminal rollout must return exactly the unclipped final state")
    elif not transition_indices or not set(transition_indices) <= set(sde_indices):
        raise ValueError("Captured transitions must identify actual stochastic steps")
    if metadata.get("capture") == "transitions":
        requested_transitions = request.get("rollout_return_step_indices")
        if transition_indices != sorted(sde_indices if requested_transitions is None else requested_transitions):
            raise ValueError("Returned transition selection mismatch")
    required_states = {j for i in transition_indices for j in (i, i + 1)}
    if not required_states <= set(state_indices):
        raise ValueError("Sparse trajectory is missing a transition boundary")
    shape = (len(state_indices), 3, request["height"], request["width"])
    storage_dtype = options.get("storage_dtype", "bfloat16")
    if storage_dtype not in ("bfloat16", "float16"):
        raise ValueError("Unsupported requested SenseNova storage_dtype")
    if storage_dtype == "float16" and (
        metadata["dynamics"] != "ode" or metadata["capture"] not in ("terminal", "ode_states")
    ):
        raise ValueError("FP16 storage requires ODE terminal or selected-state capture")
    # Legacy BF16 responses may omit the field. FP16 must be explicitly
    # declared by both sides; never repair a mismatch with a decoder cast.
    if metadata.get("storage_dtype", "bfloat16") != storage_dtype:
        raise ValueError("SenseNova storage_dtype metadata disagrees with the request")
    if not isinstance(states, torch.Tensor) or states.shape != shape or states.dtype != getattr(torch, storage_dtype):
        raise ValueError(f"SenseNova trajectory must be {storage_dtype} with shape {shape}")
    if not torch.isfinite(states).all():
        raise ValueError("Non-finite SenseNova trajectory")
    if (
        not isinstance(times, torch.Tensor)
        or times.shape != (steps,)
        or not isinstance(sigmas, torch.Tensor)
        or sigmas.shape != (steps + 1,)
    ):
        raise ValueError("Missing actual rollout timesteps or sigmas")
    if (
        not torch.isfinite(sigmas).all()
        or sigmas[0] != 1
        or sigmas[-1] != 0
        or not torch.all(sigmas[:-1] > sigmas[1:])
    ):
        raise ValueError("Invalid actual rollout sigma schedule")
    if not torch.allclose(times.float(), sigmas[:-1].float() * 1000, atol=1e-4, rtol=1e-6):
        raise ValueError("Returned timestep and sigma coordinates disagree")
    if art_capture and (times.dtype != torch.float32 or sigmas.dtype != torch.float32):
        raise ValueError("ART requires original FP32 timesteps and sigmas")
    for key, actual in (("timesteps", times), ("sigmas", sigmas)):
        if key in options:
            expected = torch.as_tensor(options[key], dtype=actual.dtype, device=actual.device)
            if expected.shape != actual.shape or not torch.allclose(expected, actual, atol=1e-5, rtol=1e-6):
                raise ValueError(f"Driver {key} were changed by the server")
    scores = response.get("rollout_log_probs")
    expected_kind = (
        "none"
        if not transition_indices
        else ("cps_residual_mse" if metadata["dynamics"] == "cps" else "gaussian_mean_logprob")
    )
    if metadata.get("score_kind") != expected_kind:
        raise ValueError("Rollout score convention mismatch")
    if transition_indices:
        if (
            not isinstance(scores, torch.Tensor)
            or scores.shape != (len(transition_indices),)
            or not torch.isfinite(scores).all()
        ):
            raise ValueError("Missing or non-finite sampling-policy transition scores")
    elif scores is not None:
        raise ValueError("Deterministic rollout must not fabricate transition scores")
    if not art_capture and trajectory.get("old_velocities") is not None:
        raise ValueError("Old ODE velocities require explicit ART capture")
    conditioning = metadata.get("conditioning") or {}
    if type(metadata.get("actual_batch_size")) is not int or metadata["actual_batch_size"] < 1:
        raise ValueError("Missing actual model batch size")
    requested_think = request["extra_sampling_params"].get("think_mode", False)
    if conditioning.get("think_mode", False) != requested_think:
        raise ValueError("Actual Think mode mismatch")
    if requested_think:
        think = conditioning.get("think") or {}
        token_ids, append_ids = think.get("token_ids"), think.get("append_token_ids")
        max_tokens = request["extra_sampling_params"].get("max_think_tokens", 1024)
        if (
            not isinstance(token_ids, list)
            or len(token_ids) > max_tokens
            or not isinstance(append_ids, list)
            or not append_ids
            or any(type(token) is not int or token < 0 for token in token_ids + append_ids)
            or type(think.get("start_index")) is not int
            or type(think.get("end_index")) is not int
            or think["end_index"] - think["start_index"] != len(token_ids) + len(append_ids)
            or think.get("sampling") not in ("greedy", "fixed_tokens")
        ):
            raise ValueError("Incomplete Think replay condition")
        requested_ids = request["extra_sampling_params"].get("think_token_ids")
        if requested_ids is not None and (token_ids != requested_ids or think["sampling"] != "fixed_tokens"):
            raise ValueError("Server changed the fixed Think condition")
    if conditioning.get("prompt") != request["prompt"] or conditioning.get("image_paths") != request["image_path"]:
        raise ValueError("Rollout I2I conditions do not match the requested prompt and ordered references")
    if conditioning.get("height") != request["height"] or conditioning.get("width") != request["width"]:
        raise ValueError("Rollout geometry mismatch")
    for key, expected in (
        ("guidance_scale", request["guidance_scale"]),
        ("image_guidance_scale", request["true_cfg_scale"]),
        *(
            (key, request["extra_sampling_params"][key])
            for key in ("cfg_norm", "cfg_interval", "t_eps", "timestep_shift", "enable_timestep_shift")
        ),
    ):
        if conditioning.get(key) != expected:
            raise ValueError(f"Actual conditioning mismatch: {key}")
    generated = _validate_reward_image(response, request)
    return {
        "request_id": response.get("request_id"),
        "group_id": request.get("group_id"),
        "member_id": request.get("member_id"),
        "seed": response.get("seed"),
        "trajectory_latents": states,
        "timesteps": times,
        "sigmas": sigmas,
        "rollout_log_probs": scores,
        "denoising_env": environment,
        "rollout_metadata": metadata,
        "sde_indices": sde_indices,
        "state_indices": state_indices,
        "transition_indices": transition_indices,
        **(
            {"selected_indices": metadata["selected_indices"], "old_velocities": trajectory["old_velocities"]}
            if art_capture
            else {}
        ),
        "condition_fingerprint": _actual_fingerprint(
            {
                "conditioning": conditioning,
                "prefix": environment.get("pos_cond_kwargs"),
            }
        ),
        "scheduler_fingerprint": _actual_fingerprint(
            {
                "sigmas": sigmas,
                "timesteps": times,
                "sde_indices": sde_indices,
                "transition_indices": transition_indices,
                "dynamics": metadata["dynamics"],
                "noise_level": metadata["noise_level"],
                "score_kind": metadata.get("score_kind"),
                **(
                    {
                        key: metadata[key]
                        for key in (
                            "capture",
                            "selected_indices",
                            "velocity_convention",
                            "velocity_dtype",
                            "solver",
                            "protocol_version",
                            "storage_dtype",
                        )
                    }
                    if art_capture
                    else {}
                ),
            }
        ),
        "rollout_debug_tensors": response.get("rollout_debug_tensors"),
        "generated_output": generated,
        "height": conditioning["height"],
        "width": conditioning["width"],
        "task": "image-editing",
    }


def _validate_reward_image(response: Mapping[str, Any], request: Mapping[str, Any]) -> Any:
    import torch

    generated = response.get("generated_output")
    if isinstance(generated, torch.Tensor) and generated.dtype == torch.uint8:
        # SGLang's GPUWorker returns a single uint8 THWC frame per image,
        # whereas the reward adapter consumes normalized RGB CHW tensors.
        height, width = request["height"], request["width"]
        if generated.shape == (1, height, width, 3):
            generated = generated[0]
        if generated.shape == (height, width, 3):
            generated = generated.permute(2, 0, 1)
        if generated.shape == (3, height, width):
            generated = generated.float().div(255).contiguous()
    if (
        not isinstance(generated, torch.Tensor)
        or generated.shape != (3, request["height"], request["width"])
        or not torch.isfinite(generated).all()
    ):
        shape = tuple(generated.shape) if isinstance(generated, torch.Tensor) else None
        dtype = generated.dtype if isinstance(generated, torch.Tensor) else None
        raise ValueError(f"Invalid reward image: type={type(generated).__name__}, shape={shape}, dtype={dtype}")
    return generated


def _validate_art_ode_cache(
    metadata: Mapping[str, Any],
    states: Any,
    old_velocities: Any,
    *,
    num_steps: int,
    state_indices: List[int],
    expected_indices: Optional[List[int]] = None,
) -> List[int]:
    """Validate CPU-owned ART capture without repairing missing or rounded
    data."""
    import torch

    if any(
        metadata.get(key) != value
        for key, value in (
            ("protocol", PROTOCOL),
            ("protocol_version", ART_PROTOCOL_VERSION),
            ("capture", "ode_states"),
            ("dynamics", "ode"),
            ("noise_level", 0.0),
            ("solver", "euler"),
            ("velocity_convention", "noise_directed"),
            ("velocity_dtype", "float32"),
            ("score_kind", "none"),
            ("sde_indices", []),
            ("transition_indices", []),
        )
    ):
        raise ValueError("ART requires a v5 Euler ODE cache with noise-directed FP32 velocity and no scores")
    selected = metadata.get("selected_indices")
    if (
        not isinstance(selected, list)
        or not selected
        or any(type(index) is not int or not 0 <= index < num_steps for index in selected)
        or selected != sorted(set(selected))
        or (expected_indices is not None and selected != expected_indices)
    ):
        raise ValueError("ART selected input step indices are missing, invalid or changed")
    if state_indices != selected + [num_steps] or metadata.get("state_indices") != state_indices:
        raise ValueError("ART must return selected input states and the raw terminal state exactly once")
    if (
        not isinstance(states, torch.Tensor)
        or states.ndim != 4
        or states.shape[0] != len(selected) + 1
        or states.dtype not in (torch.float16, torch.bfloat16)
        or metadata.get("storage_dtype") != str(states.dtype).removeprefix("torch.")
        or not isinstance(old_velocities, torch.Tensor)
        or old_velocities.dtype != torch.float32
        or old_velocities.shape != (len(selected), *states.shape[1:])
    ):
        raise ValueError("ART requires original FP16/BF16 states and aligned FP32 old velocities")
    if states.device.type != "cpu" or old_velocities.device.type != "cpu":
        raise ValueError("ART wire caches must own CPU tensor payloads")
    if not torch.isfinite(old_velocities).all() or not torch.isfinite(states).all():
        raise ValueError("Non-finite ART state or old velocity")
    return selected


def _actual_fingerprint(value: Any) -> str:
    """Hash actual CPU response data, including tensor shape/dtype and
    bytes."""
    import torch

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"{")
            for key in sorted(item):
                digest.update(json.dumps(key).encode())
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for element in item:
                update(element)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, allow_nan=False, sort_keys=True).encode())

    update(value)
    return digest.hexdigest()


_CANDIDATE_OPTIONS = frozenset(
    {
        "candidate_id",
        "initial_seed",
        "denoise_seed",
        "denoise_seeds",
    }
)


def _shared_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    shared = deepcopy(dict(request))
    shared.pop("seed", None)
    shared.pop("group_id", None)
    shared.pop("member_id", None)
    options = shared["extra_sampling_params"]["rollout_options"]
    for key in _CANDIDATE_OPTIONS:
        options.pop(key, None)
    return shared


def can_batch_sensenova_requests(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """A group shares conditioning, schedule and capture selection."""
    return (
        is_sensenova_rollout_request(left)
        and is_sensenova_rollout_request(right)
        and left.get("num_outputs_per_prompt") == right.get("num_outputs_per_prompt") == 1
        and _shared_request(left) == _shared_request(right)
    )


def pack_sensenova_rollout_requests(requests: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not requests or any(not can_batch_sensenova_requests(requests[0], req) for req in requests):
        raise ValueError("A native rollout batch requires compatible single-candidate requests")
    packed = _shared_request(requests[0])
    packed["seed"] = [req["seed"] for req in requests]
    packed["num_outputs_per_prompt"] = len(requests)
    candidates = []
    for request in requests:
        options = request["extra_sampling_params"]["rollout_options"]
        candidates.append(
            {"seed": request["seed"], **{key: options[key] for key in _CANDIDATE_OPTIONS if key in options}}
        )
    if len({item.get("candidate_id") for item in candidates}) != len(requests):
        raise ValueError("Candidate IDs must be unique within a native batch")
    packed["extra_sampling_params"]["rollout_options"]["candidates"] = candidates
    return packed
