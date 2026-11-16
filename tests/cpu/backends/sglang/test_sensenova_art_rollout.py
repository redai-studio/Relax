# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART selected-state contract and optional native SGLang CPU capture tests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.sensenova_rollout import (
    ART_PROTOCOL_VERSION,
    build_sensenova_rollout_request,
    validate_sensenova_rollout_response,
)
from relax.models.sensenova.rl_adapter import SenseNovaU1ImageEditAdapter


def _request(member=0, storage_dtype="float16", **changes):
    options = dict(
        seed=11 + member,
        candidate_id=f"g:{member}",
        group_id="g",
        member_id=member,
        algorithm="art",
        height=8,
        width=8,
        num_inference_steps=4,
        guidance_scale=1.0,
        storage_dtype=storage_dtype,
        return_step_indices=[0, 2, 3],
    )
    options.update(changes)
    return build_sensenova_rollout_request("edit", ["a.png"], **options)


def _conditions():
    return dict(
        prefix_inputs_embeds=torch.ones(1, 3, 8),
        prefix_indexes=torch.zeros(3, 3),
        prefix_attention_mask=torch.zeros(1, 1, 3, 3),
        indexes_image=torch.zeros(3, 4),
        noise_scale=torch.tensor(2.0),
        t_eps=torch.tensor(0.02),
    )


def _pair(storage_dtype="float16"):
    request = _request(storage_dtype=storage_dtype)
    response = {
        "request_id": "r",
        "seed": 11,
        "denoising_env": {
            "image_kwargs": {
                "sensenova_rollout": {
                    "protocol": "sensenova_i2i_flow_v1",
                    "actual_batch_size": 1,
                    "candidate_id": "g:0",
                    "state_layout": "T,C,H,W",
                    "dynamics": "ode",
                    "transition_indices": [],
                    "sde_indices": [],
                    "score_kind": "none",
                    "initial_seed": 11,
                    "noise_level": 0.0,
                    "denoise_seeds": [21, 22, 23, 24],
                    "conditioning": {
                        "prompt": "edit",
                        "height": 8,
                        "width": 8,
                        "image_guidance_scale": 1.0,
                        "cfg_norm": "none",
                        "cfg_interval": [0.0, 1.0],
                        "timestep_shift": 3.0,
                        "enable_timestep_shift": True,
                        "t_eps": 0.02,
                    },
                }
            }
        },
        "dit_trajectory": {
            "timesteps": torch.tensor([1000.0, 750.0, 500.0, 250.0]),
            "sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
        },
        "rollout_log_probs": None,
        "generated_output": torch.zeros(3, 8, 8),
    }
    response["denoising_env"]["image_kwargs"]["sensenova_rollout"].update(
        protocol_version=ART_PROTOCOL_VERSION,
        capture="ode_states",
        storage_dtype=storage_dtype,
        velocity_dtype="float32",
        velocity_convention="noise_directed",
        solver="euler",
        state_indices=[0, 2, 3, 4],
        selected_indices=[0, 2, 3],
    )
    response["denoising_env"]["image_kwargs"]["sensenova_rollout"]["conditioning"].update(
        image_paths=["a.png"], guidance_scale=1.0
    )
    response["denoising_env"]["pos_cond_kwargs"] = _conditions()
    response["dit_trajectory"].update(
        latents=torch.arange(4).view(4, 1, 1, 1).expand(4, 3, 8, 8).to(getattr(torch, storage_dtype)),
        old_velocities=torch.full((3, 3, 8, 8), 2**-14, dtype=torch.float32),
    )
    return request, response


def _pack(request, response):
    mapped = validate_sensenova_rollout_response(response, request)
    mapped.update(policy_version=3, weight_manifest_sha256="a" * 64)
    adapter = SenseNovaU1ImageEditAdapter()
    adapter.validate_rollout_response(mapped)
    return mapped, adapter.pack_trajectory(mapped)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_art_selected_cache_preserves_actual_dtype_and_sparse_scheduler_coordinates(dtype):
    from relax.backends.fsdp.art import art_steps_from_batch
    from relax.engine.rollout.native_generation import adapter_pack_group

    request, response = _pair(dtype)
    mapped, packed = _pack(request, response)
    assert mapped["old_velocities"] is response["dit_trajectory"]["old_velocities"]
    assert packed["art_states"].dtype == packed["terminal_state"].dtype == getattr(torch, dtype)
    assert packed["art_old_velocities"].dtype == torch.float32
    assert packed["art_step_indices"].tolist() == [0, 2, 3]
    assert packed["art_timesteps"].tolist() == [1000, 500, 250]
    assert packed["art_sigmas"].tolist() == [1, 0.5, 0.25]
    assert packed["art_ode_dts"].tolist() == [-0.25, -0.25, -0.25]
    assert packed["terminal_state"].flatten()[0] == 3  # Raw state, not the clipped reward image.
    assert "rollout_old_logp" not in packed and packed["sde_indices"].numel() == 0
    common, conditions, tracks = adapter_pack_group(SenseNovaU1ImageEditAdapter(), [mapped])
    batch = {**common, **conditions, **tracks}
    steps = art_steps_from_batch(batch)
    assert len(steps) == 3
    for slot, step in enumerate(steps):
        torch.testing.assert_close(step.state, batch["art_states"][:, slot], rtol=0, atol=0)
        torch.testing.assert_close(step.timestep, batch["art_timesteps"][slot : slot + 1], rtol=0, atol=0)
        assert step.state.untyped_storage().data_ptr() == batch["art_states"].untyped_storage().data_ptr()


@pytest.mark.parametrize("indices", [None, [], [0, 0], [-1], [4], [True]])
def test_art_builder_rejects_missing_or_non_input_step_selection(indices):
    with pytest.raises(ValueError, match="selected input"):
        _request(return_step_indices=indices)


@pytest.mark.parametrize("change", [{"dynamics": "cps"}, {"sde_step_indices": [1]}, {"sde_num_steps": 1}])
def test_art_cannot_reuse_stochastic_capture(change):
    with pytest.raises(ValueError, match="deterministic ODE"):
        _request(**change)


@pytest.mark.parametrize(
    "target,mutate",
    [
        ("metadata", lambda m: m.update(protocol_version=4)),
        ("trajectory", lambda t: t.pop("old_velocities")),
        ("trajectory", lambda t: t.update(old_velocities=t["old_velocities"].half())),
        ("trajectory", lambda t: t["old_velocities"].fill_(float("nan"))),
        ("trajectory", lambda t: t.update(old_velocities=torch.zeros(4, 3, 8, 8))),
        ("metadata", lambda m: m.update(selected_indices=[2, 0, 3])),
        ("metadata", lambda m: m.update(selected_indices=[0, 2, 4])),
        ("metadata", lambda m: m.update(state_indices=[0, 2, 3])),
        ("metadata", lambda m: m.update(velocity_convention="clean_directed")),
        ("metadata", lambda m: m.update(solver="heun")),
        ("response", lambda r: r.update(rollout_log_probs=torch.zeros(3))),
        ("trajectory", lambda t: t.update(timesteps=t["timesteps"].bfloat16())),
    ],
    ids=[
        "version",
        "missing_velocity",
        "velocity_dtype",
        "velocity_nan",
        "extra_velocity",
        "selected_order",
        "selected_terminal",
        "missing_terminal",
        "sign",
        "solver",
        "score",
        "grid_dtype",
    ],
)
def test_art_decoder_rejects_unsafe_or_downgraded_cache(target, mutate):
    request, response = _pair()
    mutate(
        {
            "trajectory": response["dit_trajectory"],
            "response": response,
            "metadata": response["denoising_env"]["image_kwargs"]["sensenova_rollout"],
        }[target]
    )
    with pytest.raises(ValueError):
        validate_sensenova_rollout_response(response, request)


def test_group_packing_does_not_slice_shared_steps_when_candidate_count_equals_selected_count():
    from relax.engine.rollout.native_generation import _shard_group_batch, adapter_pack_group

    request, response = _pair()
    mapped, _ = _pack(request, response)
    responses = [deepcopy(mapped) for _ in range(3)]
    for index, item in enumerate(responses):
        item["old_velocities"].fill_(index + 0.125)
    common, conditions, tracks = adapter_pack_group(SenseNovaU1ImageEditAdapter(), responses)
    assert common["art_timesteps"].shape == (3,)
    assert tracks["art_old_velocities"].shape == (3, 3, 3, 8, 8)
    local = _shard_group_batch({**common, **conditions, **tracks}, 3, [2])
    assert torch.equal(local["art_timesteps"], common["art_timesteps"])
    assert torch.equal(local["art_old_velocities"], tracks["art_old_velocities"][2:3])
    responses[1]["timesteps"][1] = 800
    responses[1]["sigmas"][1] = 0.8
    with pytest.raises(ValueError, match="different sigmas"):
        adapter_pack_group(SenseNovaU1ImageEditAdapter(), responses)


def test_art_adapter_request_uses_explicit_ode_steps_and_requires_exact_replay_time():
    adapter = SenseNovaU1ImageEditAdapter()
    sample = SimpleNamespace(prompt="edit", metadata={"images": ["a.png"]})
    config = dict(algorithm="art", num_inference_steps=4, guidance_scale=1.0, art={"train_step_indices": [0, 2, 3]})
    request = adapter.build_rollout_request(sample, config, seed=1)
    assert request["rollout_return_step_indices"] == [0, 2, 3]
    assert request["extra_sampling_params"]["rollout_options"]["capture"] == "ode_states"
    received = []

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            received.append(kwargs["replay_timestep"])
            return kwargs["replay_latents"]

    exact_time = torch.tensor([723.412841796875])
    batch = dict(
        _sensenova_prefix_cache=object(),
        _replay_timestep=exact_time,
        cond_indexes_image=torch.zeros(3, 4),
        cond_noise_scale=torch.tensor([2.0]),
    )
    adapter.predict_training_velocity(Model(), batch, torch.ones(1, 3, 8, 8), torch.tensor([0.5]))
    assert received[0] is exact_time


def test_art_tail_grid_is_not_replaced_by_t_eps():
    request, response = _pair()
    sigmas = torch.tensor([1.0, 0.5, 0.125, 0.01, 0.0])
    response["dit_trajectory"].update(sigmas=sigmas, timesteps=sigmas[:-1] * 1000)
    _, packed = _pack(request, response)
    torch.testing.assert_close(packed["art_sigmas"][-1], torch.tensor(0.01), rtol=0, atol=0)
    torch.testing.assert_close(packed["art_ode_dts"][-1], torch.tensor(-0.01), rtol=0, atol=0)
    torch.testing.assert_close(packed["cond_t_eps"], torch.tensor(0.02), rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["velocity_dtype", "t_eps", "grid_dtype"])
def test_art_adapter_also_rejects_invalid_cached_payloads_without_wire_revalidation(fault):
    request, response = _pair()
    mapped, _ = _pack(request, response)
    if fault == "velocity_dtype":
        mapped["old_velocities"] = mapped["old_velocities"].bfloat16()
    elif fault == "t_eps":
        mapped["denoising_env"]["pos_cond_kwargs"].pop("t_eps")
    else:
        mapped["timesteps"] = mapped["timesteps"].half()
    with pytest.raises(ValueError, match="ART"):
        SenseNovaU1ImageEditAdapter().pack_trajectory(mapped)


@pytest.mark.parametrize("legacy_runtime", [False, True])
def test_art_preflight_requires_velocity_protocol_without_lora(monkeypatch, legacy_runtime):
    import sys
    import types
    from dataclasses import make_dataclass

    from relax.backends.sglang.diffusion_engine import _verify_diffusion_patch_contract

    def model_method(*args, **kwargs):
        return None

    model = type(
        "NEOChatModel",
        (),
        {
            name: model_method
            for name in (
                "prepare_it2i_conditioning",
                "initialize_it2i_latents",
                "it2i_timesteps",
                "predict_it2i_velocity",
            )
        },
    )
    modules = {
        "sglang.multimodal_gen.configs.sample.sensenova_u1": {
            "SenseNovaU1SamplingParams": make_dataclass("Sampling", [("rollout_options", dict)])
        },
        "sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat": {"NEOChatModel": model},
        "sglang.multimodal_gen.runtime.models.sensenova_u1.rollout": {"ART_ROLLOUT_PROTOCOL_VERSION": 5},
        "sglang.multimodal_gen.runtime.models.sensenova_u1.weights": {"validate_dense_bucket": model_method},
        "sglang.multimodal_gen.runtime.post_training.rl_dataclasses": {
            "RolloutDitTrajectory": make_dataclass(
                "Trajectory", [] if legacy_runtime else [("old_velocities", object)]
            )
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    config = type("SenseNovaU1PipelineConfig", (), {})()
    if legacy_runtime:
        with pytest.raises(RuntimeError, match="v0.5.21-sensenova-art.patch"):
            _verify_diffusion_patch_contract(config)
    else:
        _verify_diffusion_patch_contract(config)
