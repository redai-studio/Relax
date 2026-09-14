import importlib.util
import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import ray
import torch
from ray import serve

from relax.entrypoints import doctor


MODEL_ARGS = [
    "--swiglu",
    "--num-layers",
    "2",
    "--hidden-size",
    "128",
    "--ffn-hidden-size",
    "256",
    "--num-attention-heads",
    "4",
    "--group-query-attention",
    "--num-query-groups",
    "2",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--vocab-size",
    "128",
    "--kv-channels",
    "32",
]


@pytest.fixture
def training_argv(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    prompt = tmp_path / "prompt.jsonl"
    prompt.write_text('{"prompt": "1+1", "label": "2"}\n', encoding="utf-8")
    return [
        "--skip-hf-validate",
        "--resource",
        '{"actor": [1, 1], "rollout": [1, 1]}',
        "--colocate",
        "--hf-checkpoint",
        str(model),
        "--ref-load",
        str(model),
        "--prompt-data",
        str(prompt),
        "--num-rollout",
        "2",
        "--rollout-batch-size",
        "2",
        "--n-samples-per-prompt",
        "2",
        "--global-batch-size",
        "4",
        "--rollout-max-response-len",
        "128",
        "--micro-batch-size",
        "1",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--advantage-estimator",
        "grpo",
        *MODEL_ARGS,
    ]


@pytest.fixture
def optional_sglang_backend(monkeypatch):
    """Keep the real Relax parser while isolating an unavailable optional
    backend."""
    force_missing = os.environ.get("RELAX_TEST_WITHOUT_SGLANG") == "1"
    if not force_missing and importlib.util.find_spec("sglang") is not None:
        return

    from relax.utils import arguments

    monkeypatch.setattr(arguments, "_parse_sglang_namespaces", lambda: (Namespace(), Namespace()))
    monkeypatch.setattr(arguments, "_validate_sglang_args", lambda _args: None)


@pytest.fixture
def megatron_backend():
    """Require the real parser instead of replacing its authoritative
    config."""
    if importlib.util.find_spec("megatron") is None:
        pytest.skip("Megatron is not installed in the minimal CI environment")


def _error_cases():
    path = Path(__file__).parent / "fixtures" / "doctor_errors.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("case", _error_cases(), ids=lambda case: case["name"])
def test_error_library_uses_real_doctor(training_argv, case, capsys, optional_sglang_backend, megatron_backend):
    argv = list(training_argv)
    if case.get("remove_prompt"):
        index = argv.index("--prompt-data")
        del argv[index : index + 2]
    for flag in case.get("remove_flags", []):
        argv.remove(flag)
    argv.extend(case["args"])

    assert doctor.main(["--format", "json", "--", *argv]) == 1
    captured = capsys.readouterr()
    report = json.loads(captured.err)
    assert case["contains"] in report["error"]
    assert case["fix"] in report["suggestion"]
    assert "Traceback" not in captured.err


def test_valid_generalized_dataset_path_returns_success(
    training_argv, tmp_path, capsys, optional_sglang_backend, megatron_backend
):
    second_prompt = tmp_path / "prompt-2.jsonl"
    second_prompt.write_text('{"prompt": "2+2", "label": "4"}\n', encoding="utf-8")
    prompt_index = training_argv.index("--prompt-data") + 1
    training_argv[prompt_index] = f"[{training_argv[prompt_index]},{second_prompt}]@[0:2]"

    assert doctor.main(["--format", "json", "--", *training_argv]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["roles"] == ["actor", "rollout"]
    assert report["resources"]["total_gpus"] == 1


def test_empty_dataset_directory_returns_nonzero(
    training_argv, tmp_path, capsys, optional_sglang_backend, megatron_backend
):
    empty_directory = tmp_path / "empty-dataset"
    empty_directory.mkdir()
    prompt_index = training_argv.index("--prompt-data") + 1
    training_argv[prompt_index] = str(empty_directory)

    assert doctor.main(["--format", "json", "--", *training_argv]) == 1
    report = json.loads(capsys.readouterr().err)
    assert "resolved to no supported files" in report["error"]
    assert "add a .jsonl or .parquet file" in report["suggestion"]


def test_existing_unsupported_dataset_file_returns_nonzero(
    training_argv, tmp_path, capsys, optional_sglang_backend, megatron_backend
):
    unsupported = tmp_path / "prompt.txt"
    unsupported.write_text("1+1\n", encoding="utf-8")
    prompt_index = training_argv.index("--prompt-data") + 1
    training_argv[prompt_index] = str(unsupported)

    assert doctor.main(["--format", "json", "--", *training_argv]) == 1
    report = json.loads(capsys.readouterr().err)
    assert "Unsupported dataset file format" in report["error"]
    assert "convert each file" in report["suggestion"]


def test_custom_sft_dataset_owns_its_training_path_semantics():
    from relax.utils.arguments import _validate_dataset_paths

    args = Namespace(
        loss_type="sft",
        custom_dataset_class_path="example.CustomDataset",
        prompt_data="custom://dataset-id",
        eval_prompt_data=None,
    )

    _validate_dataset_paths(args)


def test_custom_sft_dataset_still_validates_builtin_eval_path(tmp_path):
    from relax.utils.arguments import _validate_dataset_paths

    missing_eval = tmp_path / "missing-eval.jsonl"
    args = Namespace(
        loss_type="sft",
        custom_dataset_class_path="example.CustomDataset",
        prompt_data="custom://dataset-id",
        eval_prompt_data=["eval", str(missing_eval)],
    )

    with pytest.raises(FileNotFoundError, match="missing-eval.jsonl"):
        _validate_dataset_paths(args)


@pytest.mark.parametrize(
    ("rollout_global_dataset", "data_source_path"),
    [
        (False, "relax.engine.rollout.data_source.RolloutDataSourceWithBuffer"),
        (True, "example.CustomDataSource"),
    ],
)
def test_rl_skips_prompt_path_not_owned_by_builtin_loader(rollout_global_dataset, data_source_path):
    from relax.utils.arguments import _validate_dataset_paths

    args = Namespace(
        loss_type="rl",
        rollout_global_dataset=rollout_global_dataset,
        data_source_path=data_source_path,
        prompt_data="custom://dataset-id",
        eval_prompt_data=None,
        eval_datasets=[],
    )

    _validate_dataset_paths(args)


def test_report_uses_registry_roles_and_colocate_resources(megatron_backend):
    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=False,
        fully_async=False,
        colocate=True,
        genrm_model_path=None,
        resource={"actor": [1, 8], "rollout": [1, 8]},
    )

    report = doctor._report(args, ["--resource", '{"actor": [1, 8], "rollout": [1, 8]}'])

    assert report["roles"] == ["actor", "rollout"]
    assert report["resources"]["total_gpus"] == 8
    assert report["resources"]["shares_actor_rollout_gpus"] is True


def test_hybrid_resources_are_not_colocated(megatron_backend):
    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=True,
        fully_async=True,
        colocate=True,
        genrm_model_path=None,
        resource={"actor": [1, 8], "rollout": [1, 8]},
    )

    report = doctor._report(args, [])

    assert report["resources"]["total_gpus"] == 16
    assert report["resources"]["shares_actor_rollout_gpus"] is False


@pytest.mark.parametrize(
    ("colocate", "expected_total", "expected_sharing"),
    [(True, 8, True), (False, 16, False)],
)
def test_managed_opd_teacher_resources_follow_runtime_placement(
    megatron_backend, colocate, expected_total, expected_sharing
):
    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=False,
        fully_async=False,
        colocate=colocate,
        genrm_model_path=None,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/models/teacher",
        opd_teacher_routes=None,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
    )

    report = doctor._report(args, [])

    assert report["roles"] == ["actor", "rollout", "teacher"]
    assert report["resources"]["total_gpus"] == expected_total
    assert report["resources"]["shares_actor_rollout_gpus"] is expected_sharing


def test_debug_train_only_omits_managed_opd_teacher(megatron_backend):
    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=True,
        hybrid=False,
        fully_async=False,
        colocate=False,
        genrm_model_path=None,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/models/teacher",
        opd_teacher_routes=None,
        resource={"actor": [1, 8], "teacher": [1, 4]},
    )

    report = doctor._report(args, [])

    assert report["roles"] == ["actor"]
    assert report["resources"]["total_gpus"] == 8


def test_sft_report_does_not_claim_actor_rollout_gpu_sharing(megatron_backend):
    args = Namespace(
        loss_type="sft",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=False,
        fully_async=False,
        colocate=True,
        genrm_model_path=None,
        sft_predict_interval=None,
        resource={"sft": [1, 0], "actor": [1, 8]},
    )

    report = doctor._report(args, [])

    assert report["roles"] == ["sft", "actor"]
    assert report["resources"]["total_gpus"] == 8
    assert report["resources"]["shares_actor_rollout_gpus"] is False


def test_preflight_rejects_zero_gpu_for_an_active_model_role(megatron_backend):
    from relax.utils.arguments import validate_preflight_args

    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=False,
        fully_async=True,
        true_on_policy_mode=True,
        genrm_model_path=None,
        resource={
            "actor": [1, 1],
            "rollout": [1, 1],
            "reference": [1, 0],
            "advantages": [1, 0],
        },
        prompt_data=None,
        eval_prompt_data=None,
        eval_datasets=[],
    )

    with pytest.raises(ValueError, match="Active model role.*num_gpus > 0"):
        validate_preflight_args(args)


def test_preflight_rejects_zero_gpu_for_managed_teacher(megatron_backend):
    from relax.utils.arguments import validate_preflight_args

    args = Namespace(
        loss_type="rl",
        advantage_estimator="grpo",
        debug_rollout_only=False,
        debug_train_only=False,
        hybrid=False,
        fully_async=False,
        true_on_policy_mode=False,
        genrm_model_path=None,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/models/teacher",
        opd_teacher_routes=None,
        resource={"actor": [1, 1], "rollout": [1, 1], "teacher": [1, 0]},
        prompt_data=None,
        eval_prompt_data=None,
        eval_datasets=[],
    )

    with pytest.raises(ValueError) as exc_info:
        validate_preflight_args(args)
    assert "teacher" in str(exc_info.value)
    assert "num_gpus > 0" in str(exc_info.value)


def test_redacts_success_and_error_output():
    argv = [
        "--wandb-key",
        "WANDB_SECRET",
        "--train-env-vars",
        '{"OPENAI_API_KEY": "TRAIN_SECRET", "SAFE": "visible"}',
        "--agent-env",
        "TOKEN=AGENT_SECRET",
    ]

    sanitized = " ".join(doctor._redact_argv(argv))
    error = doctor._redact_error("bad WANDB_SECRET TRAIN_SECRET AGENT_SECRET", argv)

    assert "WANDB_SECRET" not in sanitized + error
    assert "TRAIN_SECRET" not in sanitized + error
    assert "AGENT_SECRET" not in sanitized + error
    assert sanitized.count("<redacted>") == 3


def test_malformed_structured_secret_is_redacted_as_a_whole():
    argv = ["--train-env-vars", '{"OPENAI_API_KEY": "BROKEN_SECRET"']
    message = doctor._redact_error(f"invalid value {argv[1]}", argv)

    assert "BROKEN_SECRET" not in " ".join(doctor._redact_argv(argv))
    assert "BROKEN_SECRET" not in message


def test_normal_token_and_key_configuration_is_not_redacted():
    config = {"tokenizer_model": "/model", "max_tokens_per_gpu": 4096, "reward_key": "score"}

    assert doctor._redact(config) == config


def test_bare_credential_names_are_redacted():
    config = {
        "api_key": "API_SECRET",
        "access_key": "ACCESS_SECRET",
        "auth_token": "AUTH_SECRET",
        "private_key": "PRIVATE_SECRET",
    }
    argv = ["--api-key", "API_SECRET"]

    sanitized = json.dumps(doctor._redact(config)) + " ".join(doctor._redact_argv(argv))
    error = doctor._redact_error("bad API_SECRET", argv)

    for secret in config.values():
        assert secret not in sanitized + error


def test_main_does_not_start_runtime(monkeypatch, training_argv, capsys, optional_sglang_backend, megatron_backend):
    from relax.backends.megatron import arguments as megatron_arguments
    from relax.utils import arguments

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime side effect attempted")

    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: args)
    monkeypatch.setattr(arguments, "_validate_sglang_args", lambda args: None)
    monkeypatch.setattr(ray, "init", forbidden)
    monkeypatch.setattr(serve, "start", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    assert doctor.main(["--format", "json", "--", *training_argv]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"


def test_main_parser_output_capture_has_a_file_descriptor(monkeypatch, capsys):
    from relax.utils import arguments

    def parse_args(*, strict):
        assert strict is True
        assert hasattr(arguments.sys.stdout, "fileno")
        arguments.sys.stdout.fileno()
        raise ValueError("stop after fileno check")

    monkeypatch.setattr(arguments, "parse_args", parse_args)

    assert doctor.main(["--", "--resource", "{}"]) == 1
    assert "UnsupportedOperation" not in capsys.readouterr().err


def test_main_returns_nonzero_without_traceback(monkeypatch, capsys):
    from relax.utils import arguments

    monkeypatch.setattr(arguments, "parse_args", lambda strict: (_ for _ in ()).throw(ValueError("bad config")))

    assert doctor.main(["--", "--resource", "{}"]) == 1
    captured = capsys.readouterr()
    assert "bad config" in captured.err
    assert "Fix:" in captured.err
    assert "Traceback" not in captured.err


def test_missing_dependency_has_install_suggestion(monkeypatch, capsys):
    from relax.utils import arguments

    missing = ModuleNotFoundError("No module named 'example_backend'")
    missing.name = "example_backend"
    monkeypatch.setattr(arguments, "parse_args", lambda strict: (_ for _ in ()).throw(missing))

    assert doctor.main(["--format", "json", "--", "--resource", "{}"]) == 1
    report = json.loads(capsys.readouterr().err)
    assert "example_backend" in report["error"]
    assert "install 'example_backend'" in report["suggestion"]


@pytest.mark.parametrize("help_option", ["-h", "--help"])
def test_training_help_bypasses_resource_prevalidation(monkeypatch, help_option):
    from relax.utils import arguments

    def forbidden(_args):
        raise AssertionError("resource validation should not run for help")

    monkeypatch.setattr(arguments, "_validate_resource_config", forbidden)
    monkeypatch.setattr(arguments.sys, "argv", ["relax.entrypoints.train", help_option])

    arguments._prevalidate_resource_cli()


def test_outdated_dependency_preserves_upgrade_command():
    report = doctor._error_report(RuntimeError("dependency is old. Upgrade with: pip install dependency"), [])

    assert report["error"] == "RuntimeError: dependency is old."
    assert report["suggestion"] == "upgrade with: pip install dependency"
