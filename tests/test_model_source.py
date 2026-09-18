# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils import model_source as m


@pytest.fixture(autouse=True)
def reset_provider_registry(monkeypatch):
    monkeypatch.setattr(m, "_PROVIDERS", {})
    monkeypatch.setattr(m, "_FROZEN", False)


def test_resolve_model_source_rejects_multiple_matches_and_freezes_registry():
    m.register_model_source_provider("a", lambda argv: m.ModelSource("s3://a/model/"))
    m.register_model_source_provider("b", lambda argv: m.ModelSource("s3://b/model/"))

    with pytest.raises(RuntimeError, match="Multiple model source providers matched: a, b"):
        m.resolve_model_source(["train.py"])
    with pytest.raises(RuntimeError, match="registry is frozen"):
        m.register_model_source_provider("c", lambda argv: None)


def test_resolve_model_source_reports_invalid_provider_result():
    m.register_model_source_provider("broken", lambda argv: "s3://bucket/model/")

    with pytest.raises(RuntimeError, match="provider 'broken' failed: expected ModelSource or None"):
        m.resolve_model_source(["train.py"])


def test_model_source_aliases_normalize_local_paths_and_ignore_empty_values():
    args = SimpleNamespace(
        model_source=m.ModelSource("s3://bucket/model/"),
        _model_source_original_hf_checkpoint="/models/original/",
    )

    assert m.is_model_source_alias(args, "/models/original")
    assert not m.is_model_source_alias(args, "")
    assert "" not in m.model_source_path_aliases(args, args.model_source.uri)
    assert "." not in m.model_source_path_aliases(args, args.model_source.uri)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/model/../model",
        "somesource://models/model/../model",
        "https://models.example.com/model/../model",
        "file:///models/model/../model",
    ],
)
def test_normalize_model_path_preserves_hierarchical_uris(uri):
    assert m.is_model_uri(uri)
    assert m.normalize_model_path(uri) == uri


def test_normalize_model_path_normalizes_absolute_and_relative_local_paths():
    assert m.normalize_model_path("/models/parent/../model") == "/models/model"
    assert m.normalize_model_path("models/parent/../model") == "models/model"


def test_uri_shape_detection_does_not_parse_malformed_authority():
    malformed_uri = "somesource://[invalid-authority"

    assert m.is_model_uri(malformed_uri)
    assert m.normalize_model_path(malformed_uri) == malformed_uri


def test_validate_ref_load_defers_same_model_source_uri(monkeypatch):
    from relax.utils import arguments

    logical_uri = "somesource://models/policy"
    args = SimpleNamespace(
        ref_load=logical_uri,
        model_source=m.ModelSource("s3://bucket/models/qwen35/"),
        _model_source_original_hf_checkpoint=logical_uri,
    )
    monkeypatch.setattr(
        arguments.os.path, "exists", lambda _path: pytest.fail("must not validate source alias locally")
    )

    arguments._validate_ref_load(args)


def test_validate_ref_load_rejects_distinct_model_uri():
    from relax.utils import arguments

    args = SimpleNamespace(
        ref_load="somesource://models/reference",
        model_source=m.ModelSource("s3://bucket/models/policy/"),
        _model_source_original_hf_checkpoint="somesource://models/policy",
    )

    with pytest.raises(ValueError, match="does not match the configured model source"):
        arguments._validate_ref_load(args)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("with_model_source", [False, True])
def test_validate_ref_load_keeps_local_reference_behavior(tmp_path, monkeypatch, relative, with_model_source):
    from relax.utils import arguments

    reference = tmp_path / "bf16-reference"
    reference.mkdir()
    (reference / "latest_checkpointed_iteration.txt").write_text("1")
    if relative:
        monkeypatch.chdir(tmp_path)
        ref_load = reference.name
    else:
        ref_load = str(reference)
    args = SimpleNamespace(ref_load=ref_load)
    if with_model_source:
        args.model_source = m.ModelSource("s3://bucket/models/int4-policy/")
        args._model_source_original_hf_checkpoint = "somesource://models/int4-policy"

    arguments._validate_ref_load(args)


def test_validate_ref_load_without_model_source_still_requires_local_path(tmp_path):
    from relax.utils import arguments

    args = SimpleNamespace(ref_load=str(tmp_path / "missing-reference"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        arguments._validate_ref_load(args)


def test_provider_source_is_attached_before_same_uri_ref_validation(monkeypatch):
    from relax.backends.megatron import arguments as megatron_arguments
    from relax.utils import arguments

    logical_uri = "somesource://models/policy"
    source = m.ModelSource("s3://bucket/models/policy/", provider_name="test")
    parsed = SimpleNamespace(
        hf_checkpoint=logical_uri,
        ref_load=logical_uri,
        train_backend="megatron",
        start_rollout_id=None,
        critic_train_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )
    pre = SimpleNamespace(
        debug_train_only=True,
        debug_rollout_only=True,
        load_debug_rollout_data=None,
        skip_hf_validate=False,
    )
    provider_argv = None
    validated = False

    def provider(argv):
        nonlocal provider_argv
        provider_argv = argv
        return source

    def validate(parsed_args):
        nonlocal validated
        assert parsed_args.model_source == source
        assert parsed_args._model_source_original_hf_checkpoint == logical_uri
        assert parsed_args.hf_checkpoint == source.uri
        arguments._validate_ref_load(parsed_args)
        validated = True

    m.register_model_source_provider("test", provider)
    monkeypatch.setattr(arguments.sys, "argv", ["train.py", "--hf-checkpoint", logical_uri, "--ref-load", logical_uri])
    monkeypatch.setattr(arguments, "_pre_parse_cli_model_source", lambda: None)
    monkeypatch.setattr(arguments, "_pre_parse_mode", lambda: pre)
    monkeypatch.setattr(arguments, "get_slime_extra_args_provider", lambda _custom: None)
    monkeypatch.setattr(arguments, "slime_validate_args", validate)
    monkeypatch.setattr(megatron_arguments, "_megatron_parse_args", lambda **_kwargs: parsed)
    monkeypatch.setattr(megatron_arguments, "_derive_cluster_args_from_resource", lambda _args: None)
    monkeypatch.setattr(megatron_arguments, "_set_default_megatron_args", lambda args: args)

    result = arguments.parse_args()

    assert result is parsed
    assert provider_argv == tuple(arguments.sys.argv)
    assert validated


def test_parse_args_passes_provider_source_without_rewriting_argv(monkeypatch):
    from relax.utils import arguments

    source = m.ModelSource("s3://bucket/model/", provider_name="test")
    m.register_model_source_provider("test", lambda argv: source)
    original_argv = ["train.py", "--hf-checkpoint", "/models/original", "--foo", "bar"]
    captured = {}

    def fake_parse(add_custom_arguments=None, *, model_source=None):
        captured["argv"] = list(arguments.sys.argv)
        captured["source"] = model_source
        return "parsed"

    monkeypatch.setattr(arguments.sys, "argv", original_argv)
    monkeypatch.setattr(arguments, "_parse_args_impl", fake_parse)
    monkeypatch.setattr(arguments, "_pre_parse_cli_model_source", lambda: None)

    assert arguments.parse_args() == "parsed"
    assert captured == {
        "argv": original_argv,
        "source": source,
    }
    assert arguments.sys.argv is original_argv


def test_disable_s3_model_download_skips_registered_provider(monkeypatch):
    from relax.utils import arguments

    called = False

    def provider(_argv):
        nonlocal called
        called = True
        return m.ModelSource("s3://bucket/model/")

    m.register_model_source_provider("test", provider)
    monkeypatch.setattr(arguments.sys, "argv", ["train.py", "--disable-s3-model-download"])
    monkeypatch.setattr(arguments, "_parse_args_impl", lambda _custom=None, *, model_source=None: model_source)

    assert arguments.parse_args() is None
    assert not called


def test_megatron_parse_applies_model_source_after_cli_parse(monkeypatch):
    from relax.backends.megatron import arguments

    parsed = SimpleNamespace(
        hf_checkpoint="/models/from-cli",
        critic_train_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )
    source = m.ModelSource("s3://bucket/model/")
    monkeypatch.setattr(arguments, "_megatron_parse_args", lambda **_kwargs: parsed)
    monkeypatch.setattr(arguments, "_derive_cluster_args_from_resource", lambda _args: None)
    monkeypatch.setattr(arguments, "_set_default_megatron_args", lambda args: args)

    result = arguments.megatron_parse_args(None, skip_hf_validate=True, model_source=source)

    assert result._model_source_original_hf_checkpoint == "/models/from-cli"
    assert result.hf_checkpoint == source.uri


def test_megatron_parse_without_model_source_does_not_add_provenance(monkeypatch):
    from relax.backends.megatron import arguments

    parsed = SimpleNamespace(
        hf_checkpoint="/models/from-cli",
        critic_train_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )
    monkeypatch.setattr(arguments, "_megatron_parse_args", lambda **_kwargs: parsed)
    monkeypatch.setattr(arguments, "_derive_cluster_args_from_resource", lambda _args: None)
    monkeypatch.setattr(arguments, "_set_default_megatron_args", lambda args: args)

    result = arguments.megatron_parse_args(None, skip_hf_validate=True)

    assert not hasattr(result, "_model_source_original_hf_checkpoint")
    assert result.hf_checkpoint == "/models/from-cli"
