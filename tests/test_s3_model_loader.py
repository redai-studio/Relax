# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils import s3_model_loader as m


@pytest.fixture(autouse=True)
def arguments_module(monkeypatch):
    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    module_name = "relax.utils.arguments"
    original_module = sys.modules.get(module_name)
    module_was_loaded = module_name in sys.modules
    utils_module = sys.modules.get("relax.utils")
    arguments_attr_was_set = utils_module is not None and hasattr(utils_module, "arguments")
    original_arguments_attr = getattr(utils_module, "arguments", None) if arguments_attr_was_set else None

    sys.modules.pop(module_name, None)
    if utils_module is not None and arguments_attr_was_set:
        delattr(utils_module, "arguments")
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if module_was_loaded:
            sys.modules[module_name] = original_module
        if utils_module is not None:
            if arguments_attr_was_set:
                setattr(utils_module, "arguments", original_arguments_attr)
            elif hasattr(utils_module, "arguments"):
                delattr(utils_module, "arguments")


@pytest.fixture(autouse=True)
def sglang_engine_module(monkeypatch):
    ray = ModuleType("ray")
    ray.get_runtime_context = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "ray", ray)

    sglang_router = ModuleType("sglang_router")
    sglang_router.__version__ = "0.3.2"
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    sglang = ModuleType("sglang")
    sglang_srt = ModuleType("sglang.srt")
    server_args = ModuleType("sglang.srt.server_args")

    @dataclass
    class ServerArgs:
        model_path: str = ""
        trust_remote_code: bool = True
        random_seed: int = 0
        enable_memory_saver: bool = False
        host: str = ""
        port: int = 0
        nccl_port: int = 0
        nnodes: int = 1
        node_rank: int = 0
        dist_init_addr: str = ""
        gpu_id_step: int = 1
        base_gpu_id: int = 0
        tp_size: int = 1
        dp_size: int = 1
        pp_size: int = 1
        ep_size: int = 1
        moe_dense_tp_size: int | None = None
        skip_server_warmup: bool = False
        enable_draft_weights_cpu_backup: bool = False
        enable_weights_cpu_backup: bool = False
        load_format: str = "auto"

    server_args.LOAD_FORMAT_CHOICES = ("auto", "runai_streamer", "remote", "dummy")
    server_args.ServerArgs = ServerArgs
    sglang_utils = ModuleType("sglang.srt.utils")
    sglang_utils.kill_process_tree = lambda _pid: None
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)

    checkpoint_client = ModuleType("relax.distributed.checkpoint_service.client.engine")
    checkpoint_client.create_client = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "relax.distributed.checkpoint_service.client.engine", checkpoint_client)

    ray_actor = ModuleType("relax.distributed.ray.ray_actor")
    ray_actor.RayActor = object
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.ray_actor", ray_actor)

    device = ModuleType("relax.utils.device")
    device.get_visible_devices_env_var = lambda: "CUDA_VISIBLE_DEVICES"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    async_utils = ModuleType("relax.utils.async_utils")
    async_utils.run = lambda value: value
    monkeypatch.setitem(sys.modules, "relax.utils.async_utils", async_utils)

    env = ModuleType("relax.utils.env")
    env.Envs = SimpleNamespace(
        RELAX_OPTIMIZE_ROUTING_REPLAY=False,
        RELAX_OPD_PREEXPANDED_PATCH=False,
        RELAX_OPD_PER_POS_TOKEN_IDS=False,
        RELAX_OPD_TOKEN_IDS_LOGPROB_K="0",
        RELAX_SCALE_OUT_MAX_REASON_ITEMS=3,
        RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN=120,
        RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN=512,
        RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES=512 * 1024**2,
    )
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils.get_host_info = lambda: ("worker", "127.0.0.1")
    http_utils.router_worker_base_url = lambda host, port, worker_id: f"http://{host}:{port}/workers/{worker_id}"
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)

    logging_utils = ModuleType("relax.utils.logging_utils")
    logging_utils.get_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", logging_utils)

    megatron_peft_utils = ModuleType("relax.utils.megatron_peft_utils")
    megatron_peft_utils.convert_megatron_to_sglang_target_modules = lambda value: value
    megatron_peft_utils.is_lora_enabled = lambda _args: False
    monkeypatch.setitem(sys.modules, "relax.utils.megatron_peft_utils", megatron_peft_utils)

    module_name = "relax.backends.sglang.sglang_engine"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def boto3_module(monkeypatch):
    boto3 = ModuleType("boto3")
    boto3.client = lambda *_args, **_kwargs: None
    botocore = ModuleType("botocore")
    botocore_config = ModuleType("botocore.config")

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    botocore_config.Config = Config
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    return boto3


def _assert_cache_payload_removed(dest: str) -> None:
    for suffix in (
        "",
        ".tmp",
        ".done",
        ".done.tmp",
        ".metadata.done",
        ".metadata.done.tmp",
        ".manifest.json",
        ".manifest.json.tmp",
        ".tmp.manifest.json",
        ".tmp.manifest.json.tmp",
    ):
        assert not os.path.exists(dest + suffix)


def test_pre_parse_cli_model_source(monkeypatch):
    from relax.utils.arguments import _pre_parse_cli_model_source

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--hf-checkpoint",
            "s3://bucket/model/",
        ],
    )

    assert _pre_parse_cli_model_source() == m.ModelSource(uri="s3://bucket/model/")


def test_pre_parse_cli_model_source_disabled(monkeypatch):
    from relax.utils.arguments import _pre_parse_cli_model_source

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--hf-checkpoint", "s3://bucket/model/", "--disable-s3-model-download"],
    )

    assert _pre_parse_cli_model_source() is None


def test_read_s3_model_config_uses_model_source_access_policy(monkeypatch):
    class Body:
        def __init__(self):
            self.closed = False

        def read(self):
            return b'{"model_type": "qwen3", "hidden_size": 4096}'

        def close(self):
            self.closed = True

    class Client:
        def __init__(self):
            self.body = Body()
            self.request = None

        def get_object(self, **kwargs):
            self.request = kwargs
            return {"Body": self.body}

    client = Client()
    client_options = {}

    def make_client(**kwargs):
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(m, "_make_s3_client", make_client)
    args = SimpleNamespace(
        model_source=m.ModelSource(
            "s3://bucket/models/qwen3",
            "http://s3.example",
            credential_mode="placeholder",
            addressing_style="path",
        )
    )

    assert m.read_s3_model_config(args) == {"model_type": "qwen3", "hidden_size": 4096}
    assert client_options == {
        "endpoint": "http://s3.example",
        "use_placeholder_credentials": True,
        "use_path_style": True,
        "connect_timeout": 2.0,
        "read_timeout": 3.0,
        "total_max_attempts": 1,
    }
    assert client.request == {"Bucket": "bucket", "Key": "models/qwen3/config.json"}
    assert client.body.closed


def test_read_s3_model_config_rejects_non_mapping_json(monkeypatch):
    class Body:
        def read(self):
            return b"[]"

        def close(self):
            self.closed = True

    body = Body()
    monkeypatch.setattr(
        m,
        "_make_s3_client",
        lambda **_kwargs: SimpleNamespace(get_object=lambda **_kwargs: {"Body": body}),
    )
    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/model/"))

    with pytest.raises(ValueError, match="config.json must contain a JSON object"):
        m.read_s3_model_config(args)
    assert body.closed


def test_s3_policy_dummy_does_not_affect_genrm():
    from relax.backends.sglang.sglang_engine import _compute_genrm_server_args

    args = SimpleNamespace(
        genrm_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        genrm_model_path="/models/genrm",
        seed=1,
        offload_rollout=False,
        genrm_engine_config={},
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_load_format="dummy",
        model_source=m.ModelSource("s3://bucket/student/", "http://s3.example"),
    )

    server_args, _ = _compute_genrm_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=1236,
        base_gpu_id=0,
    )

    assert server_args["load_format"] == "auto"


def test_s3_policy_auto_prefers_ready_shm(monkeypatch):
    from relax.backends.sglang import sglang_engine

    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/student/", "http://s3.example"))
    monkeypatch.setattr(sglang_engine, "get_s3_model_cached_path", lambda uri, obj: "/dev/shm/student")

    resolved = sglang_engine._apply_sglang_policy_load_plan(
        {"model_path": args.model_source.uri, "load_format": "auto"}, args
    )

    assert resolved == {"model_path": "/dev/shm/student", "load_format": "auto"}


def test_s3_policy_plan_is_noop_when_generic_download_is_disabled(monkeypatch):
    from relax.backends.sglang import sglang_engine

    args = SimpleNamespace(model_source=None)
    monkeypatch.setattr(
        sglang_engine,
        "get_s3_model_cached_path",
        lambda uri, obj: pytest.fail("disabled S3 source must not inspect SHM"),
    )

    server_args = {"model_path": "s3://bucket/student/", "load_format": "auto"}

    assert sglang_engine._apply_sglang_policy_load_plan(server_args, args) == server_args


def test_s3_policy_auto_streams_when_shm_not_ready(monkeypatch, tmp_path):
    from relax.backends.sglang import sglang_engine

    for name in (
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_EC2_METADATA_DISABLED",
        "RUNAI_STREAMER_S3_ENDPOINT",
        "RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING",
    ):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(
        model_source=m.ModelSource("s3://bucket/student/", "http://s3.example", addressing_style="path"),
        s3_model_shm_root=str(tmp_path / "missing"),
    )
    monkeypatch.setattr(sglang_engine, "_SGLANG_LOAD_FORMAT_CHOICES", ["auto", "runai_streamer", "remote"])
    monkeypatch.setattr(
        sglang_engine,
        "resolve_s3_model_metadata_to_shm",
        lambda uri, obj: "/dev/shm/student-metadata",
    )

    resolved = sglang_engine._apply_sglang_policy_load_plan(
        {"model_path": args.model_source.uri, "load_format": "auto"}, args
    )

    assert resolved == {
        "model_path": args.model_source.uri,
        "load_format": "runai_streamer",
        "tokenizer_path": "/dev/shm/student-metadata",
    }
    assert "AWS_ENDPOINT_URL" not in os.environ
    assert os.environ["RUNAI_STREAMER_S3_ENDPOINT"] == "http://s3.example"
    assert os.environ["RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING"] == "0"
    assert "AWS_EC2_METADATA_DISABLED" not in os.environ


def test_s3_policy_auto_streams_uniformly_for_multi_node(monkeypatch):
    from relax.backends.sglang import sglang_engine

    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/student/"))
    monkeypatch.setattr(sglang_engine, "get_s3_model_cached_path", lambda uri, obj: "/dev/shm/student")
    monkeypatch.setattr(sglang_engine, "_SGLANG_LOAD_FORMAT_CHOICES", ["auto", "runai_streamer"])
    monkeypatch.setattr(
        sglang_engine,
        "resolve_s3_model_metadata_to_shm",
        lambda uri, obj: "/dev/shm/student-metadata",
    )

    resolved = sglang_engine._apply_sglang_policy_load_plan(
        {"model_path": args.model_source.uri, "load_format": "auto", "nnodes": 2}, args
    )

    assert resolved == {
        "model_path": args.model_source.uri,
        "load_format": "runai_streamer",
        "nnodes": 2,
        "tokenizer_path": "/dev/shm/student-metadata",
    }


def test_explicit_s3_stream_uses_local_metadata_for_tokenizer(monkeypatch):
    from relax.backends.sglang import sglang_engine

    config = m.ModelSource("s3://bucket/student/", "http://s3.example")
    args = SimpleNamespace(model_source=config)
    monkeypatch.setattr(
        sglang_engine,
        "resolve_s3_model_metadata_to_shm",
        lambda uri, obj: "/dev/shm/student-metadata",
    )

    resolved = sglang_engine._apply_sglang_policy_load_plan(
        {"model_path": config.uri, "load_format": "runai_streamer"}, args
    )

    assert resolved == {
        "model_path": config.uri,
        "load_format": "runai_streamer",
        "tokenizer_path": "/dev/shm/student-metadata",
    }


def test_explicit_s3_stream_preserves_explicit_tokenizer_path(monkeypatch):
    from relax.backends.sglang import sglang_engine

    config = m.ModelSource("s3://bucket/student/", "http://s3.example")
    args = SimpleNamespace(model_source=config)
    monkeypatch.setattr(
        sglang_engine,
        "resolve_s3_model_metadata_to_shm",
        lambda uri, obj: pytest.fail("explicit tokenizer path must not trigger metadata preparation"),
    )

    resolved = sglang_engine._apply_sglang_policy_load_plan(
        {
            "model_path": config.uri,
            "load_format": "runai_streamer",
            "tokenizer_path": "/models/tokenizer",
        },
        args,
    )

    assert resolved == {
        "model_path": config.uri,
        "load_format": "runai_streamer",
        "tokenizer_path": "/models/tokenizer",
    }


def test_s3_stream_for_other_model_does_not_use_policy_metadata(monkeypatch):
    from relax.backends.sglang import sglang_engine

    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/policy/"))
    monkeypatch.setattr(
        sglang_engine,
        "resolve_s3_model_metadata_to_shm",
        lambda uri, obj: pytest.fail("another model must not use policy metadata"),
    )
    server_args = {"model_path": "s3://bucket/reward/", "load_format": "runai_streamer"}

    assert sglang_engine._apply_sglang_policy_load_plan(server_args, args) == server_args


def test_s3_policy_dummy_is_explicit_and_uses_shm(monkeypatch):
    from relax.backends.sglang import sglang_engine

    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/student/", "http://s3.example"))
    monkeypatch.setattr(sglang_engine, "get_s3_model_cached_path", lambda uri, obj: None)
    monkeypatch.setattr(sglang_engine, "resolve_s3_model_metadata_to_shm", lambda uri, obj: "/dev/shm/student")

    resolved = sglang_engine._apply_sglang_policy_load_plan(
        {"model_path": args.model_source.uri, "load_format": "dummy"}, args
    )

    assert resolved == {"model_path": "/dev/shm/student", "load_format": "dummy"}


def test_s3_policy_remote_load_format_is_rejected():
    from relax.backends.sglang import sglang_engine

    source = m.ModelSource("s3://bucket/student/")
    args = SimpleNamespace(model_source=source)

    with pytest.raises(ValueError, match="remote.*not an S3 model loader"):
        sglang_engine.build_sglang_load_plan(
            {"model_path": source.uri, "load_format": "remote"},
            args,
        )


def test_s3_policy_dummy_requires_shm_root(tmp_path):
    from relax.backends.sglang import sglang_engine

    config = m.ModelSource("s3://bucket/student/", "http://s3.example")
    args = SimpleNamespace(model_source=config, s3_model_shm_root=str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="SHM root does not exist"):
        sglang_engine._apply_sglang_policy_load_plan({"model_path": config.uri, "load_format": "dummy"}, args)


def test_runai_streamer_env_overrides_stale_credentials(monkeypatch):
    from relax.backends.sglang import sglang_engine

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "stale-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stale-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-session-token")
    args = SimpleNamespace(
        model_source=m.ModelSource("s3://bucket/student/", "http://s3.example", credential_mode="placeholder")
    )

    sglang_engine._configure_runai_streamer_env(args)

    assert os.environ["AWS_ACCESS_KEY_ID"] == "mock"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "mock"
    assert "AWS_SESSION_TOKEN" not in os.environ


def test_runai_streamer_env_preserves_standard_credential_chain_and_explicit_addressing(monkeypatch):
    from relax.backends.sglang import sglang_engine

    monkeypatch.setenv("RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING", "0")
    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED", raising=False)
    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/student/"))

    sglang_engine._configure_runai_streamer_env(args)

    assert os.environ["RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING"] == "0"
    assert "AWS_EC2_METADATA_DISABLED" not in os.environ


def test_runai_streamer_env_uses_service_specific_aws_endpoint_first(monkeypatch):
    from relax.backends.sglang import sglang_engine

    monkeypatch.delenv("RUNAI_STREAMER_S3_ENDPOINT", raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://global.example")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://s3.example")
    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/student/"))

    sglang_engine._configure_runai_streamer_env(args)

    assert os.environ["RUNAI_STREAMER_S3_ENDPOINT"] == "http://s3.example"


def test_runai_streamer_env_preserves_explicit_runai_endpoint(monkeypatch):
    from relax.backends.sglang import sglang_engine

    monkeypatch.setenv("RUNAI_STREAMER_S3_ENDPOINT", "http://runai.example")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://s3.example")
    args = SimpleNamespace(model_source=m.ModelSource("s3://bucket/student/", "http://provider.example"))

    sglang_engine._configure_runai_streamer_env(args)

    assert os.environ["RUNAI_STREAMER_S3_ENDPOINT"] == "http://runai.example"


def test_is_s3_uri():
    assert m.is_s3_uri("s3://b/p/")
    assert m.is_s3_uri("S3://B/P")
    assert not m.is_s3_uri("/dev/shm/x")
    assert not m.is_s3_uri("")
    assert not m.is_s3_uri(None)


def test_parse_s3_uri():
    assert m._parse_s3_uri("s3://bkt/a/b/") == ("bkt", "a/b/")
    assert m._parse_s3_uri("s3://bkt/a/b") == ("bkt", "a/b")
    for invalid_uri in ("s3://bkt/", "s3://bkt", "s3:///prefix"):
        with pytest.raises(ValueError, match="non-empty bucket and prefix"):
            m._parse_s3_uri(invalid_uri)


def test_shm_dest_dir_stable():
    a = m._shm_dest_dir("s3://bkt/a/", "/dev/shm")
    b = m._shm_dest_dir("s3://bkt/a/", "/dev/shm")
    c = m._shm_dest_dir("s3://bkt/other/", "/dev/shm")
    assert a == b and a != c
    assert a.startswith("/dev/shm/relax_model_")
    assert m._shm_dest_dir("s3://bkt/a/", "/dev/shm", "http://one") != m._shm_dest_dir(
        "s3://bkt/a/", "/dev/shm", "http://two"
    )


def test_safe_join_rejects_escape():
    root = "/dev/shm/relax_model_x"
    assert m._safe_join(root, "a/b.bin").startswith(root)
    with pytest.raises(ValueError):
        m._safe_join(root, "../evil")
    with pytest.raises(ValueError):
        m._safe_join(root, "/abs/evil")


def test_resolve_noop_on_local_path():
    class A:
        model_source = m.ModelSource("s3://bkt/pfx/", "http://s3.example")

    assert m.maybe_resolve_s3_model_to_shm("/dev/shm/local", A()) == "/dev/shm/local"
    assert m.maybe_resolve_s3_model_to_shm("/nfs/model", A()) == "/nfs/model"


def test_resolve_noop_when_s3_disabled():
    class A:
        model_source = None

    assert m.maybe_resolve_s3_model_to_shm("s3://bkt/pfx/", A()) == "s3://bkt/pfx/"


def test_resolve_noop_for_non_selected_s3_uri():
    class A:
        model_source = m.ModelSource("s3://selected/student/", "http://s3.example")

    assert m.maybe_resolve_s3_model_to_shm("s3://runai/direct/", A()) == "s3://runai/direct/"


def test_prepare_model_maybe_update_args_updates_matching_process_private_paths(monkeypatch):
    source = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(
        model_source=source,
        _model_source_original_hf_checkpoint="/models/original",
        hf_checkpoint=source.uri,
        tokenizer_model="/models/original",
        load="/models/original",
        ref_load="/models/original/",
    )
    local_model = m.LocalModel(source=source, path="/dev/shm/model", completeness="full")
    monkeypatch.setattr(
        m,
        "prepare_local_model",
        lambda obj, *, completeness="full": local_model,
    )

    assert m.prepare_model_maybe_update_args(args) is local_model
    assert args.hf_checkpoint == "/dev/shm/model"
    assert args.tokenizer_model == "/dev/shm/model"
    assert args.load == "/models/original"
    assert args.ref_load == "/dev/shm/model"


def test_prepare_model_maybe_update_args_remaps_selected_source_uri_without_provenance(monkeypatch):
    source = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(
        model_source=source,
        hf_checkpoint=source.uri,
        tokenizer_model=source.uri,
        load=source.uri,
        ref_load=source.uri,
    )
    local_model = m.LocalModel(source=source, path="/dev/shm/model", completeness="full")
    monkeypatch.setattr(m, "prepare_local_model", lambda obj, *, completeness="full": local_model)

    assert m.prepare_model_maybe_update_args(args) is local_model
    assert args.hf_checkpoint == "/dev/shm/model"
    assert args.tokenizer_model == "/dev/shm/model"
    assert args.load == source.uri
    assert args.ref_load == "/dev/shm/model"


def test_prepare_model_maybe_update_args_preserves_distinct_model_paths(monkeypatch):
    source = m.ModelSource("s3://bucket/packed-model/")
    args = SimpleNamespace(
        model_source=source,
        _model_source_original_hf_checkpoint="/models/packed-int4",
        hf_checkpoint=source.uri,
        tokenizer_model="/models/tokenizer",
        load="/checkpoints/megatron-resume",
        ref_load="/models/bf16-reference",
    )
    local_model = m.LocalModel(source=source, path="/dev/shm/packed-model", completeness="full")
    monkeypatch.setattr(m, "prepare_local_model", lambda obj, *, completeness="full": local_model)

    assert m.prepare_model_maybe_update_args(args) is local_model
    assert args.hf_checkpoint == "/dev/shm/packed-model"
    assert args.tokenizer_model == "/models/tokenizer"
    assert args.load == "/checkpoints/megatron-resume"
    assert args.ref_load == "/models/bf16-reference"


def test_prepare_local_model_metadata_does_not_restore_full_cache(monkeypatch, tmp_path):
    source = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(model_source=source, hf_checkpoint=source.uri)
    dest = tmp_path / "model"
    dest.mkdir()
    (dest / "config.json").write_text("{}")
    metadata_marker = tmp_path / "model.metadata.done"
    metadata_marker.write_text(source.uri)
    full_marker = tmp_path / "model.done"
    weight = dest / "model.safetensors"

    def resolve_metadata(uri, obj):
        assert uri == source.uri and obj is args
        return str(dest)

    monkeypatch.setattr(m, "resolve_s3_model_metadata_to_shm", resolve_metadata)
    local_model = m.prepare_model_maybe_update_args(args, completeness="metadata")

    assert local_model.completeness == "metadata"
    assert metadata_marker.exists()
    assert not full_marker.exists()
    assert not weight.exists()


def test_prepare_local_model_full_can_refill_cleaned_weights(monkeypatch, tmp_path):
    source = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(model_source=source, hf_checkpoint=source.uri)
    dest = tmp_path / "model"
    dest.mkdir()
    full_marker = tmp_path / "model.done"
    weight = dest / "model.safetensors"

    def resolve_full(uri, obj):
        assert uri == source.uri and obj is args
        weight.write_bytes(b"weights")
        full_marker.write_text(source.uri)
        return str(dest)

    monkeypatch.setattr(m, "maybe_resolve_s3_model_to_shm", resolve_full)
    local_model = m.prepare_model_maybe_update_args(args)

    assert local_model.completeness == "full"
    assert full_marker.exists()
    assert weight.read_bytes() == b"weights"


def test_prepare_metadata_then_full_only_rewrites_load_for_full(monkeypatch):
    source = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(
        model_source=source,
        _model_source_original_hf_checkpoint="/models/original",
        hf_checkpoint=source.uri,
        tokenizer_model="/models/original",
        load="/models/original",
        ref_load="/models/original",
    )
    calls = []

    def resolve_metadata(uri, _args):
        calls.append(("metadata", uri))
        return "/dev/shm/model"

    def resolve_full(uri, _args):
        calls.append(("full", uri))
        return "/dev/shm/model"

    monkeypatch.setattr(m, "resolve_s3_model_metadata_to_shm", resolve_metadata)
    monkeypatch.setattr(m, "maybe_resolve_s3_model_to_shm", resolve_full)

    m.prepare_model_maybe_update_args(args, completeness="metadata")
    assert args.hf_checkpoint == "/dev/shm/model"
    assert args.tokenizer_model == "/dev/shm/model"
    assert args.load == "/models/original"
    assert args.ref_load == "/models/original"

    m.prepare_model_maybe_update_args(args, completeness="full")
    assert args.load == "/models/original"
    assert args.ref_load == "/dev/shm/model"
    assert calls == [("metadata", source.uri), ("full", source.uri)]


def test_make_s3_client_uses_explicit_placeholder_credentials(monkeypatch):
    import boto3

    captured = {}
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: captured.update(kwargs) or object())

    m._make_s3_client(endpoint="http://s3.example", use_placeholder_credentials=True)

    assert captured["aws_access_key_id"] == "mock"
    assert captured["aws_secret_access_key"] == "mock"


def test_make_s3_client_preserves_default_credentials_for_generic_s3(monkeypatch):
    import boto3

    captured = {}
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: captured.update(kwargs) or object())

    m._make_s3_client(endpoint="http://generic-s3.example")

    assert "aws_access_key_id" not in captured
    assert "aws_secret_access_key" not in captured
    assert captured["config"].retries == {"max_attempts": 10, "mode": "standard"}


def test_make_s3_client_applies_bounded_request_policy(monkeypatch):
    import boto3

    captured = {}
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: captured.update(kwargs) or object())

    m._make_s3_client(
        endpoint="http://s3.example",
        connect_timeout=2.0,
        read_timeout=3.0,
        total_max_attempts=1,
    )

    config = captured["config"]
    assert config.connect_timeout == 2.0
    assert config.read_timeout == 3.0
    assert config.retries == {"total_max_attempts": 1, "mode": "standard"}


def _fake_s3(objects):
    # objects: {key: bytes}
    from unittest.mock import MagicMock

    cli = MagicMock()
    cli.download_bodies = []
    pages = [{"Contents": [{"Key": k, "Size": len(v)} for k, v in objects.items()]}]
    cli.get_paginator.return_value.paginate.return_value = pages

    def head(Bucket, Key):
        return {"ContentLength": len(objects[Key])}

    def get(Bucket, Key):
        body = MagicMock()
        body.iter_chunks.return_value = [objects[Key]]
        cli.download_bodies.append(body)
        return {"Body": body, "ContentLength": len(objects[Key])}

    cli.head_object.side_effect = head
    cli.get_object.side_effect = get
    return cli


def test_download_prefix_writes_files(tmp_path, monkeypatch):
    objs = {"pfx/config.json": b"{}", "pfx/model.safetensors": b"weightbytes"}
    cli = _fake_s3(objs)
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: cli)
    dest = str(tmp_path / "d")
    m._download_prefix("bkt", "pfx/", dest, endpoint=None, workers=2, retries=1)
    assert (tmp_path / "d" / "config.json").read_bytes() == b"{}"
    assert (tmp_path / "d" / "model.safetensors").read_bytes() == b"weightbytes"
    assert len(cli.download_bodies) == 2
    for body in cli.download_bodies:
        body.close.assert_called_once_with()


def test_download_prefix_skips_existing_same_size(tmp_path, monkeypatch):
    objs = {"pfx/a.bin": b"1234"}
    cli = _fake_s3(objs)
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: cli)
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / "a.bin").write_bytes(b"1234")  # 预置与远端 size 一致的文件
    m._download_prefix("bkt", "pfx/", str(dest), endpoint=None, workers=1, retries=1)
    assert (dest / "a.bin").read_bytes() == b"1234"  # 未被覆盖
    # size 一致 → 真正跳过：不能对该 key 拉正文
    cli.get_object.assert_not_called()
    for call in cli.get_object.call_args_list:
        assert call.kwargs.get("Key") != "pfx/a.bin"


def test_download_prefix_empty_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: _fake_s3({}))
    dest = str(tmp_path / "d")
    with pytest.raises(RuntimeError):
        m._download_prefix("bkt", "pfx/", dest, endpoint=None, workers=1, retries=1)


def test_download_model_metadata_to_shm_once_skips_weights(tmp_path, monkeypatch):
    objects = {
        "pfx/config.json": b"{}",
        "pfx/tokenizer.json": b"tokenizer",
        "pfx/model.safetensors.index.json": b'{"weight_map":{"layer":"custom-shard.data"}}',
        "pfx/model-00001-of-00001.safetensors": b"weights",
        "pfx/pytorch_model.bin": b"weights",
        "pfx/custom-shard.data": b"weights",
        "pfx/export.onnx": b"weights",
        "pfx/optimizer.distcp": b"optimizer",
        "pfx/training_state.pt": b"state",
        "pfx/processor.bin": b"processor",
        "pfx/audio_processor.npy": b"audio",
    }
    client = _fake_s3(objects)
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: client)
    dest = str(tmp_path / "model")

    m._download_model_metadata_to_shm_once("s3://bkt/pfx/", dest, endpoint="http://s3.example", workers=2, retries=0)

    assert (tmp_path / "model" / "config.json").is_file()
    assert (tmp_path / "model" / "tokenizer.json").is_file()
    assert (tmp_path / "model" / "model.safetensors.index.json").is_file()
    assert not (tmp_path / "model" / "model-00001-of-00001.safetensors").exists()
    assert not (tmp_path / "model" / "pytorch_model.bin").exists()
    assert not (tmp_path / "model" / "custom-shard.data").exists()
    assert not (tmp_path / "model" / "export.onnx").exists()
    assert not (tmp_path / "model" / "optimizer.distcp").exists()
    assert not (tmp_path / "model" / "training_state.pt").exists()
    assert (tmp_path / "model" / "processor.bin").is_file()
    assert (tmp_path / "model" / "audio_processor.npy").is_file()
    assert (tmp_path / "model.metadata.done").read_text() == "s3://bkt/pfx/\nendpoint=http://s3.example"
    manifest = m._read_model_manifest(dest, "s3://bkt/pfx/\nendpoint=http://s3.example")
    assert manifest is not None
    assert m._manifest_files_complete(dest, manifest, kind="metadata")


def test_download_model_metadata_to_shm_once_checks_available_capacity(tmp_path, monkeypatch):
    client = _fake_list_cli({"pfx/config.json": 100})
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: client)
    monkeypatch.setattr(m, "_free_bytes", lambda path: 10)

    with pytest.raises(RuntimeError, match="shm capacity is insufficient for model metadata"):
        m._download_model_metadata_to_shm_once(
            "s3://bkt/pfx/", str(tmp_path / "model"), endpoint=None, workers=1, retries=0
        )
    _assert_cache_payload_removed(str(tmp_path / "model"))


def test_download_model_metadata_to_shm_once_cleans_failed_download(tmp_path, monkeypatch):
    dest = str(tmp_path / "model")
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: _fake_list_cli({"pfx/config.json": 2}))

    def fail_metadata_download(_cli, _bucket, _keys, _prefix, staging, **kwargs):
        if kwargs["description"] == "metadata":
            (Path(staging) / "partial").write_bytes(b"x")
            raise RuntimeError("injected metadata failure")

    monkeypatch.setattr(m, "_download_selected_objects", fail_metadata_download)

    with pytest.raises(RuntimeError, match="injected metadata failure"):
        m._download_model_metadata_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)
    _assert_cache_payload_removed(dest)


def test_download_model_metadata_to_shm_once_rolls_back_marker_failure(tmp_path, monkeypatch):
    dest = str(tmp_path / "model")
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: _fake_s3({"pfx/config.json": b"{}"}))

    def fail_marker(path, _identity):
        Path(path + ".tmp").write_text("partial marker")
        raise OSError("injected marker failure")

    monkeypatch.setattr(m, "_write_ready_marker", fail_marker)

    with pytest.raises(OSError, match="injected marker failure"):
        m._download_model_metadata_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)
    _assert_cache_payload_removed(dest)


def test_download_model_metadata_to_shm_once_reloads_incomplete_cache(tmp_path, monkeypatch):
    dest = str(tmp_path / "model")
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: _fake_s3({"pfx/config.json": b"{}"}))

    m._download_model_metadata_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)
    os.remove(os.path.join(dest, "config.json"))
    m._download_model_metadata_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)

    assert Path(dest, "config.json").read_bytes() == b"{}"


def test_download_model_metadata_to_shm_once_rejects_manifest_without_metadata(tmp_path, monkeypatch):
    dest = str(tmp_path / "model")
    os.makedirs(dest)
    Path(dest, "model.safetensors").write_bytes(b"weights")
    m._write_model_manifest(
        dest,
        "s3://bkt/pfx/",
        [("pfx/model.safetensors", len(b"weights"))],
        "pfx/",
    )
    Path(dest + ".metadata.done").write_text("s3://bkt/pfx/")
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: _fake_s3({}))

    with pytest.raises(RuntimeError, match="no objects under"):
        m._download_model_metadata_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)
    _assert_cache_payload_removed(dest)


def test_download_prefix_size_mismatch_raises(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    # head 报 4 字节，get 只回 2 字节 → _download_one 抛 IOError，重试耗尽后 RuntimeError
    cli = MagicMock()
    pages = [{"Contents": [{"Key": "pfx/a.bin", "Size": 4}]}]
    cli.get_paginator.return_value.paginate.return_value = pages
    cli.head_object.side_effect = lambda Bucket, Key: {"ContentLength": 4}

    def get(Bucket, Key):
        body = MagicMock()
        body.iter_chunks.return_value = [b"ab"]  # 只有 2 字节，与 ContentLength=4 不符
        return {"Body": body}

    cli.get_object.side_effect = get
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: cli)
    dest = str(tmp_path / "d")
    with pytest.raises(RuntimeError):
        m._download_prefix("bkt", "pfx/", dest, endpoint=None, workers=1, retries=0)

    # 直接调 _download_one 证明走的是诊断性 IOError（message 含实际/期望字节），
    # 而不是 remove 后再 getsize 抛出的 FileNotFoundError（假阳性）。
    with pytest.raises(IOError) as ei:
        m._download_one(cli, "bkt", "pfx/a.bin", "pfx/", str(tmp_path / "d2"))
    assert not isinstance(ei.value, FileNotFoundError)
    msg = str(ei.value)
    assert "size mismatch" in msg
    assert "2" in msg and "4" in msg  # actual=2, expected=4


def test_download_one_removes_part_on_stream_error(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    # get_object 返回一个 Body，其 iter_chunks 先吐一段再抛异常 → 模拟流中断
    cli = MagicMock()
    cli.head_object.side_effect = lambda Bucket, Key: {"ContentLength": 10}
    body = MagicMock()

    def broken_iter(chunk_size):
        yield b"half"

        raise ConnectionError("stream broke mid-download")

    def get(Bucket, Key):
        body.iter_chunks.side_effect = broken_iter
        return {"Body": body}

    cli.get_object.side_effect = get

    dest = str(tmp_path / "d")
    local = os.path.join(dest, "a.bin")
    with pytest.raises(ConnectionError):
        m._download_one(cli, "bkt", "pfx/a.bin", "pfx/", dest)

    # 半截 .part 必须已被清理，成品也不应存在
    assert not os.path.exists(local + ".part")
    assert not os.path.exists(local)
    body.close.assert_called_once_with()


def test_download_one_removes_part_on_size_mismatch(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    # head=4 但 body 只回 2 字节 → size mismatch，.part 需清理且抛诊断 IOError
    cli = MagicMock()
    cli.head_object.side_effect = lambda Bucket, Key: {"ContentLength": 4}

    def get(Bucket, Key):
        body = MagicMock()
        body.iter_chunks.return_value = [b"ab"]
        return {"Body": body}

    cli.get_object.side_effect = get

    dest = str(tmp_path / "d")
    local = os.path.join(dest, "a.bin")
    with pytest.raises(IOError) as ei:
        m._download_one(cli, "bkt", "pfx/a.bin", "pfx/", dest)
    assert not isinstance(ei.value, FileNotFoundError)
    assert "size mismatch" in str(ei.value)
    assert not os.path.exists(local + ".part")


def test_download_model_to_shm_once_idempotent(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_dl(*a, **k):
        calls["n"] += 1
        os.makedirs(a[2], exist_ok=True)
        open(os.path.join(a[2], "a.bin"), "wb").write(b"xx")

    # _download_model_to_shm_once 锁内预检需要 list 一次远端；mock client + 充足 free 让预检通过
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: _fake_list_cli({"pfx/a.bin": 2}))
    monkeypatch.setattr(m, "_free_bytes", lambda p: 10**12)
    monkeypatch.setattr(m, "_download_prefix", fake_dl)
    dest = str(tmp_path / "relax_model_x")
    m._download_model_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=1)
    m._download_model_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=1)
    assert calls["n"] == 1  # 第二次命中 marker 跳过
    with open(dest + ".done") as f:
        assert f.read().strip() == "s3://bkt/pfx/"  # marker 内容==uri


def test_maybe_resolve_s3_model_to_shm_end_to_end(tmp_path, monkeypatch):
    # _download_model_to_shm_once 已含锁内预检 + 下载；这里整体 mock 掉，只验证 resolve 的编排与返回值
    monkeypatch.setattr(m, "_download_model_to_shm_once", lambda uri, dest, **k: os.makedirs(dest, exist_ok=True))

    class A:
        model_source = m.ModelSource("s3://bkt/pfx/", "http://s3.example")
        s3_model_shm_root = str(tmp_path)
        s3_model_download_workers = 4

    out = m.maybe_resolve_s3_model_to_shm("s3://bkt/pfx/", A())
    assert out.startswith(str(tmp_path)) and os.path.isdir(out)


def _fake_list_cli(sizes, *, respect_prefix=False):
    """只提供 list_objects_v2 分页的 fake client。

    sizes: {key: size_int}。respect_prefix=True 时按 Prefix 过滤（用于验证归一）。
    """
    from unittest.mock import MagicMock

    cli = MagicMock()

    def paginate(Bucket, Prefix):
        items = sizes.items()
        if respect_prefix:
            items = [(k, s) for k, s in items if k.startswith(Prefix)]
        return [{"Contents": [{"Key": k, "Size": s} for k, s in items]}]

    cli.get_paginator.return_value.paginate.side_effect = paginate
    return cli


def test_resolve_capacity_precheck(tmp_path, monkeypatch):
    # 远端一个超大对象、free 很小 → _download_model_to_shm_once 锁内预检应抛 RuntimeError
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: _fake_list_cli({"pfx/big.bin": 10**15}))
    monkeypatch.setattr(m, "_free_bytes", lambda p: 10**9)  # 1GB

    class A:
        model_source = m.ModelSource("s3://bkt/pfx/", "http://s3.example")
        s3_model_shm_root = str(tmp_path)
        s3_model_download_workers = 4

    with pytest.raises(RuntimeError, match="[Ii]nsufficient SHM capacity"):
        m.maybe_resolve_s3_model_to_shm("s3://bkt/pfx/", A())


def test_download_model_to_shm_once_marker_hit_skips_precheck(tmp_path, monkeypatch):
    # marker + complete manifest hit skips the remote precheck and download.
    dest = str(tmp_path / "relax_model_x")
    os.makedirs(dest, exist_ok=True)
    (tmp_path / "relax_model_x" / "config.json").write_bytes(b"{}")
    m._write_model_manifest(dest, "s3://bkt/pfx/", [("pfx/config.json", 2)], "pfx/")
    with open(dest + ".done", "w") as f:
        f.write("s3://bkt/pfx/")

    called = {"client": 0, "dl": 0}
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: called.__setitem__("client", called["client"] + 1))
    monkeypatch.setattr(m, "_download_prefix", lambda *a, **k: called.__setitem__("dl", called["dl"] + 1))
    monkeypatch.setattr(m, "_free_bytes", lambda p: 1)  # 1 字节，若跑预检必拒

    m._download_model_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)
    assert called == {"client": 0, "dl": 0}


def test_download_model_to_shm_once_replaces_legacy_cache_atomically(tmp_path, monkeypatch):
    objects = {"pfx/config.json": b"{}", "pfx/model.safetensors": b"weights"}
    dest = str(tmp_path / "relax_model_legacy")
    os.makedirs(dest, exist_ok=True)
    for key, body in objects.items():
        (tmp_path / "relax_model_legacy" / key[len("pfx/") :]).write_bytes(body)
    with open(dest + ".done", "w") as marker_file:
        marker_file.write("s3://bkt/pfx/")

    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: _fake_s3(objects))

    def download(_bucket, _prefix, staging, **_kwargs):
        os.makedirs(staging, exist_ok=True)
        for key, body in objects.items():
            (Path(staging) / key[len("pfx/") :]).write_bytes(body)

    monkeypatch.setattr(m, "_download_prefix", download)

    m._download_model_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)

    manifest = m._read_model_manifest(dest, "s3://bkt/pfx/")
    assert manifest is not None
    assert {entry["path"] for entry in manifest["files"]} == {"config.json", "model.safetensors"}


def test_download_model_to_shm_once_rolls_back_manifest_publish_failure(tmp_path, monkeypatch):
    dest = str(tmp_path / "model")
    objects = {"pfx/config.json": b"{}"}
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: _fake_s3(objects))

    def download(_bucket, _prefix, staging, **_kwargs):
        os.makedirs(staging, exist_ok=True)
        Path(staging, "config.json").write_bytes(b"{}")

    original_replace = m.os.replace

    def fail_manifest_publish(src, dst):
        if src == dest + ".tmp.manifest.json" and dst == dest + ".manifest.json":
            raise OSError("injected manifest publish failure")
        return original_replace(src, dst)

    monkeypatch.setattr(m, "_download_prefix", download)
    monkeypatch.setattr(m.os, "replace", fail_manifest_publish)

    with pytest.raises(OSError, match="injected manifest publish failure"):
        m._download_model_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)
    _assert_cache_payload_removed(dest)


def test_download_model_to_shm_once_rolls_back_marker_failure(tmp_path, monkeypatch):
    dest = str(tmp_path / "model")
    objects = {"pfx/config.json": b"{}"}
    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: _fake_s3(objects))

    def download(_bucket, _prefix, staging, **_kwargs):
        os.makedirs(staging, exist_ok=True)
        Path(staging, "config.json").write_bytes(b"{}")

    def fail_marker(path, _identity):
        Path(path + ".tmp").write_text("partial marker")
        raise OSError("injected marker failure")

    monkeypatch.setattr(m, "_download_prefix", download)
    monkeypatch.setattr(m, "_write_ready_marker", fail_marker)

    with pytest.raises(OSError, match="injected marker failure"):
        m._download_model_to_shm_once("s3://bkt/pfx/", dest, endpoint=None, workers=1, retries=0)
    _assert_cache_payload_removed(dest)


def test_download_model_to_shm_once_precheck_does_not_resume_partial_cache(tmp_path, monkeypatch):
    dest = tmp_path / "relax_model_y"
    dest.mkdir()
    (dest / "a.bin").write_bytes(b"x" * 100)  # 预置与远端 size 一致

    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: _fake_list_cli({"pfx/a.bin": 100, "pfx/b.bin": 100}))
    monkeypatch.setattr(m, "_free_bytes", lambda p: 150)

    dl = {"n": 0}
    monkeypatch.setattr(m, "_download_prefix", lambda *a, **k: dl.__setitem__("n", dl["n"] + 1))

    with pytest.raises(RuntimeError, match="Insufficient SHM capacity"):
        m._download_model_to_shm_once("s3://bkt/pfx/", str(dest), endpoint=None, workers=1, retries=0)
    assert dl["n"] == 0


def test_download_model_to_shm_once_precheck_normalizes_prefix(tmp_path, monkeypatch):
    # uri 无尾斜杠：list/容量口径必须只算 pfx/ 下对象，不含兄弟前缀 pfx-v2/
    cli = _fake_list_cli({"pfx/a.bin": 10, "pfx-v2/huge.bin": 10**15}, respect_prefix=True)
    monkeypatch.setattr(m, "_make_s3_client", lambda **kw: cli)
    monkeypatch.setattr(m, "_free_bytes", lambda p: 100)  # 兄弟前缀若算入则必拒
    dl = {"n": 0}

    def download(_bucket, _prefix, staging, **_kwargs):
        dl["n"] += 1
        os.makedirs(staging, exist_ok=True)
        Path(staging, "a.bin").write_bytes(b"x" * 10)

    monkeypatch.setattr(m, "_download_prefix", download)

    dest = str(tmp_path / "relax_model_z")
    m._download_model_to_shm_once("s3://bkt/pfx", dest, endpoint=None, workers=1, retries=0)  # 注意 uri 无尾 /
    assert dl["n"] == 1  # 归一为 pfx/ 后 need=10 <= free=100，通过
    # list 用的是归一后的 pfx/（否则 "pfx" 会连带 pfx-v2/ 被算进来）
    cli.get_paginator.return_value.paginate.assert_called_once_with(Bucket="bkt", Prefix="pfx/")


def test_resolve_shm_root_requires_existing_directory(tmp_path):
    class A:
        s3_model_shm_root = str(tmp_path)

    assert m._resolve_shm_root(A()) == str(tmp_path)

    class B:
        s3_model_shm_root = str(tmp_path / "does_not_exist")

    with pytest.raises(RuntimeError, match="disk fallback is intentionally disabled"):
        m._resolve_shm_root(B())

    file_path = tmp_path / "regular-file"
    file_path.write_text("not a directory")

    class FileRoot:
        s3_model_shm_root = str(file_path)

    with pytest.raises(RuntimeError, match=str(file_path)):
        m._resolve_shm_root(FileRoot())

    class C:
        pass  # 无属性 → 默认 /dev/shm

    assert m._resolve_shm_root(C()) == "/dev/shm"


def test_get_cached_path_returns_none_when_shm_root_is_missing(tmp_path):
    config = m.ModelSource("s3://bucket/model/", "http://s3.example")
    args = SimpleNamespace(model_source=config, s3_model_shm_root=str(tmp_path / "missing"))

    assert m.get_s3_model_cached_path(config.uri, args) is None


def test_get_cached_path_requires_complete_manifest(tmp_path):
    config = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(model_source=config, s3_model_shm_root=str(tmp_path))
    dest = m._shm_dest_dir(config.uri, str(tmp_path), config.endpoint)
    os.makedirs(dest)
    Path(dest, "config.json").write_bytes(b"{}")
    Path(dest + ".done").write_text(config.uri)

    assert m.get_s3_model_cached_path(config.uri, args) is None

    m._write_model_manifest(dest, config.uri, [("model/config.json", 2)], "model/")
    assert m.get_s3_model_cached_path(config.uri, args) == dest

    Path(dest, "config.json").unlink()
    assert m.get_s3_model_cached_path(config.uri, args) is None


def test_cleanup_s3_model_weights_preserves_metadata_and_invalidates_full_cache(tmp_path, monkeypatch):
    config = m.ModelSource("s3://bucket/model/", "http://s3.example")
    args = SimpleNamespace(model_source=config, s3_model_shm_root=str(tmp_path))
    dest = m._shm_dest_dir(config.uri, str(tmp_path), config.endpoint)
    os.makedirs(os.path.join(dest, "subdir"))
    identity = m._cache_identity(config.uri, config.endpoint)
    (tmp_path / (os.path.basename(dest) + ".done")).write_text(identity)
    (tmp_path / (os.path.basename(dest) + ".lock")).touch()
    (tmp_path / os.path.basename(dest) / "config.json").write_text("{}")
    (tmp_path / os.path.basename(dest) / "tokenizer.model").write_bytes(b"tokenizer")
    (tmp_path / os.path.basename(dest) / "processor.bin").write_bytes(b"processor")
    (tmp_path / os.path.basename(dest) / "model-00001-of-00002.safetensors").write_bytes(b"1234")
    (tmp_path / os.path.basename(dest) / "subdir" / "pytorch_model.bin").write_bytes(b"123")
    m._write_model_manifest(
        dest,
        identity,
        [
            ("model/config.json", 2),
            ("model/tokenizer.model", 9),
            ("model/processor.bin", 9),
            ("model/model-00001-of-00002.safetensors", 4),
            ("model/subdir/pytorch_model.bin", 3),
        ],
        "model/",
    )

    assert m.cleanup_s3_model_weights_from_shm(args) == (2, 7)
    assert not os.path.exists(dest + ".done")
    assert open(dest + ".metadata.done").read() == identity
    assert os.path.exists(os.path.join(dest, "config.json"))
    assert os.path.exists(os.path.join(dest, "tokenizer.model"))
    assert os.path.exists(os.path.join(dest, "processor.bin"))
    assert not os.path.exists(os.path.join(dest, "model-00001-of-00002.safetensors"))
    assert not os.path.exists(os.path.join(dest, "subdir", "pytorch_model.bin"))
    assert m.get_s3_model_cached_path(config.uri, args) is None

    monkeypatch.setattr(m, "_make_s3_client", lambda **kwargs: pytest.fail("metadata cache must be reusable"))
    m._download_model_metadata_to_shm_once(
        config.uri,
        dest,
        endpoint=config.endpoint,
        workers=1,
        retries=0,
    )


def test_cleanup_s3_model_weights_finishes_interrupted_cleanup(tmp_path):
    config = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(model_source=config, s3_model_shm_root=str(tmp_path))
    dest = m._shm_dest_dir(config.uri, str(tmp_path), config.endpoint)
    os.makedirs(dest)
    with open(dest + ".metadata.done", "w") as marker:
        marker.write(m._cache_identity(config.uri, config.endpoint))
    weight = os.path.join(dest, "model.safetensors")
    with open(weight, "wb") as shard:
        shard.write(b"12345")
    m._write_model_manifest(
        dest,
        m._cache_identity(config.uri, config.endpoint),
        [("model/model.safetensors", 5)],
        "model/",
    )

    assert m.cleanup_s3_model_weights_from_shm(args) == (1, 5)
    assert m.cleanup_s3_model_weights_from_shm(args) == (0, 0)


def test_cleanup_s3_model_weights_respects_disable_flag(tmp_path):
    config = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(
        model_source=config,
        s3_model_shm_root=str(tmp_path),
        disable_s3_model_cleanup=True,
    )
    dest = m._shm_dest_dir(config.uri, str(tmp_path), config.endpoint)
    os.makedirs(dest)
    with open(dest + ".done", "w") as marker:
        marker.write(m._cache_identity(config.uri, config.endpoint))
    weight = os.path.join(dest, "model.safetensors")
    with open(weight, "wb") as shard:
        shard.write(b"12345")

    assert m.cleanup_s3_model_weights_from_shm(args) == (0, 0)
    assert os.path.exists(weight)
    assert os.path.exists(dest + ".done")


def test_cleanup_s3_model_weights_skips_cache_without_ready_marker(tmp_path):
    config = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(model_source=config, s3_model_shm_root=str(tmp_path))
    dest = m._shm_dest_dir(config.uri, str(tmp_path), config.endpoint)
    os.makedirs(dest)
    weight = os.path.join(dest, "model.safetensors")
    with open(weight, "wb") as shard:
        shard.write(b"12345")

    assert m.cleanup_s3_model_weights_from_shm(args) == (0, 0)
    assert os.path.exists(weight)


def test_cleanup_s3_model_weights_skips_invalid_manifest_entry(tmp_path):
    config = m.ModelSource("s3://bucket/model/")
    args = SimpleNamespace(model_source=config, s3_model_shm_root=str(tmp_path))
    dest = m._shm_dest_dir(config.uri, str(tmp_path), config.endpoint)
    os.makedirs(dest)
    identity = m._cache_identity(config.uri, config.endpoint)
    with open(dest + ".done", "w") as marker:
        marker.write(identity)
    with open(dest + ".manifest.json", "w") as manifest:
        manifest.write('{"version": 1, "identity": "s3://bucket/model/", "files": [null]}')

    assert m.cleanup_s3_model_weights_from_shm(args) == (0, 0)
    assert os.path.exists(dest + ".done")


def test_acquire_cleanup_lock_times_out(monkeypatch, tmp_path):
    monotonic_values = iter((0.0, 1.0))
    monkeypatch.setattr(m.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(m, "_CLEANUP_LOCK_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(m.fcntl, "flock", lambda *_args: (_ for _ in ()).throw(BlockingIOError()))

    with open(tmp_path / "model.lock", "w") as lock_file:
        with pytest.raises(TimeoutError, match="SHM cache lock"):
            m._acquire_cleanup_lock(lock_file)


def test_model_manifest_uses_weight_map_and_preserves_unknown_binary_assets(tmp_path):
    dest = str(tmp_path / "model")
    os.makedirs(dest)
    (tmp_path / "model" / "model.safetensors.index.json").write_text('{"weight_map": {"layer": "custom-shard.data"}}')

    m._write_model_manifest(
        dest,
        "identity",
        [
            ("pfx/model.safetensors.index.json", 47),
            ("pfx/custom-shard.data", 10),
            ("pfx/processor.bin", 5),
        ],
        "pfx/",
    )

    manifest = m._read_model_manifest(dest, "identity")
    kinds = {entry["path"]: entry["kind"] for entry in manifest["files"]}
    assert kinds["custom-shard.data"] == "weight"
    assert kinds["processor.bin"] == "metadata"


def test_model_weight_path_only_matches_standard_names_or_index_entries():
    indexed_weights = {"weights/custom.data"}

    assert m._is_model_weight_path("weights/custom.data", indexed_weights)
    assert m._is_model_weight_path("pytorch_model-00001-of-00002.bin", set())
    assert m._is_model_weight_path("consolidated.00.pth", set())
    assert m._is_model_weight_path("model.safetensors", set())
    assert m._is_model_weight_path("export.onnx", set())
    assert m._is_model_weight_path("optimizer.distcp", set())
    assert m._is_model_weight_path("training_state.pt", set())
    assert not m._is_model_weight_path("model_args.bin", set())
    assert not m._is_model_weight_path("processor.pt", set())
    assert not m._is_model_weight_path("audio_processor.npy", set())


def test_resolve_endpoint_comes_from_platform_config():
    class A:
        model_source = m.ModelSource("s3://bucket/model/", "http://from-platform")

    assert m._resolve_endpoint(A()) == "http://from-platform"


def test_resolve_endpoint_uses_standard_aws_environment(monkeypatch):
    class A:
        model_source = m.ModelSource("s3://bucket/model/")

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://global.example")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://s3.example")

    assert m._resolve_endpoint(A()) == "http://s3.example"


def test_resolve_endpoint_falls_back_to_global_aws_environment(monkeypatch):
    class A:
        model_source = m.ModelSource("s3://bucket/model/")

    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://global.example")

    assert m._resolve_endpoint(A()) == "http://global.example"


def test_remove_stale_s3_model_caches_only_removes_relax_namespace(tmp_path):
    cache = tmp_path / "relax_model_0123456789abcdef"
    cache.mkdir()
    (cache / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "relax_model_0123456789abcdef.done").write_text("ready")
    (tmp_path / "relax_model_0123456789abcdef.state.json").write_text("state")
    (tmp_path / "relax_model_0123456789abcdef.state.json.tmp").write_text("temporary state")
    (tmp_path / "relax_model_0123456789abcdef.tmp.manifest.json").write_text("staging manifest")
    (tmp_path / "relax_model_0123456789abcdef.tmp.manifest.json.tmp").write_text("temporary manifest")
    unrelated = tmp_path / "another_model_cache"
    unrelated.mkdir()
    (unrelated / "keep.bin").write_bytes(b"keep")
    similar = tmp_path / "relax_model_not-a-digest"
    similar.mkdir()

    args = SimpleNamespace(s3_model_shm_root=str(tmp_path))
    removed_entries, removed_bytes = m.remove_stale_s3_model_caches(args)

    assert removed_entries == 6
    assert removed_bytes == (
        len(b"weights")
        + len("ready")
        + len("state")
        + len("temporary state")
        + len("staging manifest")
        + len("temporary manifest")
    )
    assert not cache.exists()
    assert not (tmp_path / "relax_model_0123456789abcdef.done").exists()
    assert not (tmp_path / "relax_model_0123456789abcdef.state.json").exists()
    assert not (tmp_path / "relax_model_0123456789abcdef.state.json.tmp").exists()
    assert not (tmp_path / "relax_model_0123456789abcdef.tmp.manifest.json").exists()
    assert not (tmp_path / "relax_model_0123456789abcdef.tmp.manifest.json.tmp").exists()
    assert (tmp_path / "relax_model_0123456789abcdef.lock").is_file()
    assert (unrelated / "keep.bin").read_bytes() == b"keep"
    assert similar.is_dir()


def test_remove_stale_s3_model_caches_missing_root_is_noop(tmp_path):
    args = SimpleNamespace(s3_model_shm_root=str(tmp_path / "missing"))

    assert m.remove_stale_s3_model_caches(args) == (0, 0)


def test_remove_stale_s3_model_caches_rescans_after_acquiring_existing_lock(tmp_path, monkeypatch):
    base = "relax_model_0123456789abcdef"
    (tmp_path / f"{base}.lock").touch()
    original_acquire = m._acquire_cleanup_lock

    def acquire_then_publish(lock_file):
        original_acquire(lock_file)
        cache = tmp_path / base
        cache.mkdir()
        (cache / "late.safetensors").write_bytes(b"late")
        (tmp_path / f"{base}.done").write_text("ready")

    monkeypatch.setattr(m, "_acquire_cleanup_lock", acquire_then_publish)

    assert m.remove_stale_s3_model_caches(SimpleNamespace(s3_model_shm_root=str(tmp_path))) == (
        2,
        len(b"late") + len("ready"),
    )
    assert not (tmp_path / base).exists()
    assert not (tmp_path / f"{base}.done").exists()


def test_build_runai_streamer_env_for_load_is_early_copy_overlay():
    source = m.ModelSource(
        "s3://bucket/model/",
        "http://provider.example",
        credential_mode="placeholder",
        addressing_style="path",
    )
    current = {"AWS_ACCESS_KEY_ID": "original"}

    overlay = m.build_runai_streamer_env_for_load(source, source.uri, "runai_streamer", current)

    assert current == {"AWS_ACCESS_KEY_ID": "original"}
    assert overlay == {
        "RUNAI_STREAMER_S3_ENDPOINT": "http://provider.example",
        "AWS_ENDPOINT_URL": "http://provider.example",
        "AWS_ENDPOINT_URL_S3": "http://provider.example",
        "RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING": "0",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_ACCESS_KEY_ID": "mock",
        "AWS_SECRET_ACCESS_KEY": "mock",
        "AWS_SESSION_TOKEN": "",
    }


@pytest.mark.parametrize("load_format", ["dummy", "remote", "full"])
def test_build_runai_streamer_env_for_load_ignores_unsupported_formats(load_format):
    source = m.ModelSource("s3://bucket/model/", "http://provider.example")

    assert m.build_runai_streamer_env_for_load(source, source.uri, load_format, {}) == {}
