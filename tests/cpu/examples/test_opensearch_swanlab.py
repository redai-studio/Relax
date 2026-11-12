# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify lazy initialization, cloud receipts and privacy settings passed to
SwanLab; SDK calls are stubbed, so actual uploads are outside this scope."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HOOK = Path(__file__).resolve().parents[3] / "examples/opensearch_vl/tools/swanlab_hook.py"
URL = "https://example.invalid/@redai-studio/Relax-opensearch-vl/runs/run-id"


@pytest.fixture
def hook(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("opensearch_swanlab_hook", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in {
        "SWANLAB_LOG_DIR": tmp_path,
        "SWANLAB_PROJ_NAME": "Relax-opensearch-vl",
        "SWANLAB_WORKSPACE": "redai-studio",
        "SWANLAB_EXP_NAME": "trial-tune-none-relax-timestamp",
        "RELAX_GIT_COMMIT": "commit",
        "UNRELATED_SECRET": "secret-fixture",
    }.items():
        monkeypatch.setenv(key, str(value))
    return module


@pytest.mark.parametrize("failure", [None, "initialization", "offline"])
def test_swanlab_hook_is_lazy_and_receipt_detects_swallowed_failures(hook, monkeypatch, tmp_path, failure):
    settings, configs, runs, scalars = [], [], [], []

    class Settings(SimpleNamespace):
        Probe = Terminal = SimpleNamespace

    class Writer:
        def __init__(self, *args, **kwargs):
            self.ready = True

        def add_scalar(self, tag, value, step):
            scalars.append((tag, value, step))

    class Run:
        @property
        def url(self):
            if failure == "offline":
                raise ValueError("Run url is unavailable in offline mode")
            return URL

    run = Run()

    def sync(types):
        assert types == ["scalar"]
        original = Writer.__init__

        def initialize(writer, *args, **kwargs):
            runs.append(run)
            if failure == "initialization":
                raise RuntimeError("Cloud initialization failed")
            original(writer, *args, **kwargs)

        Writer.__init__ = initialize

    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(
            Settings=Settings,
            merge_settings=settings.append,
            sync_tensorboard_torch=sync,
            config=SimpleNamespace(update=configs.append),
            get_run=lambda: run,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=Writer))
    monkeypatch.setitem(
        sys.modules, "relax.utils.logging_utils", SimpleNamespace(install_asyncio_noise_filter=lambda: None)
    )
    hook.install()
    assert not runs and not list(tmp_path.iterdir())
    assert settings[0].interactive is False and settings[0].on_online_error == "raise"
    assert vars(settings[0].probe) == dict.fromkeys(
        ("hardware", "runtime", "requirements", "conda", "git", "swanlab", "monitor"), False
    )
    assert settings[0].terminal.proxy_type == "none"

    class PlatformWriter(Writer):
        def __init__(self):
            try:
                super().__init__("tensorboard")
            except Exception:
                self.disabled = True

    writer = PlatformWriter()
    assert len(runs) == 1
    if failure:
        assert writer.disabled
        with pytest.raises(RuntimeError, match="cloud training receipt missing"):
            hook.check_receipts(tmp_path, "trial-tune-none-relax-")
        assert not list(tmp_path.glob("*.json"))
    else:
        writer.add_scalar("loss", 1.0, 4)
        assert scalars == [("loss", 1.0, 4)]
        hook.check_receipts(tmp_path, "trial-tune-none-relax-")
        assert configs == [{"RELAX_GIT_COMMIT": "commit"}]
        assert json.loads(next(tmp_path.glob("*.json")).read_text()) == {"url": URL}


@pytest.mark.parametrize(
    "name,url,accepted",
    [
        ("trial-tune-none-relax-timestamp", URL, True),
        ("other-trial-tune-none-relax-timestamp", URL, False),
        ("trial-tune-none-relax-timestamp", "", False),
        ("trial-tune-none-relax-timestamp", URL.replace("redai-studio", "other-workspace"), False),
        ("trial-tune-none-relax-timestamp", URL.replace("Relax-opensearch-vl", "other-project"), False),
        ("trial-tune-none-relax-timestamp", URL.removesuffix("run-id"), False),
    ],
)
def test_receipt_cli_requires_current_training_cloud_run(hook, tmp_path, name, url, accepted):
    (tmp_path / f"{name}.json").write_text(json.dumps({"url": url}))
    result = subprocess.run(
        [sys.executable, str(HOOK), "--check-receipts", "trial-"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is accepted
    assert "secret-fixture" not in result.stdout + result.stderr


def test_training_keeps_swanlab_credentials_out_of_shell_trace(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    script = root / "examples/opensearch_vl/scripts/run-qwen3.5-4B-8xgpu.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(HOOK.parents[3] / "examples/opensearch_vl/scripts" / script.name, script)
    for relative, content in {
        "scripts/entrypoint/local.sh": "set -x\nexport RUNTIME_ENV_JSON='{\"env_vars\":{}}'\n",
        "scripts/models/qwen35-4B.sh": "MODEL_ARGS=()\n",
        "examples/opensearch_vl/tools/audit_data.py": "",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    ray = binary / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\nfrom pathlib import Path\n"
        "value=next(arg.split('=',1)[1] for arg in sys.argv if arg.startswith('--runtime-env-json='))\n"
        "Path(os.environ['RUNTIME_CAPTURE']).write_text(value)\n"
    )
    ray.chmod(0o700)
    captured = tmp_path / "runtime.json"
    secret = "fake-swanlab-credential-for-trace-regression"
    for key, value in {
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "MODEL_DIR": str(model),
        "HF_CHECKPOINT": str(model),
        "DATA_DIR": str(tmp_path / "data"),
        "SAVE_DIR": str(tmp_path / "outputs"),
        "EXP_NAME": "trace-check",
        "MODEL_CONFIG": "qwen35-4B",
        "ENABLE_SWANLAB": "1",
        "SWANLAB_API_KEY": secret,
        "RUNTIME_CAPTURE": str(captured),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("RELAX_ENTRYPOINT_MODE", raising=False)
    monkeypatch.delenv("RESUME_CHECKPOINT", raising=False)
    result = subprocess.run(["bash", str(script), "relax"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout + result.stderr
    runtime = json.loads(captured.read_text())
    assert runtime["env_vars"]["SWANLAB_API_KEY"] == secret
    assert runtime["worker_process_setup_hook"] == "examples.opensearch_vl.tools.swanlab_hook.install"
