# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify lazy initialization, cloud receipts and privacy settings passed to
SwanLab; SDK calls are stubbed, so actual uploads are outside this scope."""

import importlib.util
import json
import os
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
