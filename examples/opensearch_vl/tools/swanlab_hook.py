# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Enable SwanLab where the Ray metrics replica creates its TensorBoard
writer."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def privacy_settings():
    import swanlab

    return swanlab.Settings(
        interactive=False,
        on_online_error="raise",
        probe=swanlab.Settings.Probe(
            hardware=False,
            runtime=False,
            requirements=False,
            conda=False,
            git=False,
            swanlab=False,
            monitor=False,
        ),
        terminal=swanlab.Settings.Terminal(proxy_type="none"),
    )


def check_receipts(directory: Path, experiment_prefix: str) -> None:
    expected_path = f"/@{os.environ['SWANLAB_WORKSPACE']}/{os.environ['SWANLAB_PROJ_NAME']}/runs/"
    for path in directory.glob("*.json"):
        if not path.name.startswith(experiment_prefix):
            continue
        url = urlsplit(json.loads(path.read_text())["url"])
        if (
            url.scheme in ("https", "http")
            and url.netloc
            and url.path.startswith(expected_path)
            and len(url.path) > len(expected_path)
        ):
            return
    raise RuntimeError(f"SwanLab cloud training receipt missing for {experiment_prefix}")


def install() -> None:
    from relax.utils.logging_utils import install_asyncio_noise_filter

    install_asyncio_noise_filter()
    import swanlab
    from torch.utils.tensorboard import SummaryWriter

    swanlab.merge_settings(privacy_settings())
    swanlab.sync_tensorboard_torch(types=["scalar"])
    original_init = SummaryWriter.__init__

    def initialize(writer, *args, **kwargs):
        original_init(writer, *args, **kwargs)
        swanlab.config.update(
            {
                key: os.environ[key]
                for key in ("RELAX_GIT_REF", "RELAX_GIT_COMMIT", "TRAINING_JOB_ID", "TRAINING_TRIAL_ID")
                if key in os.environ
            }
        )
        directory = Path(os.environ["SWANLAB_LOG_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        url = swanlab.get_run().url
        if not url:
            raise RuntimeError("SwanLab cloud run URL missing")
        (directory / (os.environ["SWANLAB_EXP_NAME"] + ".json")).write_text(json.dumps({"url": url}))

    SummaryWriter.__init__ = initialize


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check-receipts", required=True, help="Training experiment name prefix")
    args = parser.parse_args()
    check_receipts(Path(os.environ["SWANLAB_LOG_DIR"]), args.check_receipts)
