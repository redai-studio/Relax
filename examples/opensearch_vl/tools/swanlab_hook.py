# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Enable SwanLab where the Ray metrics replica creates its TensorBoard
writer."""

import json
import os
from pathlib import Path


def privacy_settings():
    import swanlab

    return swanlab.Settings(
        metadata_collect=False,
        collect_hardware=False,
        collect_runtime=False,
        hardware_monitor=False,
        requirements_collect=False,
        conda_collect=False,
        log_proxy_type="none",
    )


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
        (directory / (os.environ["SWANLAB_EXP_NAME"] + ".json")).write_text(
            json.dumps({"url": swanlab.get_run().public.cloud.experiment_url})
        )

    SummaryWriter.__init__ = initialize
