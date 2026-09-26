# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import datetime
import os

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.misc import SingletonMeta


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

__all__ = ["_TensorboardAdapter"]

logger = get_logger(__name__)


class _TensorboardAdapter(metaclass=SingletonMeta):
    _writer = None

    """
    # Usage example: This will return the same instance every rank
    # tb = _TensorboardAdapter(args)  # Initialize on first call
    # tb.log({"Loss": 0.1}, step=1)

    # In other files:
    # from tensorboard_utils import _TensorboardAdapter
    # tb = _TensorboardAdapter(args)  # No parameters needed to get existing instance
    # tb.log({"Accuracy": 0.9}, step=1)
    """

    def __init__(self, args):
        assert args.use_tensorboard, f"{args.use_tensorboard=}"
        tb_project_name = args.tb_project_name
        tb_experiment_name = args.tb_experiment_name
        save_dir = getattr(args, "save", None)
        if tb_project_name is None and not Envs.TENSORBOARD_DIR and not save_dir:
            # Nothing user-supplied to locate the run — fall back to a defaulted
            # project + timestamped experiment so tensorboard just works.
            tb_project_name = "relax"
            logger.warning(
                "No tb_project_name / tb_experiment_name / TENSORBOARD_DIR / args.save supplied; "
                f"defaulting tb_project_name={tb_project_name!r} and auto-generating tb_experiment_name."
            )
        if tb_project_name is not None and tb_experiment_name is None:
            tb_experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._initialize(tb_project_name, tb_experiment_name, save_dir)

    def _initialize(self, tb_project_name, tb_experiment_name, save_dir):
        """Actual initialization logic."""
        # Priority: TENSORBOARD_DIR env > args.save > default project/experiment path
        tensorboard_dir = Envs.TENSORBOARD_DIR
        if not tensorboard_dir:
            if save_dir:
                tensorboard_dir = os.path.join(save_dir, "tensorboard_log")
            else:
                tensorboard_dir = f"tensorboard_log/{tb_project_name}/{tb_experiment_name}"
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Saving tensorboard log to {tensorboard_dir}.")
        self._writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        """Log data to tensorboard.

        Args:
            data (dict): Dictionary containing metric names and values
            step (int): Current step/epoch number
        """
        for key in data:
            self._writer.add_scalar(key, data[key], step)

    def finish(self):
        """Close the tensorboard writer."""
        self._writer.close()
