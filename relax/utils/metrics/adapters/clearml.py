# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import subprocess
from pathlib import Path

from relax.utils.env import Envs, is_env_set
from relax.utils.logging_utils import get_logger
from relax.utils.misc import SingletonMeta


logger = get_logger(__name__)


class _ClearMLAdapter(metaclass=SingletonMeta):
    _task = None

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[4]

    @staticmethod
    def _run_git_command(cmds: list[str], directory_path: Path) -> str:
        if not directory_path.is_dir():
            return f"error: directory not found: {directory_path}"

        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=directory_path,
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as exc:
            return f"error: not a git repository at {directory_path}: {exc}"

        try:
            result = subprocess.run(
                cmds,
                cwd=directory_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            return f"error: git command failed: {exc.stderr}"
        except Exception as exc:  # pragma: no cover - defensive logging path
            return f"error: unexpected failure: {exc}"

    def _connect_git_metadata(self, args) -> None:
        repo_root = self._repo_root()
        self._task.set_repo(str(repo_root))
        self._task.set_user_properties(relax_repo_root=str(repo_root))

        self._task.connect_configuration(vars(args), name="Hyperparameters")
        self._task.connect_configuration(
            {
                "repo_root": str(repo_root),
                "config json": self._safe_json(vars(args)),
            },
            name="config",
        )
        self._task.connect_configuration(
            {"git log": self._run_git_command(["git", "--no-pager", "log", "-8"], repo_root)},
            name="git log",
        )
        self._task.connect_configuration(
            {"git status": self._run_git_command(["git", "status"], repo_root)},
            name="git status",
        )

    @staticmethod
    def _safe_json(value: dict) -> str:
        import json

        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:  # pragma: no cover - defensive logging path
            return f"error: failed to serialize config: {exc}"

    def __init__(self, args):
        assert args.use_clearml, f"{args.use_clearml=}"
        self.project_name, self.experiment_name, self.tags = self._get_task_info(args)

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            project_name=self.project_name,
            task_name=self.experiment_name,
            tags=self.tags,
            continue_last_task=False,
            output_uri=False,
            reuse_last_task_id=False,
            auto_connect_frameworks={"tensorboard": False, "pytorch": False},
        )
        self._connect_git_metadata(args)
        logger.info(
            f"Saving clearml log to project: {self.project_name}, experiment: {self.experiment_name}, tags: {self.tags}"
        )

    def _get_task_info(self, args):
        project_name = args.tb_project_name or Envs.CLEARML_PROJECT
        experiment_name = args.tb_experiment_name or Envs.CLEARML_TASK
        tags = Envs.CLEARML_TAGS

        if not is_env_set("CLEARML_TAGS") and (user := Envs.USER) and (region := Envs.REGION):
            tags = f"userid={user},region={region}"

        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        return project_name, experiment_name, tags

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        _logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                _logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                _logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()
