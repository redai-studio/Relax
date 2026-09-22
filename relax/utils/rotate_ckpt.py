# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re
import shutil
from argparse import Namespace
from pathlib import Path

from relax.utils.logging_utils import get_logger


_ITER_DIR_PATTERN = re.compile(r"^iter_\d{7}$")


logger = get_logger(__name__)


def rotate_ckpt(
    config: Namespace,
    global_step: int,
    save_dir: str | None = None,
    *,
    committed_only: bool = False,
):
    """Prune old ``iter_*`` checkpoint dirs under ``save_dir`` (default
    ``config.save``).

    ``save_dir`` lets a backend whose layout nests the iteration dirs one level
    deeper pass the real parent. The FSDP generative backend writes
    ``<save>/<task>/iter_NNNNNNN`` (see ``relax/backends/fsdp/checkpoint.py``),
    so it passes the per-task root; without it the glob below would match
    nothing and retention would silently never run.
    """
    if config.max_actor_ckpt_to_keep is None and not config.rotate_ckpt:
        return

    root = save_dir if save_dir is not None else config.save
    ckpt_dirs = list(Path(root).glob("iter_*"))
    if committed_only:
        ckpt_dirs = [ckpt_dir for ckpt_dir in ckpt_dirs if (ckpt_dir / "COMMITTED").is_file()]
    if not ckpt_dirs:
        return

    ckpt_dirs = [
        (int(ckpt_dir.name.split("_")[-1]), ckpt_dir)
        for ckpt_dir in ckpt_dirs
        if ckpt_dir.name.split("_")[-1].isdigit()
    ]
    if not ckpt_dirs:
        return

    ckpt_dirs.sort(key=lambda x: x[0], reverse=True)

    if config.rotate_ckpt:
        _rotate_ckpt_cleanup(config, global_step, ckpt_dirs)
    else:
        _max_keep_cleanup(config, ckpt_dirs, keep_latest=committed_only)


def _rotate_ckpt_cleanup(config: Namespace, global_step: int, ckpt_dirs: list):
    """Cleanup for rotate_ckpt mode: keep latest + up to max_ckpt save_interval
    checkpoints, delete intermediates."""
    # +1 是因为要额外保存 latest，如果 latest 同时满足 save_interval ，下面会减掉
    max_ckpt = global_step // config.save_interval + 1
    if config.max_actor_ckpt_to_keep is not None:
        max_ckpt = min(max_ckpt, config.max_actor_ckpt_to_keep + 1)

    logger.info(f"max checkpoint to keep: {max_ckpt}")

    latest_ckpt = ckpt_dirs.pop(0)
    if latest_ckpt[0] % config.save_interval == 0:
        max_ckpt -= 1

    logger.info(f"latest checkpoint: {latest_ckpt}")
    ckpt_num = 1
    for step, ckpt_dir in ckpt_dirs:
        if step % config.save_interval != 0 or ckpt_num >= max_ckpt:
            _remove_ckpt(ckpt_dir)
        else:
            ckpt_num += 1
            logger.info(f"keep checkpoint dir {ckpt_dir}, current ckpt num: {ckpt_num}")


def _max_keep_cleanup(config: Namespace, ckpt_dirs: list, *, keep_latest: bool = False):
    """Cleanup for non-rotate mode: simply keep the latest
    max_actor_ckpt_to_keep checkpoints."""
    max_keep = max(1, config.max_actor_ckpt_to_keep) if keep_latest else config.max_actor_ckpt_to_keep
    logger.info(f"max checkpoint to keep: {max_keep}")

    # ckpt_dirs is sorted descending by step; keep the first max_keep, remove the rest
    for i, (step, ckpt_dir) in enumerate(ckpt_dirs):
        if i < max_keep:
            logger.info(f"keep checkpoint dir {ckpt_dir}")
        else:
            _remove_ckpt(ckpt_dir)


def _remove_ckpt(ckpt_dir: Path):
    if not _ITER_DIR_PATTERN.match(ckpt_dir.name):
        logger.error(f"Refusing to remove {ckpt_dir}: directory name does not match iter_NNNNNNN pattern")
        return
    try:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        logger.warning(f"remove checkpoint dir {ckpt_dir}")
    except BaseException as e:
        logger.warning(f"Failed to remove checkpoint dir {ckpt_dir}, error: {e}")
