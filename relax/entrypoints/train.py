# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import atexit
import os
import signal
import sys
from pathlib import Path

import ray
import yaml
from ray import serve

from relax.utils import try_import_telemetry_hook


# Optional telemetry hook: import before Controller so it can install patches.
# Missing or broken hooks must not change training behavior.
try_import_telemetry_hook()

from relax.core.controller import Controller  # noqa: E402
from relax.core.node_group_affinity import maybe_pin_baseline_to_stable  # noqa: E402
from relax.utils.arguments import parse_args  # noqa: E402
from relax.utils.logging_utils import get_logger  # noqa: E402
from relax.utils.tracking_utils import init_tracking  # noqa: E402
from relax.utils.utils import post_process_env  # noqa: E402


cur_file_dir = Path(__file__).absolute().parent.parent.parent
logger = get_logger(__name__)

# Global reference so signal handlers / atexit can reach the controller.
_ctrl: Controller | None = None
_shutdown_done = False
_kernel_cache_heartbeat = None


def _hard_exit(code: int):
    """Exit without running Python/native extension destructors."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


def _graceful_shutdown(sig=None, frame=None, exit_code: int | None = None):
    """Shut down SGLang engines and Ray on SIGTERM / SIGINT / atexit."""
    global _kernel_cache_heartbeat, _shutdown_done

    if sig is not None:
        exit_code = 128 + sig

    if _shutdown_done:
        if exit_code is not None:
            _hard_exit(exit_code)
        return

    _shutdown_done = True

    sig_name = signal.Signals(sig).name if sig else "atexit"
    logger.info(f"Graceful shutdown triggered ({sig_name}) — cleaning up SGLang engines...")

    # Detached cache agents are outside the Ray Job's process tree. Notify them
    # first so a short Ray Job SIGTERM grace period cannot prevent publishing;
    # the normal path waits for a final snapshot after Serve stops train actors.
    if sig is not None and _kernel_cache_heartbeat is not None:
        try:
            _kernel_cache_heartbeat.request_finalize(sig_name)
        except Exception as e:
            logger.warning(f"Kernel cache finalize request failed during {sig_name}: {e}")

    if _ctrl is not None:
        try:
            _ctrl.shutdown()
        except Exception as e:
            logger.warning(f"Controller shutdown error during {sig_name}: {e}")

    if ray.is_initialized():
        try:
            serve.shutdown()
            if _kernel_cache_heartbeat is not None:
                results = _kernel_cache_heartbeat.finalize(
                    sig_name,
                    timeout_sec=int(os.environ.get("RELAX_KERNEL_CACHE_EXIT_TIMEOUT_SEC", "600")),
                )
                logger.info(f"Kernel cache agents finalized: {results}")
            ray.shutdown()
            logger.info("Ray shutdown successfully")
        except Exception as e:
            logger.warning(f"Ray shutdown error during {sig_name}: {e}")

    if exit_code is not None:
        _hard_exit(exit_code)


def main(args):
    global _ctrl, _kernel_cache_heartbeat

    # Load runtime_env from config so we can both pass it to ray.init and
    # explicitly to the Serve deployment. Ensure it's available even if Ray
    # is already initialized.
    with open(os.path.join(cur_file_dir, "configs/env.yaml")) as file:
        runtime_env = yaml.safe_load(file)

    runtime_env = post_process_env(args, runtime_env)
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env=runtime_env)
        logger.info("Ray initialized successfully")
        try:
            serve.start(
                http_options={"host": "0.0.0.0", "port": "8000"},
                detached=True,
            )
        except RuntimeError:
            pass

    if os.environ.get("RELAX_KERNEL_CACHE_DIR"):
        from relax.distributed.ray.kernel_cache import start_kernel_cache_heartbeat_from_env

        _kernel_cache_heartbeat = start_kernel_cache_heartbeat_from_env()

    # init_tracking must run after serve.start() (metrics adapter probes Ray
    # Serve for the /metrics endpoint) and before Controller() (wandb primary
    # writes wandb_run_id into args, which then propagates to remote actors).
    init_tracking(args)

    maybe_pin_baseline_to_stable(args)

    ctrl = Controller(args, runtime_env)
    _ctrl = ctrl

    # Register signal handlers so that `ray job stop` (SIGTERM) triggers cleanup.
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    atexit.register(_graceful_shutdown)

    try:
        ctrl.training_loop()
    except Exception as e:
        logger.exception(f"Training loop failed with error: {e}")
        _graceful_shutdown(exit_code=1)

    logger.info("Main func successfully")
    # Gracefully shut down SGLang engine processes before tearing down Ray Serve.
    _graceful_shutdown(exit_code=0)


if __name__ == "__main__":
    args = parse_args()
    main(args)
