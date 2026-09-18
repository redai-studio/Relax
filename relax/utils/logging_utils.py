# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import os

from relax.utils.env import Envs


_LOGGER_CONFIGURED = False
_CONFIGURED_PID = None

# Logging methods that need patching to ensure configuration and correct stacklevel
_LOG_METHODS = ("debug", "info", "warning", "error", "critical", "exception")

# ANSI color codes for different log levels
_LOG_COLORS = {
    logging.DEBUG: "\033[36m",  # Cyan
    logging.INFO: "\033[32m",  # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",  # Red
    logging.CRITICAL: "\033[35m",  # Magenta
}
_RESET_COLOR = "\033[0m"

try:
    import colorlog

    log_color = {
        "DEBUG": "bold_cyan",
        "INFO": "bold_white",
        "WARNING": "bold_yellow",
        "ERROR": "bold_red",
        "CRITICAL": "bg_white,bold_red",
    }

    def get_formatter(prefix: str = ""):
        prefix_str = f"[{prefix}]" if prefix else ""
        return colorlog.ColoredFormatter(
            "%(green)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s | %(cyan)s%(name)s:%(lineno)d%(reset)s"
            f"{prefix_str} %(log_color)s%(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_color,
        )
except ImportError:

    class ColoredFormatter(logging.Formatter):
        """A formatter that adds color to log messages based on level."""

        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)

        def format(self, record):
            try:
                # Get the color for this log level
                color = _LOG_COLORS.get(record.levelno, "")
                # Format the message
                message = super().format(record)
                # Apply color if available
                if color:
                    return f"{color}{message}{_RESET_COLOR}"
                return message
            except Exception:
                # Fall back to uncolored format
                return super().format(record)

    def get_formatter(prefix: str = ""):
        return ColoredFormatter(
            fmt=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )


def configure_logger(prefix: str = "") -> None:
    """Configure the root logger with standard format.

    This function is safe to call multiple times and idempotent.
    Each Ray worker process has its own copy of this module, so the global
    _LOGGER_CONFIGURED flag works correctly across distributed execution.

    Args:
        prefix: Additional prefix for log timestamps (e.g., "[worker]")
    """
    global _LOGGER_CONFIGURED, _CONFIGURED_PID
    current_pid = os.getpid()

    # If we're in a new process (Ray worker), reconfigure
    if _CONFIGURED_PID is not None and _CONFIGURED_PID != current_pid:
        _LOGGER_CONFIGURED = False
        _CONFIGURED_PID = None

    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True
    _CONFIGURED_PID = current_pid

    try:
        # Read at configure time, not import time: logging is set up once per
        # process, and by then a Ray worker has its runtime_env applied.
        log_level = Envs.LOG_LEVEL.upper()

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create StreamHandler with colored formatter
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(get_formatter(prefix))
        root_logger.addHandler(handler)

        # Silence noisy third-party DEBUG loggers (PIL dumps PNG chunk metadata per image)
        for noisy in ("PIL",):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        install_asyncio_noise_filter()
    except Exception:
        # Silently ignore configuration errors to prevent breaking the application
        pass


_ASYNCIO_FILTER_INSTALLED = False


class _ChainFutureFilter(logging.Filter):
    """Drop the spurious uvloop + Py3.12 `_chain_future._set_state`
    AssertionError that Ray's ServeController emits when a proxy health-check
    response arrives after the controller has already cancelled the awaiting
    future.

    The error is harmless but pollutes every run.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return not (
                "Exception in callback _chain_future.<locals>._set_state" in record.getMessage()
                and record.exc_info is not None
                and isinstance(record.exc_info[1], AssertionError)
            )
        except Exception:
            return True


def install_asyncio_noise_filter() -> None:
    """Attach the _chain_future filter to the asyncio logger in the current
    process.

    Idempotent. Safe to call from a Ray `worker_process_setup_hook` so that
    every worker (driver, training actor, SGLang engine, **and**
    ServeController) suppresses the noise even though Ray internals never
    import relax.
    """
    global _ASYNCIO_FILTER_INSTALLED
    if _ASYNCIO_FILTER_INSTALLED:
        return
    try:
        logging.getLogger("asyncio").addFilter(_ChainFutureFilter())
        _ASYNCIO_FILTER_INSTALLED = True
    except Exception:
        pass


install_asyncio_noise_filter()


class LazyConfiguredLogger(logging.Logger):
    """A logger that auto-configures on first use.

    Inherits from logging.Logger and patches logging methods to ensure:
    - Logging system is configured when first used (critical for Ray worker processes)
    - Correct stacklevel for accurate file/line reporting
    - Graceful error handling that doesn't break the application
    """

    _configured = False

    def _ensure_configured(self) -> None:
        """Configure logger if not already done in this process."""
        if not LazyConfiguredLogger._configured:
            try:
                configure_logger()
                LazyConfiguredLogger._configured = True
            except Exception:
                # Even if configuration fails, allow logging to continue
                pass


def _create_log_method(method_name: str):
    """Factory function to create a patched logging method.

    Each patched method ensures configuration and correct stacklevel.
    """

    def log_method(self, msg, *args, **kwargs):
        try:
            self._ensure_configured()
            # Add stacklevel=2 to skip this wrapper and show actual caller
            kwargs.setdefault("stacklevel", 2)
            # Call parent class method
            getattr(logging.Logger, method_name)(self, msg, *args, **kwargs)
        except Exception:
            # Silently fail logging to avoid breaking the application
            pass

    return log_method


# Dynamically patch logging methods onto LazyConfiguredLogger
for _method_name in _LOG_METHODS:
    setattr(LazyConfiguredLogger, _method_name, _create_log_method(_method_name))


def get_logger(name: str, prefix: str = "") -> logging.Logger:
    """Get a logger instance with automatic lazy configuration.

    Returns a logger that automatically configures logging on first use.
    This is critical for Ray worker processes that execute in separate Python interpreters.

    Developers only need to write this once per module:
    ```python
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    ```

    The logger will automatically configure itself when first used, even if called
    from a Ray worker process, without requiring explicit configure_logger() calls.

    Args:
        name: Logger name (typically __name__ from the calling module)
        prefix: (Deprecated) Additional prefix for log timestamps

    Returns:
        logging.Logger: A logger instance ready for use. This is normally a
            ``LazyConfiguredLogger``; if ``name`` is already registered as a
            different ``Logger`` subclass, that instance is returned as-is.
    """
    logger = logging.getLogger(name)

    if type(logger) is logging.Logger:
        # Upgrade the manager-registered instance in place instead of building a
        # detached `LazyConfiguredLogger(name)`: a Logger constructed directly
        # (not via getLogger) has parent=None and no handlers, so every record
        # it emits falls through to logging.lastResort (WARNING+) and INFO
        # output silently disappears. The plain instance exists either because
        # getLogger just created it (fresh name, default logger class) or
        # because the name was pre-registered before get_logger ran — e.g.
        # Ray/cloudpickle reconstructs @ray.remote class namespaces by value
        # and unpickles their module-global loggers via getLogger(name) (see
        # the _LOCAL_ROLLOUT_MANAGER note in relax/distributed/ray/rollout.py).
        # In-place upgrade also avoids logging.setLoggerClass entirely: that
        # would mutate process-global state, race with concurrent getLogger
        # calls on other threads, and stomp third-party default logger classes.
        logger.__class__ = LazyConfiguredLogger

    return logger
