"""Regression tests for :func:`relax.utils.logging_utils.get_logger`.

A logger name can already be registered as a plain :class:`logging.Logger`
before ``get_logger`` runs -- Ray/cloudpickle reconstructs ``@ray.remote``
class namespaces by value and unpickles their module-global loggers through
``logging.getLogger(name)`` with the default logger class.

The previous fallback built a fresh ``LazyConfiguredLogger(name)``. Because
that bypasses ``logging.getLogger``, the resulting object is detached from the
logging hierarchy: ``parent`` is ``None`` and ``handlers`` is empty, so every
record falls through to ``logging.lastResort`` (WARNING and above) and all
INFO output silently disappears.
"""

import asyncio
import logging

import pytest

from relax.utils.logging_utils import LazyConfiguredLogger, get_logger


def _unregister(name: str) -> None:
    logging.Logger.manager.loggerDict.pop(name, None)


def test_get_logger_returns_lazy_configured_logger():
    name = "relax_test.logging_utils.fresh"
    _unregister(name)
    try:
        logger = get_logger(name)
        assert isinstance(logger, LazyConfiguredLogger)
        assert logging.getLogger(name) is logger
    finally:
        _unregister(name)


def test_get_logger_upgrades_preregistered_plain_logger_in_place():
    """The pre-registered instance must be upgraded, not replaced."""
    name = "relax_test.logging_utils.preregistered"
    _unregister(name)
    try:
        plain = logging.getLogger(name)
        assert type(plain) is logging.Logger

        logger = get_logger(name)

        assert isinstance(logger, LazyConfiguredLogger)
        # Same object: upgraded in place rather than detached.
        assert logger is plain
        assert logging.getLogger(name) is logger
    finally:
        _unregister(name)


def test_upgraded_logger_stays_attached_to_hierarchy():
    """Guards the actual failure mode: silently dropped INFO records.

    A detached logger has ``parent is None`` and no handlers, so records never
    reach an ancestor handler.
    """
    name = "relax_test.logging_utils.attached"
    _unregister(name)
    try:
        logging.getLogger(name)  # pre-register as a plain Logger
        logger = get_logger(name)

        assert logger.parent is not None
        assert logger.manager is logging.Logger.manager

        # Walk to the root the way logging.Logger.callHandlers does.
        ancestors = []
        current = logger
        while current:
            ancestors.append(current)
            current = current.parent
        assert ancestors[-1] is logging.getLogger()
    finally:
        _unregister(name)


def test_upgraded_logger_emits_info_records():
    """End-to-end guard on the actual failure mode: dropped INFO records.

    ``LazyConfiguredLogger`` installs its own handler on first use, so the
    record is asserted at the logger's own handlers rather than at the root. A
    detached logger has no handlers at all and would drop the record to
    ``logging.lastResort``, which only passes WARNING and above.
    """
    name = "relax_test.logging_utils.emit"
    _unregister(name)
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    try:
        logging.getLogger(name)  # pre-register as a plain Logger
        logger = get_logger(name)
        logger.setLevel(logging.INFO)

        # Trigger lazy configuration so the real handler set is in place.
        logger.info("warmup")

        handler = _Capture(level=logging.INFO)
        target = logger if logger.handlers else logger.parent
        assert target is not None, "logger must be attached to the hierarchy"
        target.addHandler(handler)
        try:
            logger.info("rloo-metrics-visible")
        finally:
            target.removeHandler(handler)

        assert "rloo-metrics-visible" in records
    finally:
        _unregister(name)


def test_get_logger_leaves_global_logger_class_untouched():
    """get_logger must not mutate process-global logging state.

    The previous implementation flipped ``logging.setLoggerClass`` around its
    ``getLogger`` call, which raced with concurrent ``getLogger`` calls on
    other threads and reset third-party default logger classes to
    ``logging.Logger``.
    """

    class _ThirdPartyLogger(logging.Logger):
        pass

    name = "relax_test.logging_utils.global_class"
    _unregister(name)
    previous_class = logging.getLoggerClass()
    try:
        logging.setLoggerClass(_ThirdPartyLogger)

        logger = get_logger(name)

        # The third-party default class must survive get_logger.
        assert logging.getLoggerClass() is _ThirdPartyLogger
        # A name created under a third-party default class is respected
        # (registered and attached), consistent with the subclass test below.
        assert logging.getLogger(name) is logger
        assert type(logger) is _ThirdPartyLogger
    finally:
        logging.setLoggerClass(previous_class)
        _unregister(name)


def test_get_logger_preserves_logger_subclasses():
    """Only plain ``logging.Logger`` instances are upgraded."""

    class _CustomLogger(logging.Logger):
        pass

    name = "relax_test.logging_utils.subclass"
    _unregister(name)
    previous_class = logging.getLoggerClass()
    try:
        logging.setLoggerClass(_CustomLogger)
        custom = logging.getLogger(name)
        logging.setLoggerClass(previous_class)
        assert type(custom) is _CustomLogger

        logger = get_logger(name)

        # Left untouched: not our class, but not a plain Logger either.
        assert type(logger) is _CustomLogger
    finally:
        logging.setLoggerClass(previous_class)
        _unregister(name)


@pytest.mark.parametrize(
    ("callback", "error", "visible"),
    [
        ("_chain_future.<locals>._set_state", TypeError("StopIteration interacts badly with generators"), True),
        ("_chain_future.<locals>._set_state", RuntimeError("processor failed"), True),
        ("_chain_future.<locals>._set_state", None, True),
        ("other_callback", AssertionError("unexpected state"), True),
        ("_chain_future.<locals>._set_state", AssertionError(), False),
    ],
)
def test_asyncio_filter_preserves_unexpected_callback_errors(caplog, callback, error, visible):
    """Processor failures must remain visible through asyncio's error
    handler."""
    loop = asyncio.new_event_loop()
    message = f"Exception in callback {callback}(...)"
    try:
        with caplog.at_level(logging.ERROR, logger="asyncio"):
            loop.call_exception_handler({"message": message, "exception": error})
        assert any(message in record.getMessage() for record in caplog.records) is visible
    finally:
        loop.close()
