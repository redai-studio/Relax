# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward registry and per-sample rm_type resolution.

This module is deliberately CPU-light: it imports only the standard library
and ``relax.utils.logging_utils``, so it can be imported (or loaded standalone
via ``importlib.util.spec_from_file_location``) without Ray, torch, or any
reward-specific dependency. Handlers registered as ``"pkg.module:attr"``
dotted paths are resolved lazily on first call, preserving the per-reward
lazy-import behavior of the previous if/elif dispatch (e.g. ifbench must not
be imported until it is actually used).

Registrations executed at module scope (like the built-ins below) run in every
process that imports ``relax.engine.rewards``, including Ray actor processes.
Runtime ``register_reward`` calls made only in the driver are not visible to
already-running actors for ``mode="sync"`` rewards; out-of-tree rewards should
keep using ``--custom-rm-path``.
"""

import dataclasses
import importlib
import re
from collections import Counter
from typing import Any, Callable

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_BOXED_PREFIX = "boxed_"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RewardSpec:
    """A single registered reward.

    ``fn`` is either a callable or a ``"pkg.module:attr"`` dotted path that is
    resolved lazily so registration never imports the reward module.

    ``mode`` selects the execution path: ``"sync"`` handlers run in the Ray
    worker pool with signature ``fn(response, label)`` (or
    ``fn(response, label, metadata=metadata)`` when ``pass_metadata`` is set);
    ``"async"`` handlers are awaited in the event loop with signature
    ``async fn(args, sample)``.

    ``label_matcher`` optionally makes the reward eligible for label-based
    inference (see :func:`resolve_rm_type`). The callable takes the raw sample
    label and returns ``bool``; it must reject non-string labels itself.
    """

    name: str
    fn: Callable | str
    mode: str = "sync"
    pass_metadata: bool = False
    label_matcher: Callable[[Any], bool] | None = None

    def resolve(self) -> Callable:
        """Return the handler, importing it on first use for dotted paths."""
        if callable(self.fn):
            return self.fn
        module_name, _, attr = self.fn.partition(":")
        # stdlib importlib on purpose: relax.utils.misc.load_function pulls in
        # torch, which would break CPU-only use of this module.
        return getattr(importlib.import_module(module_name), attr)


_REWARDS: dict[str, RewardSpec] = {}


def register_reward(
    name: str,
    fn: Callable | str,
    *,
    mode: str = "sync",
    pass_metadata: bool = False,
    label_matcher: Callable[[Any], bool] | None = None,
) -> None:
    """Register a reward under ``name``. One call per reward; no router edit.

    Raises ValueError on an empty name, an unknown mode, or a duplicate name
    (listing the registered types, mirroring the ALGOS lookup error style).
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Reward name must be a non-empty string, got {name!r}")
    if mode not in ("sync", "async"):
        raise ValueError(f"Reward mode must be 'sync' or 'async', got {mode!r}")
    if name in _REWARDS:
        raise ValueError(f"Reward type '{name}' is already registered. Registered types: {list_reward_types()}")
    _REWARDS[name] = RewardSpec(name=name, fn=fn, mode=mode, pass_metadata=pass_metadata, label_matcher=label_matcher)


def get_reward_spec(name: str) -> RewardSpec | None:
    return _REWARDS.get(name)


def list_reward_types(mode: str | None = None) -> list[str]:
    return sorted(name for name, spec in _REWARDS.items() if mode is None or spec.mode == mode)


# ---------------------------------------------------------------------------
# Degradation warnings: log once per identity, count every occurrence
# ---------------------------------------------------------------------------

# Identity = (reason, offending value). The set deduplicates log lines so a
# large batch cannot spam the log; the counter keeps exact per-identity totals
# for auditing and tests. Same house pattern as _WARNED_DROPPED_KIMI_KWARGS in
# relax/utils/data/processing_utils.py.
_ROUTER_WARNED: set[tuple[str, Any]] = set()
_ROUTER_COUNTERS: Counter = Counter()


def _note(reason: str, value: Any, sample_index: Any = None, action: str = "", suggestion: Any = None) -> None:
    identity = (reason, value)
    _ROUTER_COUNTERS[identity] += 1
    if identity in _ROUTER_WARNED:
        return
    _ROUTER_WARNED.add(identity)
    hint = f"; label matcher suggests {suggestion!r}" if suggestion is not None else ""
    logger.warning(
        f"Reward routing degradation: reason={reason} value={value!r} sample_index={sample_index} "
        f"action={action or 'none'}{hint}. Further occurrences of this identity are counted but not logged."
    )


def get_router_stats() -> dict[tuple[str, Any], int]:
    """Exact per-identity degradation counts (copy; includes suppressed
    ones)."""
    return dict(_ROUTER_COUNTERS)


def reset_router_state() -> None:
    """Clear warn-dedup state and counters.

    Never touches registrations.
    """
    _ROUTER_WARNED.clear()
    _ROUTER_COUNTERS.clear()


# ---------------------------------------------------------------------------
# Per-sample rm_type resolution
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ResolvedRoute:
    """Resolution result consumed by RewardExecutor.execute.

    ``rm_type == ""`` means unresolved-missing: the executor keeps the baseline
    "Rule-based RM type is not specified." error. ``zero_reward`` short-
    circuits dispatch and scores 0 (reward-key aware).
    """

    rm_type: str
    boxed: bool = False
    zero_reward: bool = False


def _safe_match(matcher: Callable[[Any], bool], label: Any) -> bool:
    try:
        return bool(matcher(label))
    except Exception as e:
        logger.debug(f"label matcher {matcher!r} failed on label {label!r}: {type(e).__name__}: {e}")
        return False


def _fallback_route(fallback: str) -> ResolvedRoute:
    return ResolvedRoute(rm_type="", zero_reward=True) if fallback == "zero" else ResolvedRoute(rm_type=fallback)


def resolve_rm_type(
    *,
    metadata_rm_type: Any,
    args_rm_type: Any,
    label: Any,
    infer: bool = False,
    fallback: str | None = None,
    sample_index: Any = None,
) -> ResolvedRoute:
    """Resolve the reward type for one sample.

    Priority: explicit type (sample metadata over ``--rm-type``, exactly the
    baseline expression) > unique label-matcher inference (only when ``infer``)
    > configured ``fallback`` (``"zero"`` or a registered name) > baseline
    behavior. With ``infer=False`` and ``fallback=None`` the result is
    bit-identical to the pre-registry pipeline, including flowing unknown
    types through to the worker so they still fail there.

    Degradations (unknown / missing / conflict / ambiguous inference) are
    counted per identity and logged once each; see :func:`get_router_stats`.
    """
    explicit = (metadata_rm_type or args_rm_type or "").strip()
    boxed = explicit.startswith(_BOXED_PREFIX)
    base = explicit[len(_BOXED_PREFIX) :] if boxed else explicit

    inferred = None
    if infer:
        matches = [
            name
            for name, spec in _REWARDS.items()
            if spec.label_matcher is not None and _safe_match(spec.label_matcher, label)
        ]
        if len(matches) == 1:
            inferred = matches[0]
        elif len(matches) > 1:
            _note(
                "ambiguous_inference",
                tuple(sorted(matches)),
                sample_index,
                action="ignoring label inference",
            )

    if base:
        # An explicit type always wins; inference never overrides it (a typo
        # silently rerouted by label content would mask config bugs).
        if inferred is not None and inferred != base and base in _REWARDS:
            _note("conflict", (base, inferred), sample_index, action=f"using explicit type {base!r}")
        if fallback is not None and base not in _REWARDS:
            _note("unknown", base, sample_index, action=f"routing to fallback {fallback!r}", suggestion=inferred)
            return _fallback_route(fallback)
        return ResolvedRoute(rm_type=base, boxed=boxed)

    if inferred is not None:
        return ResolvedRoute(rm_type=inferred)
    if fallback is not None:
        _note("missing", "", sample_index, action=f"routing to fallback {fallback!r}")
        return _fallback_route(fallback)
    return ResolvedRoute(rm_type="")


# ---------------------------------------------------------------------------
# Built-in label matchers (conservative by design: a false inference trains on
# the wrong reward, a non-inference merely falls back with a warning)
# ---------------------------------------------------------------------------

_MATH_LABEL_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_MULTIPLE_CHOICE_LABEL_RE = re.compile(r"\s*<answer>\s*[A-Z]\s*</answer>\s*")


def _looks_like_math_label(label: Any) -> bool:
    """Strict numeral / decimal / simple fraction, nothing else (no LaTeX)."""
    return isinstance(label, str) and _MATH_LABEL_RE.fullmatch(label.strip()) is not None


def _looks_like_multiple_choice_label(label: Any) -> bool:
    """A single uppercase letter wrapped in <answer></answer> tags."""
    return isinstance(label, str) and _MULTIPLE_CHOICE_LABEL_RE.fullmatch(label) is not None


# ---------------------------------------------------------------------------
# Adapters preserving the exact semantics of the former if/elif branches
# ---------------------------------------------------------------------------


def _math_reward(response, label):
    from relax.engine.rewards.math_utils import grade_answer_verl

    return 1 if grade_answer_verl(response, label) else 0


def _f1_reward(response, label):
    from relax.engine.rewards.f1 import f1_score

    return f1_score(response, label)[0]


def _random_reward(response, label):
    import random

    return random.randint(0, 1)


# ---------------------------------------------------------------------------
# Built-in synchronous rewards. Async built-ins (remote_rm, dapo-genrm, dummy)
# register from relax/engine/rewards/__init__.py, next to their handlers.
# ---------------------------------------------------------------------------

register_reward("deepscaler", "relax.engine.rewards.deepscaler:get_deepscaler_rule_based_reward")
register_reward("geo3k", "relax.engine.rewards.geo3k:get_geo3k_reward")
register_reward("openr1mm", "relax.engine.rewards.openr1mm:get_openr1mm_rule_based_reward")
register_reward(
    "multiple_choice",
    "relax.engine.rewards.multiple_choice:get_multiple_choice_reward",
    label_matcher=_looks_like_multiple_choice_label,
)
register_reward("dapo", "relax.engine.rewards.math_dapo_utils:compute_score")
register_reward("math", _math_reward, label_matcher=_looks_like_math_label)
register_reward("mopd", "relax.engine.rewards.mopd:get_mopd_reward", pass_metadata=True)
register_reward("f1", _f1_reward)
register_reward("gpqa", "relax.engine.rewards.gpqa:compute_gpqa_reward", pass_metadata=True)
register_reward("ifbench", "relax.engine.rewards.ifbench:compute_ifbench_reward", pass_metadata=True)
register_reward("random", _random_reward)
