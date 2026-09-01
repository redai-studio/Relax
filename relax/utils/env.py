# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Central registry for every environment variable Relax reads.

Each variable is declared exactly once as an :class:`EnvProperty` descriptor on
:class:`Envs`, and call sites read ``Envs.<NAME>`` instead of touching
``os.environ`` directly. That keeps the name, type and default of every knob in
one place, and lets :func:`validate_env` reject typos and unregistered
variables at startup instead of letting them silently do nothing.
"""

import os
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple, TypeVar


T = TypeVar("T")

# Environment namespaces owned by Relax. Any variable in the environment under
# one of these prefixes must be declared below, otherwise `validate_env` fails.
#
# Only genuinely exclusive prefixes belong here. `GENRM_`, `ROLLOUT_`, `DCS_`
# and `METRICS_` are deliberately absent even though variables under them are
# declared: they are ordinary English words that collide with shell-local
# tuning knobs in scripts/training (`ROLLOUT_BATCH_SIZE`,
# `ROLLOUT_MAX_RESPONSE_LEN`), so claiming them would reject a user's own shell
# exports. Namespaces belonging to third-party components the training stack
# configures (NCCL_, NVTE_, NVSHMEM_, SGLANG_, TQ_, SLURM_, RAY_, CUDA_, ...)
# are excluded for the same reason: we neither own nor can enumerate them.
OWNED_ENV_PREFIXES: Tuple[str, ...] = (
    "RELAX_",
    "SLIME_",
)

# Sub-namespaces carved out of the above. `SLIME_SCRIPT_<FIELD>` names are
# generated at runtime by `utils/external/typer_utils.py::dataclass_cli` from
# whatever dataclass a launcher script defines, so they cannot be enumerated
# here even in principle.
IGNORED_ENV_PREFIXES: Tuple[str, ...] = ("SLIME_SCRIPT_",)

_TRUE_VALUES = frozenset({"1", "t", "true", "y", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "f", "false", "n", "no", "off"})


# A variable whose name contains one of these has its value masked in
# diagnostics. Substring matching, deliberately over-inclusive: masking a
# non-secret (`RELAX_OPD_TOKEN_IDS_LOGPROB_K` is a top-k count, not a
# credential) only costs a line of diagnostic output, while missing a real
# credential writes it to the log verbatim.
_SENSITIVE_ENV_MARKERS: Tuple[str, ...] = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD", "AUTH")


def mask_if_sensitive(env: str, value: Any) -> Any:
    """Redact credentials so neither a diagnostic dump nor a parse error writes
    one to a log verbatim."""
    name = env.upper()
    if value is None or not any(marker in name for marker in _SENSITIVE_ENV_MARKERS):
        return value
    text = str(value)
    # Only reveal a prefix when the value is long enough that four characters
    # are not most of it.
    return f"{text[:4]}***" if len(text) > 8 else "***"


def _parse_bool(raw: str, env: str) -> bool:
    """Parse a boolean environment value, rejecting anything ambiguous.

    Unlike the lenient ``value in ("1", "true", ...)`` checks this replaces, an
    unrecognized value raises instead of quietly resolving to ``False`` — a
    typo such as ``RELAX_TGD_PROFILE=ture`` should not silently disable the
    feature the operator meant to turn on.
    """
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    expected = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(f"Invalid boolean for env {env}: {mask_if_sensitive(env, raw)!r} (expected one of: {expected})")


def is_env_set(env: str) -> bool:
    """Whether ``env`` is present with a non-empty value.

    Mirrors :class:`EnvProperty`'s "empty means unset" rule so that call sites
    distinguishing "explicitly configured" from "fell back to the default" stay
    consistent with how the value itself resolves.
    """
    raw = os.environ.get(env)
    return raw is not None and raw.strip() != ""


class EnvProperty:
    """A typed, lazily resolved binding to a single environment variable.

    Resolution is lazy on purpose: several call sites write to ``os.environ``
    and read the value back later in the same process (``LOCAL_RANK`` in
    ``distributed/ray/train_actor.py``, ``RELAX_OPD_TOKEN_IDS_LOGPROB_K`` in
    ``backends/sglang/sglang_engine.py``), so values must not be frozen at
    import time. An unset *or empty* variable resolves to ``default``; a set but
    unparseable one raises ``ValueError``.

    Bindings are read-only. ``Envs.<NAME> = value`` is rejected by
    :class:`_EnvsMeta`; mutate ``os.environ`` when a value genuinely has to
    change at runtime, so that child processes observe the change too.
    """

    __slots__ = ("_env", "_type", "_default")

    def __init__(self, env: str, dtype: type, default: Optional[T] = None) -> None:
        self._env = env
        self._type = dtype
        self._default = default

    @property
    def env(self) -> str:
        """Name of the backing environment variable."""
        return self._env

    def parse(self, raw: Optional[str]) -> Any:
        """Convert a raw string to the declared type, ``None``/empty meaning
        default."""
        if raw is None or raw.strip() == "":
            return self._default
        if self._type is bool:
            return _parse_bool(raw, self._env)
        try:
            return self._type(raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid value for env {self._env}: {mask_if_sensitive(self._env, raw)!r} "
                f"(expected {self._type.__name__})"
            ) from exc

    def resolve(self) -> Any:
        """Read and convert the current value of the backing variable."""
        return self.parse(os.environ.get(self._env))

    def __get__(self, instance: Any, owner: type) -> Any:
        return self.resolve()


class _EnvsMeta(type):
    """Rejects ``Envs.<NAME> = value``, which would destroy the binding.

    ``__set__`` only fires for *instance* attributes. Without this guard a
    class-level assignment replaces the :class:`EnvProperty` with a plain value,
    and that name stops tracking ``os.environ`` for the rest of the process —
    silently, and only for whichever code imported the class afterwards.
    """

    def __setattr__(cls, name: str, value: Any) -> None:
        if isinstance(cls.__dict__.get(name), EnvProperty):
            raise AttributeError(
                f"{cls.__name__}.{name} is a read-only environment binding. "
                f"Set os.environ[{name!r}] instead of assigning to it."
            )
        super().__setattr__(name, value)


class Envs(metaclass=_EnvsMeta):
    # ------------- GenRM -------------
    GENRM_SERVE_MAX_ONGOING_REQUESTS = EnvProperty("GENRM_SERVE_MAX_ONGOING_REQUESTS", int, 256)
    GENRM_ENGINE_RETRY_ATTEMPTS = EnvProperty("GENRM_ENGINE_RETRY_ATTEMPTS", int, 3)
    GENRM_ENGINE_CACHE_REFRESH_COOLDOWN_S = EnvProperty("GENRM_ENGINE_CACHE_REFRESH_COOLDOWN_S", float, 5.0)

    # ------------- Rollout -------------
    ROLLOUT_SERVE_MAX_ONGOING_REQUESTS = EnvProperty("ROLLOUT_SERVE_MAX_ONGOING_REQUESTS", int, 256)
    ROLLOUT_IMAGE_FETCH_ATTEMPTS = EnvProperty("ROLLOUT_IMAGE_FETCH_ATTEMPTS", int, 3)
    ROLLOUT_IMAGE_FETCH_TIMEOUT_S = EnvProperty("ROLLOUT_IMAGE_FETCH_TIMEOUT_S", float, 10.0)
    ROLLOUT_IMAGE_FETCH_BACKOFF_S = EnvProperty("ROLLOUT_IMAGE_FETCH_BACKOFF_S", float, 0.5)

    # ------------- DCS -------------
    DCS_SERVE_MAX_ONGOING_REQUESTS = EnvProperty("DCS_SERVE_MAX_ONGOING_REQUESTS", int, 100)

    # ------------- Metrics -------------
    METRICS_SERVE_MAX_ONGOING_REQUESTS = EnvProperty("METRICS_SERVE_MAX_ONGOING_REQUESTS", int, 200)
    TENSORBOARD_DIR = EnvProperty("TENSORBOARD_DIR", str, None)
    CLEARML_PROJECT = EnvProperty("CLEARML_PROJECT", str, "unknown_project")
    CLEARML_TASK = EnvProperty("CLEARML_TASK", str, "unknown_task")
    CLEARML_TAGS = EnvProperty("CLEARML_TAGS", str, None)
    WANDB_API_KEY = EnvProperty("WANDB_API_KEY", str, None)
    WANDB_MODE = EnvProperty("WANDB_MODE", str, None)
    REGION = EnvProperty("REGION", str, None)
    USER = EnvProperty("USER", str, None)
    LOGNAME = EnvProperty("LOGNAME", str, None)

    # ------------- OPD / SGLang patches -------------
    RELAX_OPD_PREEXPANDED_PATCH = EnvProperty("RELAX_OPD_PREEXPANDED_PATCH", bool, False)
    RELAX_OPD_PER_POS_TOKEN_IDS = EnvProperty("RELAX_OPD_PER_POS_TOKEN_IDS", bool, False)
    RELAX_OPD_TOKEN_IDS_LOGPROB_K = EnvProperty("RELAX_OPD_TOKEN_IDS_LOGPROB_K", str, "0")
    RELAX_OPTIMIZE_ROUTING_REPLAY = EnvProperty("RELAX_OPTIMIZE_ROUTING_REPLAY", bool, False)
    RELAX_FORCE_LOGPROBS_BASE64 = EnvProperty("RELAX_FORCE_LOGPROBS_BASE64", bool, False)

    # ------------- Telemetry / Log -------------
    RELAX_TELEMETRY_HOOK = EnvProperty("RELAX_TELEMETRY_HOOK", str, None)
    LOG_LEVEL = EnvProperty("LOG_LEVEL", str, "INFO")
    # Ray's own knob; it accepts "1" and "legacy", so it stays a string rather
    # than a bool to avoid rejecting a value Ray itself considers valid.
    RAY_DEBUG = EnvProperty("RAY_DEBUG", str, "0")

    # ------------- Device / Training -------------
    RELAX_DEVICE_TYPE = EnvProperty("RELAX_DEVICE_TYPE", str, "")
    RELAX_EMPTY_POLL_SLEEP_MS = EnvProperty("RELAX_EMPTY_POLL_SLEEP_MS", float, 50.0)
    RELAX_FETCH_SPLIT_MAX_RETRIES = EnvProperty("RELAX_FETCH_SPLIT_MAX_RETRIES", int, 20)
    RELAX_SFT_TQ_SHARDS = EnvProperty("RELAX_SFT_TQ_SHARDS", int, 1)

    # ------------- S3 model cache cleanup -------------
    RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S = EnvProperty("RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S", float, 600.0)
    RELAX_S3_MODEL_CLEANUP_CANCEL_TIMEOUT_S = EnvProperty("RELAX_S3_MODEL_CLEANUP_CANCEL_TIMEOUT_S", float, 30.0)

    # ------------- Extra Modules -------------
    RELAX_EXTRA_MODULES = EnvProperty("RELAX_EXTRA_MODULES", str, "")
    RELAX_PROPAGATE_ENV_VARS = EnvProperty("RELAX_PROPAGATE_ENV_VARS", str, "")

    # ------------- Env validation -------------
    RELAX_STRICT_ENV = EnvProperty("RELAX_STRICT_ENV", bool, False)
    RELAX_EXTRA_ENV_ALLOWLIST = EnvProperty("RELAX_EXTRA_ENV_ALLOWLIST", str, "")

    # ------------- Healthcheck -------------
    # Unset means "keep whatever the caller passed in", hence a None default.
    RELAX_ROLLOUT_HEALTHCHECK_TIMEOUT = EnvProperty("RELAX_ROLLOUT_HEALTHCHECK_TIMEOUT", int, None)

    # ------------- Scale-out / weight-sync -------------
    # Per-replica failure reasons are canonicalized and bounded before they reach
    # ScaleOutRequest.error_message (exposed via /status and the monitor TUI).
    RELAX_SCALE_OUT_MAX_REASON_ITEMS = EnvProperty("RELAX_SCALE_OUT_MAX_REASON_ITEMS", int, 3)
    RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN = EnvProperty("RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN", int, 120)
    RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN = EnvProperty("RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN", int, 512)
    # Weight-send NCCL group rank-0 bind window. Kept below the Linux ephemeral
    # range (32768-60999) to avoid collision with OS-assigned ephemeral ports;
    # allocation never returns a port >= *_PORT_MAX.
    RELAX_WEIGHT_SYNC_PORT_BASE = EnvProperty("RELAX_WEIGHT_SYNC_PORT_BASE", int, 20000)
    RELAX_WEIGHT_SYNC_PORT_MAX = EnvProperty("RELAX_WEIGHT_SYNC_PORT_MAX", int, 32000)
    # Full group-init attempts for self-healing on bind/init failure.
    RELAX_WEIGHT_SYNC_MAX_INIT_ATTEMPTS = EnvProperty("RELAX_WEIGHT_SYNC_MAX_INIT_ATTEMPTS", int, 4)
    # Independent rendezvous window for the precheck subprocesses; must not
    # overlap the real direct-sync window above.
    RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE = EnvProperty("RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE", int, 18000)
    RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX = EnvProperty("RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX", int, 20000)
    RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS = EnvProperty(
        "RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS", int, 2
    )
    # Minimum free VRAM before launching a probe (CUDA context + small NCCL bufs).
    RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES = EnvProperty(
        "RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES", int, 512 * 1024**2
    )

    # ------------- Routing Replay -------------
    ENABLE_ROUTING_REPLAY = EnvProperty("ENABLE_ROUTING_REPLAY", bool, False)

    # ------------- TGD Profiling -------------
    RELAX_TGD_PROFILE = EnvProperty("RELAX_TGD_PROFILE", bool, False)
    RELAX_TGD_PROFILE_EVERY = EnvProperty("RELAX_TGD_PROFILE_EVERY", int, 50)

    # ------------- Trajectory replay capture -------------
    # Opt-in production capture. MegatronTrainRayActor.init calls
    # maybe_enable_from_env — no CLI argument-parsing change.
    RELAX_REPLAY_CAPTURE = EnvProperty("RELAX_REPLAY_CAPTURE", bool, False)
    RELAX_REPLAY_CAPTURE_DIR = EnvProperty("RELAX_REPLAY_CAPTURE_DIR", str, None)
    # Comma-separated actor steps as ROLLOUT_ID:STEP_ID (e.g. "0:0,0:1"). Unset = all.
    RELAX_REPLAY_CAPTURE_STEPS = EnvProperty("RELAX_REPLAY_CAPTURE_STEPS", str, None)
    # Comma-separated rollout ids (e.g. "0,1"). Unset = all.
    RELAX_REPLAY_CAPTURE_ROLLOUTS = EnvProperty("RELAX_REPLAY_CAPTURE_ROLLOUTS", str, None)

    # ------------- SGLang Speculative Decoding & External Packages -----------
    SGLANG_ENABLE_SPEC_V2 = EnvProperty("SGLANG_ENABLE_SPEC_V2", str, "")
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvProperty("SGLANG_EXTERNAL_MODEL_PACKAGE", str, None)
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvProperty("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", str, None)
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvProperty("SGLANG_EXTERNAL_MM_MODEL_ARCH", str, None)

    # ------------- Distributed Runtime -------------
    LOCAL_RANK = EnvProperty("LOCAL_RANK", int, 0)
    RANK = EnvProperty("RANK", int, 0)
    SLURM_JOB_NUM_NODES = EnvProperty("SLURM_JOB_NUM_NODES", int, 1)
    SLURM_JOB_ID = EnvProperty("SLURM_JOB_ID", str, None)
    # Required for multinode launches; `command_utils` raises when either is unset.
    SLURM_JOB_HOSTNAMES = EnvProperty("SLURM_JOB_HOSTNAMES", str, None)
    SLURM_NODEID = EnvProperty("SLURM_NODEID", int, None)
    MASTER_ADDR = EnvProperty("MASTER_ADDR", str, "127.0.0.1")
    RAY_JOB_ID = EnvProperty("RAY_JOB_ID", str, None)
    RELAX_LAUNCHER_NAMESPACE = EnvProperty("RELAX_LAUNCHER_NAMESPACE", str, None)
    RELAX_AGENTIC_LAUNCHER_CONCURRENCY = EnvProperty("RELAX_AGENTIC_LAUNCHER_CONCURRENCY", str, None)
    SLIME_HOST_IP = EnvProperty("SLIME_HOST_IP", str, None)
    SLIME_PREFER_IPV6 = EnvProperty("SLIME_PREFER_IPV6", bool, False)

    # ------------- Launcher scripts -------------
    SLIME_SCRIPT_EXTERNAL_RAY = EnvProperty("SLIME_SCRIPT_EXTERNAL_RAY", bool, False)
    SLIME_SCRIPT_ENABLE_RAY_SUBMIT = EnvProperty("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", bool, True)
    SLIME_TEST_ENABLE_INFINITE_RUN = EnvProperty("SLIME_TEST_ENABLE_INFINITE_RUN", bool, False)
    GITHUB_COMMIT_NAME = EnvProperty("GITHUB_COMMIT_NAME", str, None)

    # ------------- Agentic runner -------------
    RELAX = EnvProperty("RELAX", str, None)

    # ------------- Entrypoint / launcher scripts -------------
    # Set by scripts/entrypoint/*.sh; read back by the training scripts to
    # detect whether they were invoked through an entrypoint wrapper.
    RELAX_ENTRYPOINT_MODE = EnvProperty("RELAX_ENTRYPOINT_MODE", str, None)
    # Forwarded to the SGLang/torch-memory-saver side, not parsed by Relax, so
    # it stays a raw string.
    RELAX_SKIP_TORCH_MEMORY_SAVER = EnvProperty("RELAX_SKIP_TORCH_MEMORY_SAVER", str, "0")
    SLIME_LAYER_SNAPSHOT = EnvProperty("SLIME_LAYER_SNAPSHOT", str, "0")
    SLIME_ENABLE_PROFILING = EnvProperty("SLIME_ENABLE_PROFILING", str, "false")

    # ------------- Agentic session subprocesses -------------
    # Written into the runner subprocess environment by agentic/pipeline/runtime.py.
    RELAX_BASE_URL = EnvProperty("RELAX_BASE_URL", str, None)
    RELAX_SESSION_ID = EnvProperty("RELAX_SESSION_ID", str, None)
    RELAX_GROUP_ID = EnvProperty("RELAX_GROUP_ID", str, None)
    RELAX_ROLLOUT_MODE = EnvProperty("RELAX_ROLLOUT_MODE", str, None)
    RELAX_SESSION_IO_DIR = EnvProperty("RELAX_SESSION_IO_DIR", str, None)
    RELAX_INPUT_JSON = EnvProperty("RELAX_INPUT_JSON", str, None)
    RELAX_OUTPUT_JSON = EnvProperty("RELAX_OUTPUT_JSON", str, None)

    # ------------- Debug tooling (scripts/tools/repro_*) -------------
    # Declared, though only scripts/tools reads them, so that exporting one in
    # a shell does not make a subsequent training run fail validation.
    RELAX_REPRO_ROLE = EnvProperty("RELAX_REPRO_ROLE", str, None)
    RELAX_REPRO_ONLY_LOAD_WEIGHT = EnvProperty("RELAX_REPRO_ONLY_LOAD_WEIGHT", str, "1")
    RELAX_REPRO_PROFILE = EnvProperty("RELAX_REPRO_PROFILE", str, "0")
    RELAX_REPRO_PROFILE_DIR = EnvProperty("RELAX_REPRO_PROFILE_DIR", str, None)
    RELAX_REPRO_PROFILE_EXIT_AFTER_DUMP = EnvProperty("RELAX_REPRO_PROFILE_EXIT_AFTER_DUMP", str, "0")
    RELAX_REPRO_PROFILE_STEPS = EnvProperty("RELAX_REPRO_PROFILE_STEPS", int, 50)
    RELAX_REPRO_PROFILE_WARMUP = EnvProperty("RELAX_REPRO_PROFILE_WARMUP", int, 5)

    # ------------- NCCL / FP8 -------------
    NCCL_CUMEM_ENABLE = EnvProperty("NCCL_CUMEM_ENABLE", str, "0")
    NVTE_FP8_BLOCK_SCALING_FP32_SCALES = EnvProperty("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", str, "1")


def iter_env_properties() -> Iterator[Tuple[str, EnvProperty]]:
    """Yield every ``(attribute_name, binding)`` declared on
    :class:`Envs`."""
    for name, value in vars(Envs).items():
        if isinstance(value, EnvProperty):
            yield name, value


def known_env_names() -> FrozenSet[str]:
    """Every environment variable name declared on :class:`Envs`."""
    return frozenset(prop.env for _, prop in iter_env_properties())


def resolve_in(name: str, environment: Mapping[str, str]) -> Any:
    """Resolve the binding *name* against *environment* rather than
    ``os.environ``.

    Needed by :func:`validate_env`, which inspects the driver environment
    merged with variables that only exist in ``configs/env.yaml``: reading
    ``Envs.<NAME>`` there would silently miss anything supplied that way.
    """
    binding = vars(Envs).get(name)
    if not isinstance(binding, EnvProperty):
        raise KeyError(f"{name} is not a declared environment binding")
    return binding.parse(environment.get(binding.env))


def format_effective_config() -> str:
    """Render the resolved value of every declared variable, for
    diagnostics."""
    lines: List[str] = []
    for name, prop in sorted(iter_env_properties()):
        origin = "env" if is_env_set(prop.env) else "default"
        try:
            value: Any = mask_if_sensitive(prop.env, prop.resolve())
        except ValueError as exc:
            value = f"<invalid: {exc}>"
        lines.append(f"  {name} = {value!r} ({origin})")
    return "Relax effective configuration:\n" + "\n".join(lines)


def validate_env(extra_env: Optional[Mapping[str, str]] = None) -> None:
    """Check the environment at startup. Two problems, deliberately handled
    with different severities.

    1. A *declared* variable whose value cannot be parsed raises. Resolution is
       lazy, so a bad value would otherwise surface at an arbitrary later point
       — inside a hot loop, or at the import of
       ``utils/opd/opd_sglang_patch.py`` — rather than at startup. This one is
       fatal because the run will genuinely misbehave: ``RELAX_TGD_PROFILE=ture``
       is not profiling anything.
    2. A variable under :data:`OWNED_ENV_PREFIXES` that no :class:`EnvProperty`
       declares is only *warned* about. It usually means a knob was added to
       ``RUNTIME_ENV_JSON`` that no code reads (or an existing name was
       mistyped), which is a code-hygiene problem rather than a reason to
       refuse to start — plenty of scripts carry historical exports. Set
       ``RELAX_STRICT_ENV=1`` to promote it to an error in CI, or list known-good
       names in ``RELAX_EXTRA_ENV_ALLOWLIST`` (comma-separated) to keep the
       warning meaningful — plugins loaded via ``RELAX_EXTRA_MODULES`` own their
       own ``RELAX_*`` knobs and legitimately land here.

    ``extra_env`` carries variables destined for Ray workers but absent from the
    driver's own environment (``configs/env.yaml``), so they are checked before
    any worker is created.
    """
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)

    environment: Dict[str, str] = dict(os.environ)
    if extra_env:
        environment.update({str(key): str(value) for key, value in extra_env.items()})

    # Parse declared variables first, so that reading the settings below cannot
    # itself raise an unreported error.
    invalid: List[str] = []
    for _, prop in sorted(iter_env_properties()):
        try:
            prop.parse(environment.get(prop.env))
        except ValueError as exc:
            invalid.append(str(exc))
    if invalid:
        raise ValueError("Invalid environment variable values:\n  - " + "\n  - ".join(invalid))

    # Both settings are read out of `environment`, not `Envs`: they may have
    # been supplied through `configs/env.yaml` and so exist only in `extra_env`,
    # never in the driver's own os.environ.
    allowlist_raw = resolve_in("RELAX_EXTRA_ENV_ALLOWLIST", environment)
    allowlist = {name.strip() for name in allowlist_raw.split(",") if name.strip()}
    known = known_env_names()
    unknown = sorted(
        name
        for name in environment
        if name.startswith(OWNED_ENV_PREFIXES)
        and not name.startswith(IGNORED_ENV_PREFIXES)
        and name not in known
        and name not in allowlist
    )
    if not unknown:
        return

    message = (
        f"Unregistered environment variables in a Relax-owned namespace: {unknown}. "
        f"Nothing reads them, so they have no effect. Declare each one as an EnvProperty in "
        f"relax/utils/env.py::Envs so its type and default live alongside the rest of "
        f"the configuration, and read it via Envs.<NAME> rather than os.environ. "
        f"Silence known-good names with RELAX_EXTRA_ENV_ALLOWLIST."
    )
    if resolve_in("RELAX_STRICT_ENV", environment):
        raise ValueError(message)
    logger.warning(message)
