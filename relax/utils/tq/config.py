# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build TransferQueue backend config dicts from Relax CLI args.

This module is the single place that maps Relax-side *intent* flags
(``--tq-rdma-mode``, ``--tq-rdma-device``) into the OmegaConf dict that
``tq.init`` expects, and the only place that validates the configuration
preconditions for MooncakeStore.  Actual host-RDMA capability is established by
the real attach handshake in :mod:`relax.utils.tq.lifecycle`, not here.

Mooncake internals (endpoint, buffer size, segment size, master address,
timeout) are intentionally *not* exposed as CLI flags — they come from
internal defaults or the deployment environment, per maintainer guidance.
"""

from __future__ import annotations

import math
import os
from typing import Any

from relax.utils.tq.correctness import ensure_mooncake_correctness_guards


# Accepted ``--tq-rdma-mode`` values; ``off`` keeps the SimpleStorage path.
TQ_RDMA_MODES = frozenset({"off", "auto", "required"})

# ---------------------------------------------------------------------------
# Defaults (kept here rather than in config.yaml so they are visible to
# Relax contributors without reading the TQ package).
# ---------------------------------------------------------------------------

_DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024**3  # 4 GiB per client (config.yaml:52)
_DEFAULT_LOCAL_BUFFER_SIZE = 1 * 1024**3  # 1 GiB per client (config.yaml:54)
_DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"  # config.yaml:42-43


def validate_config(args: Any) -> list[str]:
    """Return error messages for an invalid RDMA configuration.

    Called at startup before anything is initialised.  An empty list means the
    requested mode is structurally valid; whether host RDMA actually works is
    decided later by the real attach handshake.  ``argparse`` already
    constrains the mode for CLI runs, so this also covers configs restored from
    a checkpoint or built programmatically.
    """
    errors: list[str] = []
    mode = getattr(args, "tq_rdma_mode", "off")
    device = getattr(args, "tq_rdma_device", "")

    if not isinstance(mode, str) or mode not in TQ_RDMA_MODES:
        errors.append(f"--tq-rdma-mode must be one of {', '.join(sorted(TQ_RDMA_MODES))}.")
    if not isinstance(device, str):
        errors.append("--tq-rdma-device must be a string.")
    elif device and any(character.isspace() or not character.isprintable() for character in device):
        errors.append("--tq-rdma-device must be empty or a single printable device name without whitespace.")
    return errors


def _split_host_port(address: str) -> tuple[str, int]:
    """Parse ``host:port`` (and bracketed IPv6) or raise ``ValueError``.

    Format-only: no DNS resolution and no connection attempt, so this stays a
    configuration check rather than becoming another capability probe.

    Failure messages name the *kind* of defect and never echo ``address``: the
    caller logs them, and a deployment endpoint is internal infrastructure
    detail that must not leak into job logs.
    """
    value = address.strip()
    if value.startswith("["):
        end = value.find("]")
        if end < 0 or end + 2 > len(value) or value[end + 1] != ":":
            raise ValueError("bracketed endpoint must be [host]:port")
        host, port_text = value[1:end], value[end + 2 :]
    else:
        host, separator, port_text = value.rpartition(":")
        if not separator:
            raise ValueError("endpoint must be host:port")
        if ":" in host:
            # A bare IPv6 literal: rpartition would silently treat its last
            # group as the port (``fe80::1`` -> host ``fe80:``, port 1).
            raise ValueError("IPv6 endpoint must be bracketed as [host]:port")
    if not host:
        raise ValueError("endpoint has an empty host")
    if any(character.isspace() or not character.isprintable() for character in host):
        raise ValueError("endpoint host must not contain whitespace or control characters")
    if not port_text.isdigit():
        raise ValueError("endpoint port must be a decimal number")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError("endpoint port is outside 1-65535")
    return host, port


def resolve_mooncake_master_address() -> str:
    """Return the externally managed Mooncake master endpoint.

    ``MC_MASTER_ADDRESS`` is required and must parse as ``host:port``.  A
    loopback default would make every node of a multi-node job treat its own
    localhost as the master, so ``auto`` would degrade and ``required`` would
    abort even when a shared master is healthy elsewhere.
    """
    address = os.environ.get("MC_MASTER_ADDRESS", "").strip()
    if not address:
        raise RuntimeError(
            "MooncakeStore requires MC_MASTER_ADDRESS=<host:port> of the externally "
            "managed mooncake master in the driver environment; Relax never assumes "
            "a loopback endpoint."
        )
    try:
        _split_host_port(address)
    except ValueError as error:
        raise RuntimeError(f"MC_MASTER_ADDRESS is not a usable endpoint: {error}") from error
    return address


def resolve_global_segment_size() -> int:
    """Per-client Mooncake segment size in bytes.

    Defaults to 4 GiB (TQ config.yaml:52).  Deployments whose worst-case in-
    flight payload exceeds that (see :func:`estimate_payload_bytes`) set
    ``RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB`` instead of editing code; capacity
    validation and the client config read the same value so the check can never
    pass a size the client does not actually mount.
    """
    raw = os.environ.get("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "").strip()
    if not raw:
        return _DEFAULT_GLOBAL_SEGMENT_SIZE
    try:
        gib = float(raw)
    except ValueError:
        raise RuntimeError("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB must be a finite positive number of GiB") from None
    if not math.isfinite(gib) or gib <= 0:
        raise RuntimeError("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB must be a finite positive number of GiB")
    size_bytes = int(gib * 1024**3)
    if size_bytes <= 0:
        raise RuntimeError("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB must resolve to at least one byte")
    return size_bytes


def validate_mooncake_runtime_contract() -> None:
    """Validate the Relax-side portion of the Mooncake safety contract.

    The environment guard and versioned correctness-contract check run before
    every Mooncake client is created or attached. This function does not
    monkey-patch TransferQueue.
    """
    ensure_mooncake_correctness_guards()


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_simple_storage_config(total_storage_size: int | None, num_data_storage_units: int) -> dict[str, Any]:
    """Build the SimpleStorage backend config.

    ``total_storage_size=None`` preserves TransferQueue's unlimited-capacity
    semantics. Relax uses it for production too because agent/tool fan-out can
    make physical row counts exceed the logical rollout batch size.
    """
    return {
        # ``tq.init`` selects the manager from this key alone (TQ config.yaml:22
        # defaults it to SimpleStorage); a backend section without it is ignored.
        "storage_backend": "SimpleStorage",
        "SimpleStorage": {
            "total_storage_size": total_storage_size,
            "num_data_storage_units": num_data_storage_units,
        },
    }


def build_mooncake_config(
    *,
    master_address: str,
    device: str = "",
    protocol: str = "rdma",
    global_segment_size: int | None = None,
) -> dict[str, Any]:
    """Build the ``backend`` dict for MooncakeStore.

    Parameters
    ----------
    master_address
        External master server address. It is validated here as well as by
        :func:`resolve_mooncake_master_address` so direct benchmark callers
        cannot bypass the format contract.
    device
        Explicit RDMA device name; empty lets Mooncake select one natively.
    protocol
        Transport. Production only ever builds ``rdma``; ``tcp`` exists solely
        for the C1 benchmark baseline.
    global_segment_size
        Override the per-client segment size.  ``None`` resolves from
        ``RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB`` (default 4 GiB, see
        :func:`resolve_global_segment_size`).  Benchmarks may pass a larger
        value (e.g. 8 GiB) to avoid staging-buffer pressure.
    """
    if not isinstance(master_address, str):
        raise ValueError("master_address must be a host:port string")
    try:
        _split_host_port(master_address)
    except ValueError as error:
        raise ValueError(f"master_address is not a usable endpoint: {error}") from None
    master_address = master_address.strip()

    if global_segment_size is None:
        segment_size = resolve_global_segment_size()
    elif isinstance(global_segment_size, bool) or not isinstance(global_segment_size, int) or global_segment_size <= 0:
        raise ValueError("global_segment_size must be a positive integer number of bytes")
    else:
        segment_size = global_segment_size

    cfg: dict[str, Any] = {
        # Selects the manager inside ``tq.init`` (interface.py reads
        # ``backend.storage_backend``, TQ config.yaml:22 defaults it to
        # SimpleStorage).  Without this key the MooncakeStore section below is
        # parsed and then silently ignored -- the job still runs on ZMQ/TCP.
        "storage_backend": "MooncakeStore",
        "MooncakeStore": {
            # Transport
            "protocol": protocol,
            "device_name": device,
            # Master / metadata — externally managed, never auto-init.
            "auto_init": False,
            "master_server_address": master_address,
            "metadata_server": _DEFAULT_METADATA_SERVER,
            "local_hostname": "",  # empty = auto-detect via Ray node IP
            # Memory
            "global_segment_size": segment_size,
            "local_buffer_size": _DEFAULT_LOCAL_BUFFER_SIZE,
            # Do NOT silently evict produced-but-unconsumed data.
            "hard_pin": True,
            # Host RDMA only in this phase: GDR is pinned off here rather than
            # left to the upstream default so the payload always traverses host
            # memory.  A GDR path needs its own probe/verification and lands in
            # a separate change (mooncake_client.py:75-88 for the client side).
            "use_gdr": False,
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# Capacity validation
# ---------------------------------------------------------------------------


# Worst-case payload factors used by the segment-capacity pre-check.
#
# The largest image/video tensor layout among the currently supported
# multimodal processors is Qwen3-VL's 16x16 spatial patch, temporal patch 2,
# RGB input, spatial merge 2x2, stored as float32.  One schedulable vision
# token can therefore retain four flattened patch rows:
#
#   16 * 16 * 2 * 3 * 4 rows/token * 4 bytes/value = 24,576 bytes/token
#
# Keep this as one explicit bound instead of loading model config during
# Controller startup.  If a supported processor gains a wider transported
# feature row, this bound and its regression test must be updated together.
_MULTIMODAL_BYTES_PER_TOKEN = 16 * 16 * 2 * 3 * (2 * 2) * 4
# Text: token ids, logprobs, masks and rewards; 32 B/token rounds them up.
_TEXT_BYTES_PER_TOKEN = 32


def resolve_tq_capacity_batch_size(args: Any) -> int:
    """Return the largest batch that one TQ step may need to hold.

    Dynamic partial rollout schedules from the over-sampling pool, so its
    capacity contract is ``over_sampling_batch_size`` rather than the smaller
    nominal ``rollout_batch_size``. This batch size is used to estimate
    MooncakeStore's payload capacity.
    """
    rollout_batch = getattr(args, "rollout_batch_size")
    if getattr(args, "partial_rollout", False) and getattr(args, "use_dynamic_global_batch_size", False):
        over_sampling_batch = getattr(args, "over_sampling_batch_size", None)
        if over_sampling_batch is not None:
            return int(over_sampling_batch)
    return int(rollout_batch)


def estimate_payload_bytes(args: Any) -> int:
    """Worst-case per-step payload upper bound in bytes.

    Derived from the token budget instead of a fixed per-sample constant: the
    processor cannot emit more multimodal tokens than ``--seq-length`` allows,
    so the transported tensor payload of one sample is bounded by
    ``seq_length * _MULTIMODAL_BYTES_PER_TOKEN`` (192 MiB at
    ``seq_length=8192``) rather than a fixed per-sample guess that can pass
    configurations which later fail puts mid-training.
    """
    n_samples = args.n_samples_per_prompt
    capacity_batch = resolve_tq_capacity_batch_size(args)
    seq_length = int(getattr(args, "seq_length", 0) or 0)
    if seq_length <= 0:
        raise RuntimeError(
            "MooncakeStore segment-capacity validation needs args.seq_length to bound "
            "the per-sample payload; got a missing or non-positive value."
        )
    per_sample = seq_length * _TEXT_BYTES_PER_TOKEN
    if getattr(args, "multimodal_keys", None) is not None:
        per_sample += seq_length * _MULTIMODAL_BYTES_PER_TOKEN
    return capacity_batch * n_samples * per_sample


def validate_segment_capacity(args: Any) -> str | None:
    """Return an error message if segment capacity is insufficient, else None.

    Only used for MooncakeStore. Compares the per-client segment size
    (``global_segment_size``) against the estimated in-flight payload.
    """
    max_staleness = getattr(args, "max_staleness", 0)
    capacity_batch = resolve_tq_capacity_batch_size(args)
    payload = estimate_payload_bytes(args)
    needed = payload * (max_staleness + 1)
    available = resolve_global_segment_size()

    if needed > available:
        return (
            f"MooncakeStore segment capacity insufficient: worst-case in-flight payload "
            f"{needed / 1024**3:.1f} GiB (effective_batch={capacity_batch} × "
            f"n_samples={args.n_samples_per_prompt} × staleness+1={max_staleness + 1}) "
            f"exceeds global_segment_size {available / 1024**3:.1f} GiB. "
            f"Reduce batch size / max_staleness, or raise RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB."
        )
    return None


# ---------------------------------------------------------------------------
# Top-level resolver
# ---------------------------------------------------------------------------


def build_backend_config(
    args: Any,
    *,
    device: str,
    master_address: str,
) -> tuple[dict[str, Any], str | None]:
    """Return ``(backend_config_dict, error_or_none)`` for the host-RDMA path.

    ``master_address`` must already be validated by
    :func:`resolve_mooncake_master_address`; it is threaded through so the
    checked value is the one the client receives. On a capacity error, return
    an empty config and the error; the caller handles logging and decides
    whether to build a fallback or fail.
    """
    cap_error = validate_segment_capacity(args)
    if cap_error:
        return {}, cap_error

    return build_mooncake_config(master_address=master_address, device=device), None
