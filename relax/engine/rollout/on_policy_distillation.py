# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

import aiohttp
import numpy as np

from relax.engine.inference.client import InferenceClient
from relax.engine.inference.routing import candidate_replicas, select_replica
from relax.utils.logging_utils import get_logger
from relax.utils.opd import opd_main_worker, opd_opsd_worker
from relax.utils.types import Sample


try:
    import orjson

    def _dumps_to_bytes(payload: dict) -> bytes:
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
except ImportError:  # pragma: no cover - orjson is normally available via sglang

    def _dumps_to_bytes(payload: dict) -> bytes:
        return json.dumps(payload).encode("utf-8")


logger = get_logger(__name__)

EncodeMultimodalInputs = Callable[[dict], Awaitable[tuple[dict, float]]]


def _aiohttp_json_post_kwargs(payload: dict) -> dict:
    """Bypass aiohttp's ``json=`` kwarg: serialize via orjson and ship as raw
    ``data=``."""
    return {
        "data": _dumps_to_bytes(payload),
        "headers": {"Content-Type": "application/json"},
    }


def is_opd_enabled(args, evaluation: bool = False) -> bool:
    return not evaluation and getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"


def _create_teacher_client_session(args) -> aiohttp.ClientSession:
    connector_limit = int(getattr(args, "opd_teacher_connector_limit", 256))
    connector = aiohttp.TCPConnector(limit=connector_limit)
    timeout = aiohttp.ClientTimeout(total=float(args.opd_teacher_timeout_s))
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


# --- Teacher URL selection: MOPD routing (by data_source) x replica round-robin ---
# Two layers:
#   1. Routing (MOPD): when ``args.opd_teacher_routes_map`` is set, pick the
#      teacher for this sample by ``sample.metadata[args.opd_teacher_key]``.
#      Each route value is a LIST of that teacher's replica URLs.
#   2. Replica round-robin: spread requests across a teacher's replicas with a
#      per-teacher in-process counter (single-threaded asyncio makes ``+= 1`` safe).
#      The counter advances once per GRPO *group*, not once per sample, so that a
#      group's shared prompt prefix is prefilled by one replica -- see
#      ``_pick_replica``.
# Single-teacher path falls back to ``args.opd_teacher_urls`` / ``opd_teacher_url``.
#
# Which replicas a managed teacher has is read from its gateway's discovery
# endpoint when ``args.opd_teacher_discovery_url`` is set, so a replica rebuilt
# at another address is picked up without a restart. The URLs captured in
# ``args`` at startup remain the fallback, and the only source for an external
# teacher (``--opd-teacher-url``).
_TEACHER_URL_RR: dict[str, int] = {}
# (teacher key, group_index) -> replica ordinal. Bounded by clearing wholesale:
# this is a pure cache-locality heuristic, so a dropped entry only costs prefix
# reuse for a group still in flight, never correctness.
_TEACHER_GROUP_REPLICA: dict[tuple[str, int], int] = {}
_MAX_TEACHER_GROUP_REPLICA = 1 << 16


# Model name the single managed teacher is published under by its gateway.
_SINGLE_TEACHER_MODEL = "__default__"
# How long a discovered topology is used before it is fetched again, and the
# minimum spacing of fetches triggered by failing teacher requests.
_TEACHER_DISCOVERY_MAX_AGE_S = 5.0
_TEACHER_DISCOVERY_COOLDOWN_S = 1.0
_TEACHER_DISCOVERY_TIMEOUT_S = 2.0
# discovery URL -> client caching that teacher role's topology.
_TEACHER_TOPOLOGY: dict[str, InferenceClient] = {}
_TEACHER_TOPOLOGY_REFRESHING: set[str] = set()
# Discovery URLs already reported unreachable, so the warning is not repeated per batch.
_TEACHER_DISCOVERY_WARNED: set[str] = set()


def _fetch_teacher_topology(discovery_url: str) -> dict:
    import requests

    response = requests.get(discovery_url, timeout=_TEACHER_DISCOVERY_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def _teacher_topology(args) -> InferenceClient | None:
    discovery_url = getattr(args, "opd_teacher_discovery_url", None)
    if not discovery_url:
        return None
    client = _TEACHER_TOPOLOGY.get(discovery_url)
    if client is None:
        client = InferenceClient(
            lambda: _fetch_teacher_topology(discovery_url),
            max_age_s=_TEACHER_DISCOVERY_MAX_AGE_S,
            refresh_cooldown_s=_TEACHER_DISCOVERY_COOLDOWN_S,
        )
        _TEACHER_TOPOLOGY[discovery_url] = client
    return client


async def _refresh_teacher_topology(args) -> None:
    """Bring the managed teacher's topology up to date, off the event loop.

    Teacher URLs are picked synchronously per sample, so the picking itself
    only reads the cached topology; this is where it is fetched. Concurrent
    callers share one fetch and never wait for it.
    """
    client = _teacher_topology(args)
    if client is None:
        return
    discovery_url: str = args.opd_teacher_discovery_url
    if discovery_url in _TEACHER_TOPOLOGY_REFRESHING or not client.needs_refresh():
        return
    _TEACHER_TOPOLOGY_REFRESHING.add(discovery_url)
    try:
        await asyncio.to_thread(client.snapshot)
        _TEACHER_DISCOVERY_WARNED.discard(discovery_url)
    except Exception as exc:
        if discovery_url not in _TEACHER_DISCOVERY_WARNED:
            _TEACHER_DISCOVERY_WARNED.add(discovery_url)
            logger.warning(
                "OPD teacher discovery at %s is unreachable; using the teacher URLs captured at startup: %s",
                discovery_url,
                f"{type(exc).__name__}: {str(exc)[:256]}",
            )
    finally:
        _TEACHER_TOPOLOGY_REFRESHING.discard(discovery_url)


def _discovered_replicas(args, model: str) -> list[str] | None:
    """Replica URLs of one managed teacher as last discovered.

    ``None`` when there is nothing to go on -- no discovery configured, no
    topology fetched yet, or no replica reported ready -- in which case the
    caller keeps the URLs from ``args``.
    """
    client = _teacher_topology(args)
    snapshot = client.last_snapshot if client is not None else None
    model_snapshot = snapshot.model(model) if snapshot is not None else None
    if model_snapshot is None:
        return None
    return [f"{replica.base_url}/generate" for replica in candidate_replicas(model_snapshot)] or None


def _report_teacher_failure(args) -> None:
    """A teacher request failed: look at the topology again before the next
    batch, in case the replica was rebuilt elsewhere."""
    client = _teacher_topology(args)
    if client is not None:
        client.report_failure()


def _pick_replica(replicas: list[str], sample, rr_key: str) -> str:
    """Load-balance across ONE teacher's replicas at GRPO-group granularity.

    ``rr_key`` already identifies the teacher — the caller resolved
    ``data_source`` to a replica list first — so this only decides which copy of
    that same model serves the request. Text/VL teacher routing is unaffected.

    The ``n_samples_per_prompt`` samples of a prompt share ``group_index``
    (assigned in ``engine/rollout/data_source.py``) and therefore share the whole
    prompt prefix, which for multi-image data is tens of thousands of tokens.
    Advancing the per-teacher cursor once per group instead of once per sample
    keeps a group on one replica, so that replica's radix cache serves the prefix
    once rather than every replica re-prefilling it.

    Deliberately NOT ``group_index % len(replicas)``: ``group_index`` is a global
    counter that does not restart per ``data_source``, so with an interleaved
    multi-source dataset one teacher sees only every k-th index and the modulo
    collapses onto a subset of its replicas (e.g. an alternating text/VL dataset
    gives one teacher only even indices, which all map to replica 0).
    """
    group_index = getattr(sample, "group_index", None) if sample is not None else None
    # The rule itself is shared with the inference gateway; the cursor and the
    # group memory stay here, per process.
    return select_replica(
        replicas,
        cursor_key=rr_key,
        cursors=_TEACHER_URL_RR,
        affinity_key=int(group_index) if group_index is not None else None,
        affinity_memory=_TEACHER_GROUP_REPLICA,
        affinity_memory_cap=_MAX_TEACHER_GROUP_REPLICA,
    )


def _pick_teacher_url(args, sample=None) -> str:
    routes_map = getattr(args, "opd_teacher_routes_map", None)
    if routes_map and sample is not None:
        key_field = getattr(args, "opd_teacher_key", None) or "data_source"
        metadata = getattr(sample, "metadata", None) or {}
        routing_value = metadata.get(key_field)
        if routing_value is None:
            raise ValueError(
                f"MOPD routing: sample missing key '{key_field}' in metadata. "
                f"Available metadata keys: {list(metadata.keys())}. "
                f"Ensure the dataset has a '{key_field}' column and it is surfaced "
                "via --metadata-key or the data pipeline."
            )
        replicas = routes_map.get(routing_value)
        if not replicas:
            raise KeyError(
                f"MOPD routing: no teacher route for '{key_field}={routing_value}'. "
                f"Available routes: {list(routes_map.keys())}."
            )
        replicas = _discovered_replicas(args, routing_value) or replicas
        return _pick_replica(replicas, sample, routing_value)
    # Single-teacher path: round-robin over replicas if configured.
    discovered = _discovered_replicas(args, _SINGLE_TEACHER_MODEL)
    urls = discovered or getattr(args, "opd_teacher_urls", None)
    if urls and len(urls) > 1:
        return _pick_replica(urls, sample, "__single__")
    return discovered[0] if discovered else args.opd_teacher_url


class OpdManager:
    def __init__(self, args):
        self.args = args
        self.topk_worker: opd_main_worker.TopkWorker | None = None
        self.sampled_worker: opd_main_worker.SampledTokenWorker | None = None  # 仅 student_sampled
        self.opsd_worker: opd_opsd_worker.OpsdWorker | None = None

        token_selection = args.opd_token_selection
        if token_selection != "student_sampled":
            self.topk_worker = opd_main_worker.TopkWorker.from_args(args)
        else:
            self.sampled_worker = opd_main_worker.SampledTokenWorker.from_args(args)

        opsd_worker = opd_opsd_worker.OpsdWorker.from_args(args)
        self.opsd_worker = opsd_worker if opsd_worker.is_opsd else None

    @property
    def is_topk(self) -> bool:
        return self.topk_worker is not None

    @property
    def is_opsd(self) -> bool:
        return self.opsd_worker is not None

    def schema_opd_transfer_data(self) -> list[str]:
        fields: list[str] = []
        if self.topk_worker is not None:
            fields.extend(self.topk_worker.topk_transfer_fields())
        if self.sampled_worker is not None:
            fields.extend(self.sampled_worker.sampled_transfer_fields())
        return fields

    def produce_opd_transfer_data(self, samples: list[Sample], train_data: dict) -> None:
        if self.topk_worker is not None:
            for field_name in opd_main_worker.TopkWorker.TRANSFER_FIELDS:
                if not any(getattr(s, field_name, None) is not None for s in samples):
                    continue
                flat: list = []
                for s in samples:
                    v = getattr(s, field_name, None)
                    if v is None:
                        flat.append([])
                    else:
                        flat.append(v.reshape(-1).tolist())
                train_data[field_name] = flat
            kl_field = opd_main_worker.TopkWorker.TRANSFER_K_LENGTHS
            if self.topk_worker.spec.name == "union" and any(getattr(s, kl_field, None) is not None for s in samples):
                train_data[kl_field] = [
                    getattr(s, kl_field).tolist() if getattr(s, kl_field, None) is not None else [] for s in samples
                ]
        elif self.sampled_worker is not None:
            train_data[opd_main_worker.SampledTokenWorker.TRANSFER_TEACHER_LOG_PROBS] = [
                s.teacher_log_probs if s.teacher_log_probs is not None else [] for s in samples
            ]

    def before_rollout(self, payload: dict) -> None:
        if self.topk_worker is None:
            return
        fields = self.topk_worker.student_rollout_payload()
        if fields:
            payload.update(fields)

    def parse_rollout_logprobs(self, meta_info: dict, tokens: list, log_probs: list) -> tuple[list, list]:
        if self.topk_worker is None:
            return tokens, log_probs
        val_b64 = meta_info.get("output_token_logprobs_val_b64")
        if val_b64 is None:
            return tokens, log_probs
        import numpy as np
        import pybase64

        val = np.frombuffer(pybase64.b64decode(val_b64), dtype=np.float32)
        idx_b64 = meta_info.get("output_token_logprobs_idx_b64")
        idx = np.frombuffer(pybase64.b64decode(idx_b64), dtype=np.int32) if idx_b64 else np.array([], dtype=np.int32)
        n_expected = len(tokens)
        if idx.size == n_expected and val.size == n_expected:
            out_tokens = idx.tolist()
            out_log_probs = val.tolist()
        else:
            out_tokens = tokens
            out_log_probs = log_probs
        return out_tokens, out_log_probs

    def after_rollout(self, sample: Sample, output: dict) -> None:
        if self.topk_worker is None:
            return
        pair = opd_main_worker.LogprobResponse(output).self_topk("rollout", self.topk_worker.top_k)
        if pair is None:
            return
        token_ids, log_probs = pair
        if sample.student_topk_token_ids is None:
            sample.student_topk_token_ids = token_ids
            sample.student_topk_log_probs = log_probs
        else:  # multi-turn
            sample.student_topk_token_ids = np.vstack([sample.student_topk_token_ids, token_ids])
            sample.student_topk_log_probs = np.vstack([sample.student_topk_log_probs, log_probs])

    async def prefill(
        self,
        samples: Sample | Sequence[Sample],
        encode_multimodal_inputs: EncodeMultimodalInputs | None = None,
    ) -> None:
        """Run the three stages back to back, as generation does inline."""
        sample_list = list(samples) if isinstance(samples, Sequence) else [samples]

        async with _create_teacher_client_session(self.args) as session:
            await self._run_teacher_stage(sample_list, session)
            if self.needs_student_stage:
                await self._run_student_stage(sample_list, session, encode_multimodal_inputs)

        self.assemble_stage(sample_list)

    # The stages are also run one at a time by deferred scoring, which has to
    # switch GPU memory between them: the teacher stage only needs the teacher,
    # the student stage only rollout.

    @property
    def needs_student_stage(self) -> bool:
        """Whether the student must score the teacher's top-k tokens."""
        return self.topk_worker is not None and self.topk_worker.spec.student_at_teacher

    async def teacher_stage(self, samples: list[Sample], session: aiohttp.ClientSession | None = None) -> None:
        """Ask the teacher for its log-probs on every sample.

        Raises:
            RuntimeError: every sample with a response failed.
        """
        if session is not None:
            await self._run_teacher_stage(samples, session)
            return
        async with _create_teacher_client_session(self.args) as own_session:
            await self._run_teacher_stage(samples, own_session)

    async def _run_teacher_stage(self, samples: list[Sample], session: aiohttp.ClientSession) -> None:
        if self.opsd_worker is not None:
            await asyncio.gather(*[self.opsd_worker.build_teacher_inputs(self.args, s) for s in samples])

        await _refresh_teacher_topology(self.args)
        fetch_results = await asyncio.gather(*[self._teacher_prefill(s, session) for s in samples])
        if not all(fetch_results):
            _report_teacher_failure(self.args)
        self._raise_if_all_failed(samples, fetch_results)

    async def student_stage(
        self,
        samples: list[Sample],
        session: aiohttp.ClientSession | None = None,
        encode_multimodal_inputs: EncodeMultimodalInputs | None = None,
    ) -> None:
        """Ask the student (rollout) for its log-probs on the teacher's top-k
        tokens; a no-op unless the token selection needs it."""
        if not self.needs_student_stage:
            return
        if session is not None:
            await self._run_student_stage(samples, session, encode_multimodal_inputs)
            return
        async with _create_teacher_client_session(self.args) as own_session:
            await self._run_student_stage(samples, own_session, encode_multimodal_inputs)

    async def _run_student_stage(
        self,
        samples: list[Sample],
        session: aiohttp.ClientSession,
        encode_multimodal_inputs: EncodeMultimodalInputs | None,
    ) -> None:
        await asyncio.gather(*[self._student_prefill(s, session, encode_multimodal_inputs) for s in samples])

    def assemble_stage(self, samples: list[Sample]) -> None:
        """Combine what the two models returned into the fields training
        reads."""
        self._assemble_transfer(samples)

    async def _post_logprob(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        sample: Sample,
        err_tag: str,
    ) -> opd_main_worker.LogprobResponse | None:
        try:
            async with session.post(url, **_aiohttp_json_post_kwargs(payload)) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("OPD %s failed: status=%s, url=%s, body=%s", err_tag, resp.status, url, body[:2048])
                    resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logger.error(
                "OPD %s fetch failed for sample_index=%s, error=%s",
                err_tag,
                getattr(sample, "index", None),
                f"{type(exc).__name__}: {str(exc)[:256]}",
            )
            return None

        return opd_main_worker.LogprobResponse(data)

    async def _teacher_prefill(self, sample: Sample, session: aiohttp.ClientSession) -> bool:
        from relax.utils.opd.opd_utils import build_teacher_preexpanded_image_data

        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            return True

        # OPSD: expanded teacher_tokens + image_data；
        if self.opsd_worker is not None:
            image_data = await build_teacher_preexpanded_image_data(sample)
            teacher_input_ids = self.opsd_worker.teacher_input_ids(sample, response_length)
            prompt_length = self.opsd_worker.teacher_prompt_len(sample, response_length)
        else:
            image_data = None
            teacher_input_ids = sample.rollout_tokens or sample.tokens
            prompt_length = len(sample.tokens) - response_length
        logprob_start_len = max(prompt_length - 1, 0)

        mm_fields = {"image_data": image_data} if image_data is not None else None
        if self.topk_worker is not None:
            payload = self.topk_worker.build_teacher_payload(
                input_ids=teacher_input_ids,
                logprob_start_len=logprob_start_len,
                student_topk_ids=sample.student_topk_token_ids,
                response_length=response_length,
                mm_fields=mm_fields,
            )
        else:
            payload = opd_main_worker.build_prefill_payload_base(teacher_input_ids, logprob_start_len)
            if mm_fields:
                payload.update(mm_fields)

        teacher_url = _pick_teacher_url(self.args, sample)
        resp_obj = await self._post_logprob(session, teacher_url, payload, sample, "teacher prefill")
        if resp_obj is None:
            return False

        token_logprobs = resp_obj.base_logprobs_1d()
        if token_logprobs is None or len(token_logprobs) == 0:
            logger.error(
                "Invalid OPD teacher response for sample_index=%s: missing input_token_logprobs.",
                getattr(sample, "index", None),
            )
            return False

        if len(token_logprobs) != response_length + 1:
            logger.error(
                "Teacher log-prob length mismatch for sample_index=%s: got=%s expected=%s",
                getattr(sample, "index", None),
                len(token_logprobs) - 1,
                response_length,
            )
            return False

        if self.sampled_worker is not None:
            sample.teacher_log_probs = [float(v) for v in token_logprobs[1 : 1 + response_length]]

        if self.topk_worker is not None:
            if self.topk_worker.spec.teacher_self_topk:
                pair = self.topk_worker.parse_prefill_self_topk(resp_obj, response_length)
                sample.teacher_topk_token_ids = pair[0] if pair else None
                sample.teacher_topk_log_probs = pair[1] if pair else None
            if self.topk_worker.spec.teacher_at_student:
                sample.teacher_at_student_topk_log_probs = self.topk_worker.parse_prefill_other_topk(
                    resp_obj, response_length
                )

        return True

    async def _student_prefill(
        self, sample: Sample, session: aiohttp.ClientSession, encode_mm_fn: EncodeMultimodalInputs | None
    ) -> None:

        from relax.utils.opd.opd_utils import build_student_preexpanded_image_data

        response_length = int(sample.response_length or 0)
        teacher_topk_ids = sample.teacher_topk_token_ids

        prompt_length = len(sample.tokens) - response_length
        logprob_start_len = max(prompt_length - 1, 0)

        preexpanded_image_data = await build_student_preexpanded_image_data(sample)
        if preexpanded_image_data is not None:
            student_input_ids = sample.tokens
            mm_fields = {"image_data": preexpanded_image_data}
        else:
            student_input_ids = sample.rollout_tokens or sample.tokens
            mm_fields = None
            if encode_mm_fn is not None and sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
                mm_fields, _ = await encode_mm_fn(sample.multimodal_inputs)

        payload = self.topk_worker.build_student_payload(
            input_ids=student_input_ids,
            logprob_start_len=logprob_start_len,
            teacher_topk_ids=teacher_topk_ids,
            response_length=response_length,
            mm_fields=mm_fields,
        )

        student_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
        resp_obj = await self._post_logprob(session, student_url, payload, sample, "student-at-teacher-topk")
        if resp_obj is None:
            return
        sample.student_at_teacher_topk_log_probs = self.topk_worker.parse_prefill_other_topk(resp_obj, response_length)

    def _assemble_transfer(self, samples: list[Sample]) -> None:
        if self.topk_worker is None:
            return
        for sample in samples:
            student_self = (
                (sample.student_topk_token_ids, sample.student_topk_log_probs)
                if sample.student_topk_token_ids is not None
                else None
            )
            teacher_self = (
                (sample.teacher_topk_token_ids, sample.teacher_topk_log_probs)
                if sample.teacher_topk_token_ids is not None
                else None
            )
            channels = self.topk_worker.build_transfer_channels(
                student_self_topk=student_self,
                teacher_self_topk=teacher_self,
                teacher_at_student_lp=sample.teacher_at_student_topk_log_probs,
                student_at_teacher_lp=sample.student_at_teacher_topk_log_probs,
            )
            sample.opd_topk_token_ids = channels.get(opd_main_worker.TopkWorker.TRANSFER_TOKEN_IDS)
            sample.opd_topk_student_log_probs = channels.get(opd_main_worker.TopkWorker.TRANSFER_STUDENT_LOG_PROBS)
            sample.opd_topk_teacher_log_probs = channels.get(opd_main_worker.TopkWorker.TRANSFER_TEACHER_LOG_PROBS)
            sample.opd_topk_ksz = channels.get(opd_main_worker.TopkWorker.TRANSFER_K_LENGTHS)

    @staticmethod
    def _raise_if_all_failed(samples: list[Sample], fetch_results: list[bool]) -> None:
        eligible = [(s, ok) for s, ok in zip(samples, fetch_results) if int(getattr(s, "response_length", 0) or 0) > 0]
        if not eligible or any(ok for _, ok in eligible):
            return
        raise RuntimeError(
            f"All OPD teacher fetches failed for {len(eligible)} non-empty samples "
            f"(total={len(samples)}); url={getattr(samples[0], 'opd_teacher_url', None) if samples else None}"
        )


def produce_opd_transfer_data(args, samples: list[Sample], train_data: dict) -> None:
    if not is_opd_enabled(args):
        return
    OpdManager(args).produce_opd_transfer_data(samples, train_data)
