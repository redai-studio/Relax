# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

import aiohttp
import numpy as np

from relax.utils.logging_utils import get_logger
from relax.utils.opd import opd_main_worker, opd_opsd_worker
from relax.utils.opd.opd_multimodal import (
    encode_image_inputs,
    get_image_token_id,
    image_inputs,
    validate_image_input_ids,
)
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


def _replace_response_eos_for_teacher(
    input_ids: Sequence[int],
    *,
    prompt_length: int,
    response_length: int,
    student_eos_token_id: int | None,
    teacher_eos_token_id: int | None,
) -> list[int]:
    teacher_input_ids = list(input_ids)
    if (
        response_length <= 0
        or student_eos_token_id is None
        or teacher_eos_token_id is None
        or student_eos_token_id == teacher_eos_token_id
    ):
        return teacher_input_ids

    response_start = max(int(prompt_length), 0)
    response_end = min(response_start + int(response_length), len(teacher_input_ids))
    for token_index in range(response_start, response_end):
        if teacher_input_ids[token_index] == student_eos_token_id:
            teacher_input_ids[token_index] = teacher_eos_token_id
            break
    return teacher_input_ids


def _insert_teacher_response_prefix(
    input_ids: Sequence[int],
    *,
    prompt_length: int,
    prefix_token_ids: Sequence[int],
) -> tuple[list[int], int]:
    teacher_input_ids = list(input_ids)
    prefix_token_ids = list(prefix_token_ids)
    if not prefix_token_ids:
        return teacher_input_ids, int(prompt_length)

    insertion_index = min(max(int(prompt_length), 0), len(teacher_input_ids))
    teacher_input_ids[insertion_index:insertion_index] = prefix_token_ids
    return teacher_input_ids, insertion_index + len(prefix_token_ids)


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
_TEACHER_URL_RR: dict[str, int] = {}
# (teacher key, group_index) -> replica ordinal. Bounded by clearing wholesale:
# this is a pure cache-locality heuristic, so a dropped entry only costs prefix
# reuse for a group still in flight, never correctness.
_TEACHER_GROUP_REPLICA: dict[tuple[str, int], int] = {}
_MAX_TEACHER_GROUP_REPLICA = 1 << 16


def _round_robin(urls: list[str], key: str) -> str:
    i = _TEACHER_URL_RR.get(key, 0)
    _TEACHER_URL_RR[key] = i + 1
    return urls[i % len(urls)]


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
    if len(replicas) == 1:
        return replicas[0]
    group_index = getattr(sample, "group_index", None) if sample is not None else None
    if group_index is None:
        return _round_robin(replicas, rr_key)
    key = (rr_key, int(group_index))
    idx = _TEACHER_GROUP_REPLICA.get(key)
    if idx is None:
        if len(_TEACHER_GROUP_REPLICA) > _MAX_TEACHER_GROUP_REPLICA:
            _TEACHER_GROUP_REPLICA.clear()
        idx = _TEACHER_URL_RR.get(rr_key, 0)
        _TEACHER_URL_RR[rr_key] = idx + 1
        _TEACHER_GROUP_REPLICA[key] = idx
    return replicas[idx % len(replicas)]


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
        return _pick_replica(replicas, sample, routing_value)
    # Single-teacher path: round-robin over replicas if configured.
    urls = getattr(args, "opd_teacher_urls", None)
    if urls and len(urls) > 1:
        return _pick_replica(urls, sample, "__single__")
    return args.opd_teacher_url


class OpdManager:
    def __init__(self, args):
        self.args = args
        self.student_eos_token_id = getattr(args, "opd_student_eos_token_id", None)
        self.teacher_eos_token_id = getattr(args, "opd_teacher_eos_token_id", None)
        self.teacher_response_prefix_token_ids = tuple(
            getattr(args, "opd_teacher_response_prefix_token_ids", None) or ()
        )
        self.topk_worker: opd_main_worker.TopkWorker | None = None
        self.sampled_worker: opd_main_worker.SampledTokenWorker | None = None  # 仅 student_sampled
        self.opsd_worker: opd_opsd_worker.OpsdWorker | None = None

        token_selection = args.opd_token_selection
        if token_selection != "student_sampled":
            self.topk_worker = opd_main_worker.TopkWorker.from_args(args)
            if bool(getattr(args, "opd_fkl_entropy_gate", False)):
                # EOPD: sampled-token reverse-KL advantages ride alongside the
                # entropy-gated top-k forward-KL loss; both workers are active.
                self.sampled_worker = opd_main_worker.SampledTokenWorker.from_args(args)
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
        if self.sampled_worker is not None:
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
        native = bool(image_inputs(sample.multimodal_inputs) or image_inputs(sample.teacher_multimodal_inputs))
        response_ids = output.get("output_ids")
        if native:
            if response_ids is None:
                raise ValueError("Native image OPD requires output_ids from student rollout.")
            n_new = len(response_ids)
            if n_new and (
                sample.tokens[-n_new:] != response_ids
                or sample.rollout_tokens[-n_new:] != response_ids
                or len(sample.rollout_log_probs or []) != sample.response_length
                or not np.isfinite(sample.rollout_log_probs[-n_new:]).all()
            ):
                raise ValueError("Native image OPD student rollout IDs/logprobs do not match the sampled response.")
            if not n_new:
                return
        if self.topk_worker is None or not self.topk_worker.spec.student_self_topk:
            return
        response = opd_main_worker.LogprobResponse(output)
        pair = (
            response.strict_topk("output_top_logprobs", len(response_ids), self.topk_worker.top_k)
            if native
            else response.self_topk("rollout", self.topk_worker.top_k)
        )
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
        sample_list = list(samples) if isinstance(samples, Sequence) else [samples]

        if self.opsd_worker is not None:
            await asyncio.gather(*[self.opsd_worker.build_teacher_inputs(self.args, s) for s in sample_list])

        async with _create_teacher_client_session(self.args) as session:
            fetch_results = await asyncio.gather(*[self._teacher_prefill(s, session) for s in sample_list])
            self._raise_if_all_failed(sample_list, fetch_results)

            if self.topk_worker is not None and self.topk_worker.spec.student_at_teacher:
                await asyncio.gather(
                    *[self._student_prefill(s, session, encode_multimodal_inputs) for s in sample_list]
                )

        self._assemble_transfer(sample_list)

    async def _post_logprob(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        sample: Sample,
        err_tag: str,
        expected_input_ids: list[int] | None = None,
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
            if expected_input_ids is not None:
                raise RuntimeError(f"Native image OPD {err_tag} failed for sample_index={sample.index}") from exc
            return None

        response = opd_main_worker.LogprobResponse(data)
        if expected_input_ids is not None:
            response.validate_native_prefill(expected_input_ids, int(sample.response_length))
        return response

    def _validate_image_request(
        self, input_ids: list[int], expanded_ids: list[int], response_length: int, images: list
    ) -> None:
        if not images:
            return
        from relax.engine.rollout.sglang_rollout import GenerateState

        state = GenerateState(self.args)
        validate_image_input_ids(
            input_ids,
            expanded_ids,
            response_length,
            len(images),
            get_image_token_id(state.processor, state.tokenizer),
        )

    def _validate_query_ids(self, ids: np.ndarray | None, response_length: int) -> None:
        if ids is None or ids.shape != (response_length, self.topk_worker.top_k) or np.any(ids < 0):
            raise ValueError("Native image OPD requires complete R x K query IDs before cross-topk prefill.")

    async def _teacher_prefill(self, sample: Sample, session: aiohttp.ClientSession) -> bool:
        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            return True

        teacher_mm = sample.teacher_multimodal_inputs
        if teacher_mm is None:
            teacher_mm = sample.multimodal_inputs
        images = image_inputs(teacher_mm)
        native = bool(images or image_inputs(sample.multimodal_inputs))
        expanded_ids = list(sample.teacher_tokens if sample.teacher_tokens is not None else sample.tokens)
        if self.opsd_worker is not None:
            teacher_input_ids = self.opsd_worker.teacher_input_ids(sample, response_length)
            prompt_length = self.opsd_worker.teacher_prompt_len(sample, response_length)
        else:
            prompt_length = len(sample.tokens) - response_length
            collapsed_ids = sample.rollout_tokens or ([] if images else sample.tokens)
            teacher_input_ids = _replace_response_eos_for_teacher(
                collapsed_ids,
                prompt_length=len(collapsed_ids) - response_length,
                response_length=response_length,
                student_eos_token_id=self.student_eos_token_id,
                teacher_eos_token_id=self.teacher_eos_token_id,
            )
            if native and teacher_input_ids[-response_length:] != sample.tokens[-response_length:]:
                raise ValueError(
                    "Native image OPD requires identical response token IDs; EOS remapping is unsupported."
                )
            teacher_input_ids, _ = _insert_teacher_response_prefix(
                teacher_input_ids,
                prompt_length=len(teacher_input_ids) - response_length,
                prefix_token_ids=self.teacher_response_prefix_token_ids,
            )
            expanded_ids, prompt_length = _insert_teacher_response_prefix(
                expanded_ids,
                prompt_length=prompt_length,
                prefix_token_ids=self.teacher_response_prefix_token_ids,
            )
        if native and (prompt_length <= 0 or prompt_length != len(expanded_ids) - response_length):
            raise ValueError("Native image OPD requires a nonempty, correctly measured expanded teacher prompt.")
        self._validate_image_request(teacher_input_ids, expanded_ids, response_length, images)
        logprob_start_len = max(prompt_length - 1, 0)

        mm_fields = await encode_image_inputs(teacher_mm)
        if self.topk_worker is not None:
            if native and self.topk_worker.spec.teacher_at_student:
                self._validate_query_ids(sample.student_topk_token_ids, response_length)
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
        if native:
            payload["return_prompt_token_ids"] = True

        teacher_url = _pick_teacher_url(self.args, sample)
        resp_obj = await self._post_logprob(
            session, teacher_url, payload, sample, "teacher prefill", expanded_ids if native else None
        )
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

        fields = {}
        if self.sampled_worker is not None:
            fields["teacher_log_probs"] = [float(v) for v in token_logprobs[1 : 1 + response_length]]
        if self.topk_worker is not None:
            if self.topk_worker.spec.teacher_self_topk:
                pair = (
                    resp_obj.strict_topk("input_top_logprobs", response_length, self.topk_worker.top_k)
                    if native
                    else self.topk_worker.parse_prefill_self_topk(resp_obj, response_length)
                )
                fields["teacher_topk_token_ids"] = pair[0] if pair else None
                fields["teacher_topk_log_probs"] = pair[1] if pair else None
            if self.topk_worker.spec.teacher_at_student:
                fields["teacher_at_student_topk_log_probs"] = (
                    resp_obj.strict_topk(
                        "input_token_ids_logprobs",
                        response_length,
                        self.topk_worker.top_k,
                        sample.student_topk_token_ids,
                    )[1]
                    if native
                    else self.topk_worker.parse_prefill_other_topk(resp_obj, response_length)
                )
        for name, value in fields.items():
            setattr(sample, name, value)

        return True

    async def _student_prefill(
        self, sample: Sample, session: aiohttp.ClientSession, encode_mm_fn: EncodeMultimodalInputs | None
    ) -> None:

        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            return
        teacher_topk_ids = sample.teacher_topk_token_ids

        prompt_length = len(sample.tokens) - response_length
        logprob_start_len = max(prompt_length - 1, 0)

        images = image_inputs(sample.multimodal_inputs)
        native = bool(images or image_inputs(sample.teacher_multimodal_inputs))
        student_input_ids = sample.rollout_tokens or ([] if images else sample.tokens)
        self._validate_image_request(student_input_ids, sample.tokens, response_length, images)
        mm_fields = await encode_image_inputs(sample.multimodal_inputs)
        if native:
            self._validate_query_ids(teacher_topk_ids, response_length)

        payload = self.topk_worker.build_student_payload(
            input_ids=student_input_ids,
            logprob_start_len=logprob_start_len,
            teacher_topk_ids=teacher_topk_ids,
            response_length=response_length,
            mm_fields=mm_fields,
        )
        if native:
            payload["return_prompt_token_ids"] = True

        student_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
        resp_obj = await self._post_logprob(
            session, student_url, payload, sample, "student-at-teacher-topk", sample.tokens if native else None
        )
        if resp_obj is None:
            return
        sample.student_at_teacher_topk_log_probs = (
            resp_obj.strict_topk(
                "input_token_ids_logprobs", response_length, self.topk_worker.top_k, teacher_topk_ids
            )[1]
            if native
            else self.topk_worker.parse_prefill_other_topk(resp_obj, response_length)
        )

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
