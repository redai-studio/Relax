# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

import aiohttp
import numpy as np

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


# --- Teacher request target ---
# A Relax-managed teacher is reached only through its Gateway, which resolves
# the model from the Manager snapshot and leaves replica choice to the model's
# SGLang Router. MOPD sends the sample's ``args.opd_teacher_key`` metadata as
# the Gateway ``route_key``. Every request also carries its GRPO group as the
# Router's routing key, so a group's shared prompt prefix is prefilled by one
# replica. An external teacher (``--opd-teacher-url``) keeps its raw URL.
TEACHER_ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


def _teacher_route_key(args, sample: Sample) -> str:
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
    if routing_value not in args.opd_teacher_route_keys:
        raise KeyError(
            f"MOPD routing: no teacher route for '{key_field}={routing_value}'. "
            f"Available routes: {list(args.opd_teacher_route_keys)}."
        )
    return routing_value


def _teacher_request_target(args, sample, payload: dict) -> tuple[str, dict]:
    gateway_url = getattr(args, "opd_teacher_gateway_url", None)
    if not gateway_url:
        return args.opd_teacher_url, payload
    payload = dict(payload)
    if getattr(args, "opd_teacher_route_keys", None):
        payload["route_key"] = _teacher_route_key(args, sample)
    return f"{gateway_url.rstrip('/')}/generate", payload


def _teacher_routing_headers(args, sample) -> dict[str, str]:
    """Pin a managed teacher request to its group's replica; each teacher model
    has its own Router, so the group index alone is the key."""
    group_index = getattr(sample, "group_index", None)
    if not getattr(args, "opd_teacher_gateway_url", None) or group_index is None:
        return {}
    return {TEACHER_ROUTING_KEY_HEADER: str(group_index)}


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

    @property
    def needs_student_prefill(self) -> bool:
        """Whether this token selection needs a second pass on the student."""
        return self.topk_worker is not None and self.topk_worker.spec.student_at_teacher

    async def prepare_teacher_inputs(self, samples: Sequence[Sample]) -> None:
        """Build the teacher-side inputs OPSD expands before any request."""
        if self.opsd_worker is not None:
            await asyncio.gather(*[self.opsd_worker.build_teacher_inputs(self.args, s) for s in samples])

    async def score_teacher(
        self, samples: Sequence[Sample], session: aiohttp.ClientSession | None = None
    ) -> list[bool]:
        """Run teacher prefill for ``samples`` and report per-sample success.

        Separated from :meth:`prefill` so a deferred pipeline can run this
        stage inside the teacher's activation phase and the student stage in
        another, instead of holding both models resident at once.
        """
        if session is not None:
            return list(await asyncio.gather(*[self._teacher_prefill(s, session) for s in samples]))
        async with _create_teacher_client_session(self.args) as owned:
            return list(await asyncio.gather(*[self._teacher_prefill(s, owned) for s in samples]))

    async def score_student_at_teacher(
        self,
        samples: Sequence[Sample],
        session: aiohttp.ClientSession | None = None,
        encode_multimodal_inputs: EncodeMultimodalInputs | None = None,
    ) -> None:
        """Second pass on the student, for token selections that need it."""
        if not self.needs_student_prefill:
            return
        if session is not None:
            await asyncio.gather(*[self._student_prefill(s, session, encode_multimodal_inputs) for s in samples])
            return
        async with _create_teacher_client_session(self.args) as owned:
            await asyncio.gather(*[self._student_prefill(s, owned, encode_multimodal_inputs) for s in samples])

    def assemble_transfer(self, samples: Sequence[Sample]) -> None:
        """Assemble the per-sample training channels from the scored fields."""
        self._assemble_transfer(list(samples))

    async def prefill(
        self,
        samples: Sample | Sequence[Sample],
        encode_multimodal_inputs: EncodeMultimodalInputs | None = None,
    ) -> None:
        sample_list = list(samples) if isinstance(samples, Sequence) else [samples]

        await self.prepare_teacher_inputs(sample_list)

        async with _create_teacher_client_session(self.args) as session:
            fetch_results = await self.score_teacher(sample_list, session)
            self._raise_if_all_failed(sample_list, fetch_results)
            await self.score_student_at_teacher(sample_list, session, encode_multimodal_inputs)

        self._assemble_transfer(sample_list)

    async def _post_logprob(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        sample: Sample,
        err_tag: str,
        headers: dict[str, str] | None = None,
    ) -> opd_main_worker.LogprobResponse | None:
        post_kwargs = _aiohttp_json_post_kwargs(payload)
        post_kwargs["headers"].update(headers or {})
        try:
            async with session.post(url, **post_kwargs) as resp:
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

        teacher_url, payload = _teacher_request_target(self.args, sample, payload)
        resp_obj = await self._post_logprob(
            session,
            teacher_url,
            payload,
            sample,
            "teacher prefill",
            headers=_teacher_routing_headers(self.args, sample),
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
