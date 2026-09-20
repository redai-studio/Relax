# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pybase64


@dataclass
class TokenSelectionSpec:
    name: str
    student_self_topk: bool
    teacher_self_topk: bool
    teacher_at_student: bool
    student_at_teacher: bool  # only adv


SPECS: dict[str, TokenSelectionSpec] = {
    "student_sampled": TokenSelectionSpec("student_sampled", False, False, False, False),
    "student_topk": TokenSelectionSpec("student_topk", True, False, True, False),
    "teacher_topk": TokenSelectionSpec("teacher_topk", False, True, False, True),
    "union": TokenSelectionSpec("union", True, True, True, True),
}


class LogprobResponse:
    def __init__(self, resp: dict | None):
        self.meta = (resp or {}).get("meta_info", {}) or {}

    @staticmethod
    def _b64_decode(b64_str: str | None, dtype: str) -> np.ndarray:
        if not b64_str:
            return np.array([], dtype=np.dtype(dtype))
        return np.frombuffer(pybase64.b64decode(b64_str), dtype=np.dtype(dtype))

    def _decode_topk_legacy(self, prefix: str, response_length, top_k: int):
        # Fallback for stock SGLang servers that return list-form top logprobs
        # (e.g. "input_top_logprobs": [[(lp, id, txt), ...] or None, ...]) instead
        # of the b64-packed relay fields.
        legacy = self.meta.get(prefix)
        if not legacy:
            return None
        rows_lp: list = []
        rows_id: list = []
        for entry in legacy:
            if not entry:
                continue
            lp = [float(t[0]) for t in entry[:top_k]]
            ids = [int(t[1]) for t in entry[:top_k]]
            while len(lp) < top_k and lp:
                lp.append(lp[-1])
                ids.append(ids[-1])
            if len(lp) == top_k:
                rows_lp.append(lp)
                rows_id.append(ids)
        n = len(rows_lp)
        if n <= 0:
            return None
        if response_length is None:
            response_length = n
        if response_length <= 0 or n < response_length:
            return None
        lps = np.asarray(rows_lp[-response_length:], dtype=np.float32)
        ids_arr = np.asarray(rows_id[-response_length:], dtype=np.int32)
        return ids_arr, lps

    def _decode_topk_2d(self, prefix: str, response_length: int | None, top_k: int):
        if top_k <= 0:
            return None
        val = self._b64_decode(self.meta.get(f"{prefix}_val_b64"), "float32")
        n = val.size // top_k
        if n <= 0:
            return self._decode_topk_legacy(prefix, response_length, top_k)
        if response_length is None:
            response_length = n
        if response_length <= 0 or n < response_length:
            return None
        take = response_length * top_k
        lps = val[-take:].reshape(response_length, top_k)
        idx = self._b64_decode(self.meta.get(f"{prefix}_idx_b64"), "int32")
        ids = idx[-take:].reshape(response_length, top_k) if idx.size >= take else None
        return ids, lps

    def base_logprobs_1d(self) -> np.ndarray | None:
        val = self._b64_decode(self.meta.get("input_token_logprobs_val_b64"), "float32")
        if val.size:
            return val
        legacy = self.meta.get("input_token_logprobs")
        if legacy:
            return np.asarray([t[0] for t in legacy], dtype=np.float32)
        return None

    def self_topk(self, source: str, top_k: int, response_length: int | None = None):
        prefix = "output_top_logprobs" if source == "rollout" else "input_top_logprobs"
        pair = self._decode_topk_2d(prefix, response_length, top_k)
        if pair is None or pair[0] is None:
            return None
        return pair  # (ids[R,K] int32, logps[R,K] float32) 已是 numpy

    def other_topk(self, response_length: int, top_k: int):
        pair = self._decode_topk_2d("input_token_ids_logprobs", response_length, top_k)
        return pair[1] if pair is not None else None


def build_prefill_payload_base(input_ids: list[int], logprob_start_len: int) -> dict:
    return {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }


class SampledTokenWorker:
    TRANSFER_TEACHER_LOG_PROBS = "teacher_log_probs"
    TRANSFER_STUDENT_LOG_PROBS = "rollout_log_probs"

    @classmethod
    def from_args(cls, args) -> "SampledTokenWorker":
        return cls()

    def sampled_transfer_fields(self) -> list[str]:
        return [self.TRANSFER_TEACHER_LOG_PROBS, self.TRANSFER_STUDENT_LOG_PROBS]


class TopkWorker:
    TRANSFER_TOKEN_IDS = "opd_topk_token_ids"
    TRANSFER_TEACHER_LOG_PROBS = "opd_topk_teacher_log_probs"
    # only as_adv
    TRANSFER_STUDENT_LOG_PROBS = "opd_topk_student_log_probs"
    # only union
    TRANSFER_K_LENGTHS = "opd_topk_ksz"
    TRANSFER_FIELDS = (TRANSFER_TOKEN_IDS, TRANSFER_STUDENT_LOG_PROBS, TRANSFER_TEACHER_LOG_PROBS)

    def __init__(
        self,
        token_selection: str,
        top_k: int,
        opd_kl_coef: float = 0.0,
        opd_loss_coef: float = 0.0,
    ):
        self.top_k = top_k
        self.is_advantage = float(opd_kl_coef or 0.0) != 0.0
        self.spec = replace(
            SPECS[token_selection],
            student_at_teacher=self.is_advantage and SPECS[token_selection].student_at_teacher,
        )
        self._student_rollout_tpl: dict = {"top_logprobs_num": self.top_k} if self.spec.student_self_topk else {}
        self._teacher_prefill_tpl: dict = {}
        if self.spec.teacher_self_topk:
            self._teacher_prefill_tpl["top_logprobs_num"] = self.top_k
        self._student_prefill_tpl: dict = {} if self.spec.student_at_teacher else {}

    @classmethod
    def from_args(cls, args) -> "TopkWorker":
        return cls(
            args.opd_token_selection,
            args.opd_log_prob_top_k,
            opd_kl_coef=(0.0 if bool(getattr(args, "opd_fkl_entropy_gate", False)) else args.opd_kl_coef),
            opd_loss_coef=args.opd_loss_coef,
        )

    @staticmethod
    def _flatten_other_topk_ids(ids_2d: np.ndarray | None, response_length: int, top_k: int) -> list[int]:
        if top_k <= 0 or response_length <= 0:
            return []
        buf = np.zeros((response_length, top_k), dtype=np.int64)
        if ids_2d is not None:
            r = min(ids_2d.shape[0], response_length)
            k = min(ids_2d.shape[1], top_k)
            buf[:r, :k] = ids_2d[:r, :k]
        buf[buf < 0] = 0
        return buf.reshape(-1).tolist() + [0] * top_k

    def student_rollout_payload(self) -> dict:
        return dict(self._student_rollout_tpl)

    def build_teacher_payload(
        self,
        *,
        input_ids: list[int],
        logprob_start_len: int,
        student_topk_ids: np.ndarray | None,
        response_length: int,
        mm_fields: dict | None = None,
    ) -> dict:
        payload = build_prefill_payload_base(input_ids, logprob_start_len)
        payload.update(self._teacher_prefill_tpl)
        if self.spec.teacher_at_student:
            query_ids = self._flatten_other_topk_ids(student_topk_ids, response_length, self.top_k)
            if query_ids:
                payload["token_ids_logprob"] = query_ids
        if mm_fields:
            payload.update(mm_fields)
        return payload

    def build_student_payload(
        self,
        *,
        input_ids: list[int],
        logprob_start_len: int,
        teacher_topk_ids: np.ndarray | None,
        response_length: int,
        mm_fields: dict | None = None,
    ) -> dict:
        payload = build_prefill_payload_base(input_ids, logprob_start_len)
        payload.update(self._student_prefill_tpl)
        if self.spec.student_at_teacher:
            query_ids = self._flatten_other_topk_ids(teacher_topk_ids, response_length, self.top_k)
            if query_ids:
                payload["token_ids_logprob"] = query_ids
        if mm_fields:
            payload.update(mm_fields)
        return payload

    def parse_prefill_self_topk(self, resp: LogprobResponse, response_length: int):
        return resp.self_topk("prefill", self.top_k, response_length)

    def parse_prefill_other_topk(self, resp: LogprobResponse, response_length: int):
        return resp.other_topk(response_length, self.top_k)

    def build_transfer_channels(
        self,
        *,
        student_self_topk: tuple | None,  # (ids[R,K], logps[R,K]) 学生 rollout self-topk
        teacher_self_topk: tuple | None,  # (ids[R,K], logps[R,K]) 老师 prefill self-topk
        teacher_at_student_lp,  # logps[R,K] | None  老师在学生 topk ids 上
        student_at_teacher_lp,  # logps[R,K] | None  学生在老师 topk ids 上
    ) -> dict:
        name = self.spec.name
        if name == "student_topk":
            ids = student_self_topk[0] if student_self_topk else None
            student_lp = student_self_topk[1] if student_self_topk else None
            teacher_lp = teacher_at_student_lp
        elif name == "teacher_topk":
            ids = teacher_self_topk[0] if teacher_self_topk else None
            teacher_lp = teacher_self_topk[1] if teacher_self_topk else None
            student_lp = student_at_teacher_lp
        elif name == "union":
            return self._merge_union(
                student_self_topk, teacher_self_topk, teacher_at_student_lp, student_at_teacher_lp
            )
        else:
            return {}
        return self._pack(ids, student_lp, teacher_lp)

    def _pack(self, token_ids, student_lp, teacher_lp) -> dict:
        out: dict = {}
        if token_ids is not None:
            out[self.TRANSFER_TOKEN_IDS] = token_ids
        if teacher_lp is not None:
            out[self.TRANSFER_TEACHER_LOG_PROBS] = teacher_lp
        if self.is_advantage and student_lp is not None:
            out[self.TRANSFER_STUDENT_LOG_PROBS] = student_lp
        return out

    def _merge_union(self, student_self, teacher_self, teacher_at_student_lp, student_at_teacher_lp) -> dict:
        s_ids = student_self[0] if student_self else None
        s_student_lp = student_self[1] if student_self else None
        t_ids = teacher_self[0] if teacher_self else None
        t_teacher_lp = teacher_self[1] if teacher_self else None
        s_teacher_lp = teacher_at_student_lp
        t_student_lp = student_at_teacher_lp

        if s_ids is None or t_ids is None or s_teacher_lp is None or t_teacher_lp is None:
            return {}
        if self.is_advantage and (s_student_lp is None or t_student_lp is None):
            return {}

        R = s_ids.shape[0]
        ids_rows: list[np.ndarray] = []
        teacher_rows: list[np.ndarray] = []
        student_rows: list[np.ndarray] | None = [] if self.is_advantage else None
        for r in range(R):
            ids_cat = np.concatenate([s_ids[r], t_ids[r]])
            uniq, first = np.unique(ids_cat, return_index=True)
            ids_rows.append(uniq)
            lp_cat = np.concatenate([s_teacher_lp[r], t_teacher_lp[r]])
            teacher_rows.append(lp_cat[first])
            if student_rows is not None:
                lp_cat = np.concatenate([s_student_lp[r], t_student_lp[r]])
                student_rows.append(lp_cat[first])

        max_kp = max(row.shape[0] for row in ids_rows)
        ids_pad = np.full((R, max_kp), -1, dtype=np.int32)
        teacher_pad = np.zeros((R, max_kp), dtype=np.float32)
        student_pad = np.zeros((R, max_kp), dtype=np.float32) if student_rows is not None else None
        k_lengths = np.zeros(R, dtype=np.int32)
        for r in range(R):
            kp = ids_rows[r].shape[0]
            ids_pad[r, :kp] = ids_rows[r]
            teacher_pad[r, :kp] = teacher_rows[r]
            if student_pad is not None:
                student_pad[r, :kp] = student_rows[r]
            k_lengths[r] = kp

        out = self._pack(ids_pad, student_pad, teacher_pad)
        out[self.TRANSFER_K_LENGTHS] = k_lengths
        return out

    def topk_transfer_fields(self) -> list[str]:
        fields: list[str] = [self.TRANSFER_TOKEN_IDS, self.TRANSFER_TEACHER_LOG_PROBS]
        if self.is_advantage:
            fields.append(self.TRANSFER_STUDENT_LOG_PROBS)
        if self.spec.name == "union":
            fields.append(self.TRANSFER_K_LENGTHS)
        return fields


def restore_opd_topk_rollout_fields(rollout_data: dict, args, device) -> None:
    import torch

    token_selection = getattr(args, "opd_token_selection", None)
    opd_top_k = int(getattr(args, "opd_log_prob_top_k", 0) or 0)
    if opd_top_k <= 0 or token_selection not in ("student_topk", "teacher_topk", "union"):
        return
    is_union = token_selection == "union"
    response_lengths = rollout_data.get("response_lengths", [])
    for key, dtype in (
        (TopkWorker.TRANSFER_TOKEN_IDS, torch.long),
        (TopkWorker.TRANSFER_TEACHER_LOG_PROBS, torch.float32),
        (TopkWorker.TRANSFER_STUDENT_LOG_PROBS, torch.float32),
    ):
        if rollout_data.get(key) is None:
            continue
        restored = []
        for i, v in enumerate(rollout_data[key]):
            if v is None:
                restored.append(v)
                continue
            t = torch.as_tensor(v, device=device, dtype=dtype)
            if t.numel() == 0:
                kp = opd_top_k if not is_union else 1
                restored.append(t.reshape(0, kp))
                continue
            if is_union:
                resp_len = int(response_lengths[i]) if i < len(response_lengths) else 0
                if resp_len <= 0:
                    restored.append(t.reshape(0, 1))
                    continue
                max_kp = t.numel() // resp_len
                restored.append(t.reshape(resp_len, max_kp))
            else:
                restored.append(t.reshape(-1, opd_top_k))
        rollout_data[key] = restored

    if is_union and rollout_data.get(TopkWorker.TRANSFER_K_LENGTHS) is not None:
        rollout_data[TopkWorker.TRANSFER_K_LENGTHS] = [
            torch.as_tensor(v, device=device, dtype=torch.long) if v is not None else None
            for v in rollout_data[TopkWorker.TRANSFER_K_LENGTHS]
        ]
