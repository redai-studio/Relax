# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Text-edit sentence reward: (0.7 s_ned + 0.3 s_acc) * f_lpips * f_dup.

OCR is remote; LPIPS/AlexNet runs on CPU with explicitly provisioned offline
weights. Missing GT, OCR failure or unavailable LPIPS never silently changes
the reward into MSE or an unpenalized OCR score.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from relax.engine.rewards._textpecker_ff import (
    extract_rec_texts,
    get_rec_template,
    match_substring,
    parse_json_from_answer,
    process_raw_text,
    tokenize_text,
)
from relax.engine.rewards.generative import BaseGenerativeScorer, RewardRequest, load_track_uris
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_TARGET_RE = re.compile(r'替换成\s*["“]([^"”]*)["”]')


def _prompt_text(request: dict[str, Any]) -> str:
    prompt = request.get("prompt", "")
    if isinstance(prompt, list):
        return " ".join(seg.get("content", "") if isinstance(seg, dict) else str(seg) for seg in prompt)
    return str(prompt)


def extract_targets(request: dict[str, Any]) -> list[str]:
    targets = (request.get("metadata") or {}).get("target_texts")
    if isinstance(targets, str):
        try:
            targets = json.loads(targets)
        except (ValueError, TypeError):
            targets = None
    if isinstance(targets, list):
        targets = [str(t) for t in targets if str(t).strip()]
        if targets:
            return targets
    return [t for t in _TARGET_RE.findall(_prompt_text(request)) if t.strip()]


def recognized_text(reply: str) -> str:
    try:
        payload, _ = parse_json_from_answer(reply)
        text = payload.get("recognized_text")
    except Exception:
        text = extract_rec_texts(reply)
    if not isinstance(text, (str, list)) or (isinstance(text, list) and any(not isinstance(t, str) for t in text)):
        raise ValueError("TextPecker reply has no recognized_text")
    return process_raw_text(text)


def ocr_sentence_reward(text: str, targets: list[str]) -> float:
    """Business held-out metric: OCR only, without LPIPS or duplicate gates."""
    if not targets:
        raise ValueError("OCR reward requires target texts")
    clean = re.sub(r"<#+>", "", text)
    clean = re.sub(r"[<>#]", "", clean)
    matches = match_substring(targets, [clean], case_sensitive=True, strip_punct=True)
    return float(sum(0.7 * row["sim"] + 0.3 * row["exact"] for row in matches) / len(matches))


def sentence_reward(text: str, targets: list[str], lpips_distance: float) -> float:
    if not targets or not math.isfinite(lpips_distance) or lpips_distance < 0:
        raise ValueError("Targets and a finite nonnegative LPIPS distance are required")
    base = ocr_sentence_reward(text, targets)
    gt = Counter(tokenize_text(" ".join(targets)))
    pred = Counter(tokenize_text(text))
    extra = sum(max(0, pred[token] - count) for token, count in gt.items())
    dup = math.exp(-0.5 * extra / sum(gt.values())) if gt and pred else 1.0
    sim = max(0.0, min(1.0, 1.0 - lpips_distance))
    return float(base * (0.3 + 0.7 * sim) * dup)


class TextPeckerRewardScorer(BaseGenerativeScorer):
    required_tracks = ("image",)
    components = ("textpecker_lpips",)
    requires_gt = True

    def __init__(self, args: Any, *, allow_cuda: bool = True) -> None:
        super().__init__(args, allow_cuda=False)
        endpoint = str(getattr(args, "reward_endpoint", None) or "").rstrip("/")
        if not endpoint.endswith("/v1"):
            raise ValueError("TextPecker requires an explicit reward_endpoint ending in /v1")
        self._endpoint = endpoint
        self._model_name = getattr(args, "reward_model_path", None) or "TextPecker"
        self._client = None
        self._concurrency = 8

    def _load_model(self) -> torch.nn.Module:
        # Check before constructing torchvision's pretrained trunk: no implicit download.
        backbone = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
        if not backbone.is_file():
            raise FileNotFoundError(f"Provision the pretrained AlexNet weights in the offline Torch cache: {backbone}")
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("TextPecker reward requires the optional lpips dependency") from exc
        model = lpips.LPIPS(net="alex", verbose=False).eval()
        model.requires_grad_(False)
        return model

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._endpoint, api_key="EMPTY", max_retries=0)
        return self._client

    def _complete(self, image: Any) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
        client = self._ensure_client()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}},
                                {"type": "text", "text": get_rec_template()},
                            ],
                        }
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                    timeout=120.0,
                    extra_body={"repetition_penalty": 1.2},
                )
                content = response.choices[0].message.content
                if not isinstance(content, str):
                    raise ValueError("Empty TextPecker response")
                # HTTP success is insufficient: malformed/truncated OCR must
                # share the same bounded retry budget as transport failures.
                recognized_text(content)
                return content
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError("TextPecker request exhausted three attempts") from exc
                logger.warning(f"TextPecker request failed on attempt {attempt + 1}; retrying: {exc}")
                time.sleep(min(5 * (2**attempt), 30))
        raise AssertionError("Unreachable retry loop")

    def _read_candidate(self, request: RewardRequest) -> tuple[Any, Any, list[str]]:
        from PIL import Image

        uris = load_track_uris(request, "image")
        targets = extract_targets(request)
        gt_path = (request.get("metadata") or {}).get("gt_path")
        if len(uris) != 1 or not targets or (self.requires_gt and not gt_path):
            raise ValueError("TextPecker requires one generated image, target texts and metadata.gt_path")
        memory = (request.get("memory_images") or {}).get(uris[0])
        if memory is not None:
            image = Image.fromarray(memory).convert("RGB")
        else:
            with Image.open(uris[0]) as handle:
                image = handle.convert("RGB")
        gt = None
        if self.requires_gt:
            with Image.open(gt_path) as handle:
                gt = handle.convert("RGB")
        return image, gt, targets

    def _ocr_one(self, request: RewardRequest) -> tuple[Any, Any, list[str], str] | None:
        try:
            image, gt, targets = self._read_candidate(request)
            text = recognized_text(self._complete(image))
            return image, gt, targets, text
        except Exception as exc:
            sample_id = (request.get("metadata") or {}).get("sample_id", "unknown")
            logger.warning(f"TextPecker candidate {sample_id!r} invalid: {exc}")
            return None

    @torch.no_grad()
    def _lpips_distance(self, image: Any, gt: Any) -> float:
        import numpy as np
        from PIL import Image

        if self._model is None:
            raise RuntimeError("Call scorer.onload() before LPIPS scoring")

        def tensor(im: Any) -> torch.Tensor:
            resized = im.resize((256, 256), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

        return float(self._model(tensor(image), tensor(gt)).flatten()[0].item())

    def score_batch(self, requests: list[RewardRequest]) -> dict[str, list]:
        # Construct the shared HTTP client before threads, avoiding lazy-init races.
        if not requests:
            return {self.components[0]: [], "__valid_mask__": []}
        self._ensure_client()
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            results = list(pool.map(self._ocr_one, requests))
        scores, valid = [], []
        for result in results:
            if result is None:
                scores.append(0.0)
                valid.append(False)
                continue
            image, gt, targets, text = result
            # Systemic LPIPS failure is fatal, not a silent change of algorithm.
            score = self._score_recognition(image, gt, targets, text)
            scores.append(score)
            valid.append(True)
        return {self.components[0]: scores, "__valid_mask__": valid}

    def _score_recognition(self, image: Any, gt: Any, targets: list[str], text: str) -> float:
        distance = self._lpips_distance(image, gt)
        return sentence_reward(text, targets, distance)


class TextPeckerEvalScorer(TextPeckerRewardScorer):
    """Explicit OCR-only evaluator matching the business eval reward config.

    Use separately from the training scorer; no GT is invented for the 32
    business held-out rows that have none, and no LPIPS model is constructed.
    """

    components = ("textpecker_ocr",)
    requires_gt = False

    def _load_model(self) -> None:
        return None

    def _score_recognition(self, image: Any, gt: Any, targets: list[str], text: str) -> float:
        return ocr_sentence_reward(text, targets)
