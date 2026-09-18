# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""PickScore reward scorer (CLIP-based text-image alignment).

Matches the PickScore alignment recipe: CLIP-H processor +
``yuvalkirstain/PickScore_v1`` model in fp32, score =
``logit_scale * (text_embs @ image_embs.T).diag() / 26``. Emits the
``pickscore`` component. The embedding→score math is a pure, CPU-testable
function.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from relax.engine.rewards.generative import BaseGenerativeScorer, RewardRequest, load_track_uris
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = ["PickScoreScorer", "pickscore_from_embeds"]

_PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
_MODEL_ID = "yuvalkirstain/PickScore_v1"
_PICKSCORE_NORM = 26.0
# Score in chunks: a rollout hands the scorer its whole batch (256 candidates at
# n=8 x 32 prompts), and one CLIP-H forward over all of them needs tens of GB of
# activations. On the `colocate` runtime that lands on a rollout GPU still
# holding the resident diffusion engine, so an unchunked forward OOMs. The
# reference implementation chunks the same way.
_SCORE_BATCH_SIZE = 32


def pickscore_from_embeds(
    text_embs: torch.Tensor, image_embs: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    """Per-pair PickScore from L2-normalized embeddings.

    ``score_i = logit_scale * <text_i, image_i> / 26``. Inputs are L2-normalized
    here so the caller can pass raw CLIP features.
    """
    text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
    image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
    scale = logit_scale.exp() if logit_scale.ndim == 0 else logit_scale
    return scale * (text_embs * image_embs).sum(dim=-1) / _PICKSCORE_NORM


class PickScoreScorer(BaseGenerativeScorer):
    """CLIP PickScore over (prompt, generated image) pairs."""

    required_tracks: Tuple[str, ...] = ("image",)
    component_name = "pickscore"
    track = "image"

    def __init__(self, args, *, allow_cuda: bool = True) -> None:
        super().__init__(args, allow_cuda=allow_cuda)
        self._processor = None
        self._model_id = getattr(args, "reward_model_path", None) or _MODEL_ID

    def _load_model(self):
        from transformers import CLIPModel, CLIPProcessor

        # Prefer the processor bundled with the model checkpoint (PickScore_v1 is a
        # fine-tuned CLIP-H and ships its own preprocessor/tokenizer), so an
        # air-gapped node with a local --reward-model-path needs no HF download.
        # Fall back to the canonical CLIP-H processor ID only when the checkpoint
        # directory has no processor files — that fallback DOES hit the network, so
        # it must stay a fallback and never the first attempt.
        try:
            self._processor = CLIPProcessor.from_pretrained(self._model_id)
        except OSError:  # no processor/tokenizer files at _model_id
            logger.warning(
                f"PickScore: {self._model_id!r} ships no CLIP processor; falling back to {_PROCESSOR_ID!r} "
                "(this requires network access — set --reward-model-path to a checkpoint that includes "
                "preprocessor_config.json on offline clusters)."
            )
            self._processor = CLIPProcessor.from_pretrained(_PROCESSOR_ID)
        model = CLIPModel.from_pretrained(self._model_id).eval().to(dtype=torch.float32)
        return model

    @torch.no_grad()
    def score_batch(self, requests: List[RewardRequest]) -> Dict[str, List[float]]:
        scores: List[float] = []
        for start in range(0, len(requests), _SCORE_BATCH_SIZE):
            scores.extend(self._score_chunk(requests[start : start + _SCORE_BATCH_SIZE]))
        return {self.component_name: scores}

    def _score_chunk(self, requests: List[RewardRequest]) -> List[float]:
        images = []
        prompts = []
        for pos, request in enumerate(requests):
            uris = load_track_uris(request, self.track)
            if not uris:
                raise ValueError(f"PickScore: request {pos} has no required {self.track!r} artifact.")
            memory_image = request.get("memory_images", {}).get(uris[0])
            if memory_image is not None:
                from PIL import Image

                images.append(Image.fromarray(memory_image).convert("RGB"))
                prompts.append(_prompt_text(request))
                continue
            try:
                from PIL import Image

                with Image.open(uris[0]) as image:
                    images.append(image.convert("RGB"))
            except Exception as e:
                raise ValueError(f"PickScore: failed to load required {self.track!r} artifact {uris[0]!r}.") from e
            prompts.append(_prompt_text(request))

        image_inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        text_inputs = self._processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        # Use the full CLIPModel forward: its CLIPOutput.{image,text}_embeds are the
        # projected embeddings, stable across transformers versions (unlike
        # get_{image,text}_features, whose return type became a ModelOutput in
        # transformers 5.x).
        outputs = self._model(
            pixel_values=image_inputs["pixel_values"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
        )
        image_embs = outputs.image_embeds.float()
        text_embs = outputs.text_embeds.float()
        scores = pickscore_from_embeds(text_embs, image_embs, self._model.logit_scale)
        return [float(score) for score in scores.cpu().tolist()]


def _prompt_text(request: RewardRequest) -> str:
    prompt = request.get("prompt", "")
    if isinstance(prompt, list):
        # chat-format prompt → concatenate the text turns
        return " ".join(seg.get("content", "") if isinstance(seg, dict) else str(seg) for seg in prompt)
    return str(prompt)
