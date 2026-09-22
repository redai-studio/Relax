# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Periodic SFT predict pass.

Renders the SFT eval dataset's user-side messages into prompts (drops the
trailing assistant turn), batches them through SGLang via
``RolloutManager.generate_predict``, and writes per-step JSONL to
``<save>/predict/predictions_step_<train_step>.jsonl``.

Eval data source mirrors ``components/sft.py::SFT._init_data_pipeline``:
``--eval-prompt-data`` (separate dataset) or ``--eval-size`` (fixed-seed
random subset of ``--prompt-data``). Argparse guarantees one of them is set whenever
``--sft-predict-interval`` is set.
"""

import json
from pathlib import Path

from relax.engine.sft.dataset.sample import CanonicalSample
from relax.engine.sft.runtime import resolve_sft_split_indices
from relax.utils.logging_utils import get_logger
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.training.eval_config import build_named_prompt_data_configs


logger = get_logger(__name__)


def _build_multimodal_inputs(sample: CanonicalSample) -> dict | None:
    """Pack CanonicalSample media into the dict shape expected by
    ``relax.engine.rollout.sglang_rollout._encode_multimodal_inputs``.

    Returns ``None`` if the sample has no media. Note the singular ``audio``
    key — that's what the encoder expects, even though CanonicalSample uses
    ``audios``.

    Raw entries from the dataset reader (HF-style ``{"bytes", "path"}`` dicts,
    file paths, raw bytes, data URIs) are decoded into PIL Images / tensors via
    the same loaders the train path uses (see
    ``relax/engine/sft/dataset/multimodal.py::_fetch_media``); the rollout
    encoders downstream assume already-decoded media.
    """
    if not (sample.images or sample.videos or sample.audios):
        return None
    from relax.utils.multimodal.audio_utils import load_audio
    from relax.utils.multimodal.image_utils import load_image
    from relax.utils.multimodal.video_utils import load_video

    return {
        "images": [load_image(p) for p in (sample.images or [])],
        "videos": [load_video(p) for p in (sample.videos or [])],
        "audio": [load_audio(p) for p in (sample.audios or [])],
    }


def split_prompt_and_reference(sample: CanonicalSample) -> tuple[list[dict], str]:
    """Drop the trailing assistant turn and return ``(prompt_messages,
    reference_text)``.

    ``prompt_messages`` is OpenAI-style ``[{"role": ..., "content": ...}]`` with
    everything up to (not including) the LAST assistant message.

    ``reference`` is the dropped assistant message's content. List-of-parts
    content (e.g. multimodal) is JSON-serialized so the JSONL line stays
    valid; downstream consumers can parse it back if they care.
    """
    messages = sample.messages
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx < 0:
        prompt_messages = [{"role": m.role, "content": m.content} for m in messages]
        return prompt_messages, ""
    prompt_messages = [{"role": m.role, "content": m.content} for m in messages[:last_assistant_idx]]
    ref_content = messages[last_assistant_idx].content
    reference = ref_content if isinstance(ref_content, str) else json.dumps(ref_content, ensure_ascii=False)
    return prompt_messages, reference


def render_eval_prompts(config) -> list[tuple[str, str, dict | None]]:
    """Build the eval dataset and render each sample to ``(prompt, reference,
    multimodal_inputs)``.

    Uses the same source resolution as ``SFT._init_data_pipeline``:
    ``--eval-size`` uses the same deterministic random row split as SFT PPL
    evaluation; ``--eval-prompt-data`` loads a separate dataset.

    ``multimodal_inputs`` is the dict shape consumed by
    ``_encode_multimodal_inputs`` (or ``None`` for text-only samples).
    """
    from transformers import AutoTokenizer

    from relax.engine.sft.dataset.streaming import SFTStreamingDataset

    prepare_model_maybe_update_args(config, completeness="metadata")
    tokenizer = AutoTokenizer.from_pretrained(config.hf_checkpoint, trust_remote_code=True)
    cp_size = max(1, getattr(config, "context_parallel_size", 1) or 1)
    capacity = config.max_tokens_per_gpu * cp_size
    seed = getattr(config, "seed", 42)
    eval_prompt_data = build_named_prompt_data_configs(getattr(config, "eval_prompt_data", None))
    eval_size_arg = getattr(config, "eval_size", None)

    if eval_size_arg is not None:
        dataset = SFTStreamingDataset(
            path=config.prompt_data,
            tokenizer=tokenizer,
            processor_pool=None,
            capacity=capacity,
            prompt_key=config.input_key,
            label_key=config.label_key,
            multimodal_keys=config.multimodal_keys,
            conversation_key_map=getattr(config, "conversation_key_map", None),
            metadata_key=config.metadata_key,
            tool_key=config.tool_key,
            system_prompt=config.system_prompt,
            require_response=False,
            seed=seed,
            prefetch_max_cached=0,
            pad_token_ids=None,
            invalid_multimodal_strategy=getattr(config, "sft_invalid_multimodal_strategy", "error"),
            apply_chat_template_kwargs=getattr(config, "apply_chat_template_kwargs", None),
        )
        n_avail = len(dataset)
        _, eval_indices = resolve_sft_split_indices(n_avail, eval_size_arg, seed)
    elif eval_prompt_data:
        eval_input_key = getattr(config, "eval_input_key", None) or config.input_key
        eval_label_key = getattr(config, "eval_label_key", None) or config.label_key
        eval_tool_key = getattr(config, "eval_tool_key", None) or config.tool_key
        dataset = SFTStreamingDataset(
            path=[d.path for d in eval_prompt_data],
            tokenizer=tokenizer,
            processor_pool=None,
            capacity=capacity,
            prompt_key=eval_input_key,
            label_key=eval_label_key,
            multimodal_keys=config.multimodal_keys,
            conversation_key_map=getattr(config, "conversation_key_map", None),
            metadata_key=config.metadata_key,
            tool_key=eval_tool_key,
            system_prompt=config.system_prompt,
            source_name="+".join(d.name for d in eval_prompt_data),
            require_response=False,
            seed=seed,
            prefetch_max_cached=0,
            pad_token_ids=None,
            invalid_multimodal_strategy=getattr(config, "sft_invalid_multimodal_strategy", "error"),
            apply_chat_template_kwargs=getattr(config, "apply_chat_template_kwargs", None),
        )
        eval_indices = tuple(range(len(dataset)))
    else:
        raise RuntimeError("render_eval_prompts requires --eval-prompt-data or --eval-size")

    if not eval_indices:
        dataset.stop()
        return []

    out: list[tuple[str, str, dict | None]] = []
    for idx in eval_indices:
        try:
            sample = dataset.get_canonical_sample(idx)
        except Exception:
            logger.exception(f"render_eval_prompts: load failed for idx={idx}; skipping.")
            continue
        prompt_msgs, reference = split_prompt_and_reference(sample)
        try:
            prompt = tokenizer.apply_chat_template(
                prompt_msgs,
                tools=sample.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.exception(f"render_eval_prompts: chat_template failed for idx={idx}; skipping.")
            continue
        out.append((prompt, reference, _build_multimodal_inputs(sample)))

    dataset.stop()
    return out


async def generate_and_write_predictions(
    rollout_manager,
    prompts_and_refs: list[tuple[str, str, dict | None]],
    out_path: Path,
) -> None:
    """Generate all completions concurrently and write a JSONL of ``{prompt,
    reference, completion}`` rows to ``out_path``.

    Overwrites ``out_path`` if it exists (idempotent on filename — caller
    encodes the train_step in the filename for uniqueness across rounds).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = [p for p, _, _ in prompts_and_refs]
    mm_inputs_list = [m for _, _, m in prompts_and_refs]
    completions = await rollout_manager.generate_predict(prompts, mm_inputs_list)
    if len(completions) != len(prompts):
        logger.warning(
            f"generate_predict returned {len(completions)} completions for {len(prompts)} prompts; "
            "padding/truncating to align."
        )
        completions = list(completions)
        if len(completions) < len(prompts):
            completions += [""] * (len(prompts) - len(completions))
        else:
            completions = completions[: len(prompts)]
    out_lines = [
        json.dumps(
            {"prompt": prompt, "reference": reference, "completion": completion},
            ensure_ascii=False,
        )
        for (prompt, reference, _), completion in zip(prompts_and_refs, completions, strict=True)
    ]
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


async def run_predict_loop(rollout_manager, config, train_step: int) -> None:
    """Render eval prompts, batch-generate, write
    ``<config.save>/predict/predictions_step_<train_step>.jsonl``.

    No-op (with a warning log) if the eval source produces zero prompts. The
    rollout manager must already have KV/cuda-graph onloaded; the caller
    (``RolloutManager.run_predict``) handles onload/offload coordination.
    """
    assert getattr(config, "save", None), "run_predict_loop requires config.save"
    out_path = Path(config.save) / "predict" / f"predictions_step_{train_step}.jsonl"

    prompts_and_refs = render_eval_prompts(config)
    if not prompts_and_refs:
        logger.warning(f"SFT predict @ step {train_step}: 0 prompts after rendering; skipping.")
        return

    logger.info(f"SFT predict @ step {train_step}: generating {len(prompts_and_refs)} completions → {out_path}")
    await generate_and_write_predictions(rollout_manager, prompts_and_refs, out_path)
    logger.info(f"SFT predict @ step {train_step}: wrote {len(prompts_and_refs)} lines to {out_path}")
