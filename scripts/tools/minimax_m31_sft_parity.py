# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare the Miles mask and select the MSA implementation for a MiniMax
parity run."""

import argparse
import ast
import json
import math
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


def _generate_fixture_media(media: Path) -> None:
    import numpy as np
    from imageio.v2 import get_writer
    from PIL import Image, ImageDraw

    def frame(red: int, left: int, top: int) -> Image.Image:
        image = Image.new("RGB", (112, 112), (red, 0, 0))
        ImageDraw.Draw(image).rectangle((left, top, left + 39, top + 43), fill=(red, 230, 0))
        return image

    media.mkdir(exist_ok=True)
    frame(60, 18, 22).save(media / "green.png")
    frame(120, 38, 34).save(media / "red.png")
    with get_writer(
        media / "green.mp4",
        fps=4,
        codec="libx264",
        pixelformat="yuv420p",
        quality=None,
        macro_block_size=1,
    ) as writer:
        for index in range(8):
            writer.append_data(np.asarray(frame(30 + 15 * index, 10 + 4 * index, 16 + 3 * index)))


def make_fixture(data: Path) -> None:
    media = data.parent / "media"
    with data.open("x") as output:
        _generate_fixture_media(media)
        green, red, video = (str((media / name).resolve()) for name in ("green.png", "red.png", "green.mp4"))
        note = "The red square is to the left of the blue circle. "
        long_question = (
            "<image>Notes: " + (note * 196).rstrip() + "\nWhat is 0 plus seven? Answer with only the number."
        )
        cases = [
            ("<image>What color is the square?", "It is green.", [green], []),
            ("<video>Does the shape move?", "Yes, it moves.", [], [video]),
            (
                "Compare <image> and <image>. Then watch <video>.",
                "The square changes position.",
                [green, red],
                [video],
            ),
            (long_question, "7", [green], []),
            ("What is two plus two?", "Four.", [], []),
            ("<video>Watch again <video>.", "The square moves to the right.", [], [video, video]),
        ]
        for prompt, answer, images, videos in cases * 28:
            row = {
                "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                "images": images,
                "videos": videos,
            }
            output.write(json.dumps(row) + "\n")


def _parity_template(template: str) -> str:
    # With one answer and no reasoning, Miles also supervises </mm:think>.
    before, assistant, conversation = template.partition("{%- if message.role == 'assistant' -%}")
    block, tool, after = conversation.partition("{%- elif message.role == 'tool' -%}")
    start = "{%- if reasoning_content -%}"
    end = "{{- eos_token ~ '\\n' }}"
    if not assistant or not tool or block.count(start) != 1 or block.count(end) != 1:
        raise ValueError("Expected the official MiniMax-M3.1-Flash chat template.")
    block = block.replace(start, "{%- generation -%}\n" + start)
    block = block.replace(end, end + "\n{%- endgeneration -%}")
    return before + assistant + block + tool + after


def verify_inputs(checkpoint: Path, data: Path, template_out: Path) -> None:
    import torch
    from miles.utils.mask_utils import MultiTurnLossMaskGenerator

    from relax.engine.sft.dataset.streaming import SFTStreamingDataset
    from relax.utils.data import processor_pool
    from relax.utils.data.image_refs import SFT_HF_IMAGE_REBUILD_SUPPORTED
    from relax.utils.data.minimax import IMAGE_TOKEN, VIDEO_TOKEN
    from relax.utils.data.processing_utils import load_tokenizer
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    tokenizer = load_tokenizer(str(checkpoint), trust_remote_code=True)
    template = _parity_template(tokenizer.chat_template)
    miles_mask = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen")
    processor_pool._init_worker(str(checkpoint), True, None)
    layouts = set()
    verified = set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        dataset = SFTStreamingDataset(
            str(data),
            prompt_key="messages",
            tokenizer=tokenizer,
            processor_pool=SimpleNamespace(executor=executor),
            multimodal_keys={"image": "images", "video": "videos"},
            apply_chat_template_kwargs={"chat_template": template},
            prefetch_max_cached=0,
        )
        for idx in range(len(dataset)):
            row = dataset.reader[idx]
            if len(row["messages"]) != 2 or [msg["role"] for msg in row["messages"]] != ["user", "assistant"]:
                raise ValueError("Parity data must contain one user turn and one plain assistant answer per row.")
            messages = deepcopy(row["messages"])
            answer = messages[1]
            if "mm:think" in answer["content"] or answer.get("reasoning_content") or answer.get("tool_calls"):
                raise ValueError("Parity data requires assistant answers without reasoning or tool calls.")
            messages[0]["content"] = (
                messages[0]["content"].replace("<image>", IMAGE_TOKEN).replace("<video>", VIDEO_TOKEN)
            )
            images, videos = row.get("images") or [], row.get("videos") or []
            key = json.dumps(row, sort_keys=True)
            if key in verified:
                continue
            verified.add(key)
            layouts.add((bool(images), bool(videos)))

            short_ids, short_mask = miles_mask.get_loss_mask(messages)
            target = [token for token, learn in zip(short_ids, short_mask, strict=True) if learn]
            if not target or target != short_ids[-len(target) :] or sum(short_mask[: -len(target)]) != 0:
                raise ValueError(f"Row {idx}: Miles did not return an unbroken supervised suffix.")
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            assert tokenizer(text, add_special_tokens=False).input_ids == short_ids, f"Row {idx}: mixed prompt render"
            expected = short_ids
            media = {}
            if images or videos:
                expected, media = processor_pool.process_sample_from_paths_in_worker(
                    text, {"image": images, "video": videos}, {}
                )
                media.pop(SFT_HF_IMAGE_REBUILD_SUPPORTED, None)
            if expected[-len(target) :] != target:
                raise ValueError(f"Row {idx}: processor modified the Miles supervised suffix.")
            mask = [0] * (len(expected) - len(target)) + [1] * len(target)

            actual = dataset._process_one(idx)
            assert actual is not None, f"Row {idx}: Relax dropped the sample"
            assert actual.tokens.tolist() == expected, f"Row {idx}: token IDs differ"
            assert actual.loss_mask.tolist() == mask, f"Row {idx}: loss mask differs"
            assert (actual.multimodal_train_inputs or {}).keys() == media.keys(), f"Row {idx}: media keys differ"
            for name, value in media.items():
                torch.testing.assert_close(actual.multimodal_train_inputs[name], value, rtol=0, atol=0)
            logger.info("Row %d: %d tokens, %d supervised, pixels and grids match", idx, len(expected), len(target))
    if (True, False) not in layouts or (False, True) not in layouts or (True, True) not in layouts:
        raise ValueError("Parity data must include image-only, video-only and joint image/video samples.")
    template_out.write_text(template)
    logger.info("Verified %d distinct samples across %d input layouts", len(verified), len(layouts))


def configure_worker(args: argparse.Namespace) -> None:
    import torch
    from megatron.training.determinism import DETERMINISM_ENV_VAR_DEFAULTS

    from relax.utils.logging_utils import get_logger

    expected = DETERMINISM_ENV_VAR_DEFAULTS | {"NCCL_PROTO": "Simple", "NCCL_NVLS_ENABLE": "0", "PYTHONHASHSEED": "0"}
    assert all(os.environ.get(key) == value for key, value in expected.items()), expected
    assert args.deterministic_mode and args.seed == int(os.environ["MINIMAX_PARITY_SEED"])
    assert not args.cross_entropy_loss_fusion and not args.tp_comm_overlap
    assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark

    implementation = os.environ["MINIMAX_MSA_IMPL"]
    if os.environ.get("MINIMAX_TEXT_PARITY") == "1":
        if implementation != "relax" or args.loss_type != "sft" or not args.calculate_per_token_loss:
            raise ValueError("Text parity requires Relax MSA and per-token SFT.")
        from relax.backends.megatron import loss
        from scripts.tools.minimax_m31_text_parity import relax_sft_loss

        loss.sft_loss_function = relax_sft_loss

    if implementation == "miles":
        from relax.models.minimax_m31 import bridge

        bridge.minimax_m31_block_spec = _miles_block_spec
    elif implementation != "relax":
        raise ValueError(f"Unknown MiniMax MSA implementation {implementation!r}.")
    if args.rank == 0:
        get_logger(__name__).info("MiniMax parity: msa=%s seed=%d determinism=%s", implementation, args.seed, expected)


def _miles_block_spec(config, **kwargs):
    from megatron.bridge.models.minimax_m3.minimax_m3_bridge import minimax_m3_block_spec
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec
    from miles_plugins.models.minimax_m3.attention import MiniMaxM3Attention, MiniMaxM3HeadNorm
    from miles_plugins.models.minimax_m3.minimax_m3 import apply_minimax_m3_config

    apply_minimax_m3_config(config, SimpleNamespace(sparse_attention_config=config.msa_config))
    spec = minimax_m3_block_spec(config, **kwargs)
    for layer_id, original in enumerate(spec.layer_specs):
        layer = deepcopy(original)
        attention = layer.submodules.self_attention
        attention.submodules.linear_qkv = TEColumnParallelLinear
        attention.submodules.q_layernorm = MiniMaxM3HeadNorm
        attention.submodules.k_layernorm = MiniMaxM3HeadNorm
        layer.submodules.input_layernorm = TENorm
        layer.submodules.self_attention = ModuleSpec(
            module=MiniMaxM3Attention,
            params={**(attention.params or {}), "sparse": True},
            submodules=attention.submodules,
        )
        spec.layer_specs[layer_id] = layer
    return spec


def _read_metrics(path: Path, steps: int, implementation: str) -> dict[int, dict[str, float]]:
    log = path.read_text()
    implementations = set(re.findall(r"MiniMax parity: msa=(\w+) seed=", log))
    if implementations != {implementation}:
        raise ValueError(f"{path}: expected MSA implementation {implementation!r}, found {sorted(implementations)}")
    metrics = {}
    for match in re.finditer(r"step (\d+): (\{[^\n]+\})", log):
        if "'train/loss'" in match[2]:
            values = ast.literal_eval(match[2])
            metrics[int(match[1])] = {
                key: values[f"train/{key}"] for key in ("loss", "grad_norm", "global_batch_size", "pack_tokens_mean")
            }
    if sorted(metrics) != list(range(steps)):
        raise ValueError(f"{path}: expected {steps} steps and found {sorted(metrics)}")
    if not all(math.isfinite(value) for step in metrics.values() for value in step.values()):
        raise ValueError(f"{path}: nonfinite training metrics")
    return metrics


def _load_training_tensors(checkpoint: Path) -> dict:
    import torch
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = FileSystemReader(checkpoint)
    state = {
        key: torch.empty(entry.size, dtype=entry.properties.dtype)
        for key, entry in reader.read_metadata().state_dict_metadata.items()
        if isinstance(entry, TensorStorageMetadata) and (".adapter." in key or key.startswith("optimizer."))
    }
    adapter_count = sum(".adapter." in key for key in state)
    if adapter_count != 16 or len(state) <= adapter_count:
        raise ValueError(f"{checkpoint}: expected the four-layer adapter and optimizer tensors")
    load(state, storage_reader=reader)
    return state


def compare_runs(logs: list[Path], checkpoints: list[Path], steps: int, *, full: bool = False) -> None:
    import torch

    from relax.utils.logging_utils import get_logger

    if steps < 1:
        raise ValueError("Comparison requires at least one step.")
    if checkpoints[0].resolve() == checkpoints[1].resolve():
        raise ValueError("Comparison requires separate Relax and Miles checkpoint directories.")
    logger = get_logger(__name__)
    left_metrics = _read_metrics(logs[0], steps, "relax")
    right_metrics = _read_metrics(logs[1], steps, "miles")
    if full:
        from scripts.tools.minimax_m31_sft_parity_state import read_rank_digests

        for root in checkpoints:
            if not (root / f"iter_{steps - 1:07d}" / ".metadata").is_file():
                raise ValueError(f"{root}: missing final full checkpoint.")
    mismatches = []
    for index in range(steps):
        fields = [
            name
            for name, value in left_metrics[index].items()
            if struct.pack("!d", value) != struct.pack("!d", right_metrics[index][name])
        ]
        if full:
            left, right = (read_rank_digests(root, index) for root in checkpoints)
            tensors = [
                f"{rank}.{name}" for rank in left for name in left[rank] if left[rank][name] != right[rank][name]
            ]
            num_compared = sum(len(components) for components in left.values())
            compared = f"{num_compared} rank/component state digests"
        else:
            left, right = (_load_training_tensors(root / f"iter_{index:07d}") for root in checkpoints)
            if left.keys() != right.keys():
                raise ValueError(f"Step {index + 1}: different checkpoint tensor keys: {left.keys() ^ right.keys()}")
            tensors = [
                key
                for key, value in left.items()
                if value.shape != right[key].shape
                or value.dtype != right[key].dtype
                or not torch.equal(value.reshape(-1).view(torch.uint8), right[key].reshape(-1).view(torch.uint8))
            ]
            compared = f"{len(left)} checkpoint tensors"
        if fields or tensors:
            mismatches.append(index + 1)
            logger.error("Step %d differs: metrics=%s tensors=%s", index + 1, fields, tensors)
        else:
            logger.info(
                "Step %d: loss=%r grad_norm=%r; %s match",
                index + 1,
                left_metrics[index]["loss"],
                left_metrics[index]["grad_norm"],
                compared,
            )
    if mismatches:
        raise SystemExit(f"Bitwise mismatch at steps {mismatches}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--checkpoint", required=True, type=Path)
    prepare.add_argument("--data-dir", required=True, type=Path)
    prepare.add_argument("--make-fixture", action="store_true")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--logs", nargs=2, required=True, type=Path)
    compare.add_argument("--checkpoints", nargs=2, required=True, type=Path)
    compare.add_argument("--steps", type=int, default=10)
    compare.add_argument("--full", action="store_true", help="Compare full-SFT rank digests and final checkpoints.")
    args = parser.parse_args()
    if args.command == "prepare":
        args.data_dir.mkdir(parents=True, exist_ok=True)
        data = args.data_dir / "minimax-m3.1-flash-sft.jsonl"
        if args.make_fixture:
            make_fixture(data)
        verify_inputs(args.checkpoint, data, args.data_dir / "miles-mask.jinja")
    else:
        compare_runs(args.logs, args.checkpoints, args.steps, full=args.full)
