"""Read-only checkpoint selection; fresh outputs, no history/RNG
reconstruction."""

import argparse
import json
import math
import stat
import struct
from pathlib import Path

from examples.retool.lifecycle import require_training_success


COMMIT = "2b8f3a9840119d461f983619e9d68a7e3619da3e"


def hf_identity(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    assert [config[k] for k in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size")] == [
        "qwen2",
        3584,
        28,
        152064,
    ]
    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert len(index["weight_map"]) == 339
    tensors = {}
    total = 0
    for name in set(index["weight_map"].values()):
        assert Path(name).name == name
        shard = path / name
        assert shard.resolve().is_relative_to(path.resolve()), "External/dangling shard"
        with shard.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            assert 0 < length < 10000000
            header = json.loads(f.read(length))
        spans = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            assert key not in tensors and index["weight_map"][key] == name
            assert value["dtype"] == "BF16"
            start, end = value["data_offsets"]
            assert end - start == math.prod(value["shape"]) * 2
            spans.append((start, end))
            tensors[key] = value["shape"]
            total += end - start
        spans.sort()
        assert spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        assert shard.stat().st_size == 8 + length + spans[-1][1]
    assert set(tensors) == set(index["weight_map"]) and total == index["metadata"]["total_size"]
    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert (path / name).is_file() and (path / name).stat().st_size
    assert (path / "chat_template.jinja").is_file() or json.loads((path / "tokenizer_config.json").read_text()).get(
        "chat_template"
    )
    return dict(
        path=str(path),
        tensors=len(tensors),
        shards=len(set(index["weight_map"].values())),
        tensor_bytes=total,
        header_index_length_check="PASS",
    )


def validate_native(view, sft):
    from torch.distributed.checkpoint import FileSystemReader

    from examples.retool.checkpoint import validate_checkpoint

    step = validate_checkpoint(view, sft=sft)
    assert step is not None
    checkpoint = view / f"iter_{step:07d}"
    metadata = FileSystemReader(checkpoint).read_metadata()
    keys = [str(k) for k in metadata.state_dict_metadata]
    # Native MCore keys are unprefixed (not "model.*"); TE layers are stacked.
    for required in (
        "embedding.word_embeddings.weight",
        "decoder.layers.",
        "output_layer.weight",
        "optimizer",
        "rng_state",
    ):
        assert any(required in k for k in keys), "Missing native " + required
    # Verify every DCP referenced byte span exists without reading all Adam tensors.
    assert metadata.storage_data
    extents = {}
    for item in metadata.storage_data.values():
        extents[item.relative_path] = max(extents.get(item.relative_path, 0), item.offset + item.length)
    for name, end in extents.items():
        file = checkpoint / name
        assert file.resolve(strict=True).is_relative_to(checkpoint.resolve(strict=True))
        info = file.stat()
        assert stat.S_ISREG(info.st_mode) and info.st_size >= end
    return step


def select_checkpoint(roots, mode, target, views, validator=validate_native):
    rejected = []
    candidates = []
    for root in map(Path, roots):
        tracker = root / "latest_checkpointed_iteration.txt"
        try:
            committed = int(tracker.read_text().strip())
        except FileNotFoundError:
            continue
        assert 0 <= committed < target, "Checkpoint exceeds frozen total target"
        # Producer's exact latest pointer includes short debug runs (e.g. CP2).
        # Only the fixed historical save boundaries are attempted as fallbacks.
        points = [committed] + list(range(24, committed, 25))[::-1]
        candidates.extend((step, root, root / f"iter_{step:07d}") for step in points)
    for step, root, path in sorted(candidates, key=lambda x: x[0], reverse=True):
        view = views / (mode + "_" + str(len(rejected)))
        view.mkdir(parents=True)
        (view / path.name).symlink_to(path.resolve(), target_is_directory=True)
        (view / "latest_checkpointed_iteration.txt").write_text(str(step))
        state_dir = "sft_data_state" if mode == "sft" else "dataset"
        (view / state_dir).symlink_to((root / state_dir).resolve(), target_is_directory=True)
        try:
            assert validator(view, mode == "sft") == step
            state_file = root / state_dir / (f"{step + 1:07d}.json" if mode == "sft" else f"retool_{step}.json")
            state = json.loads(state_file.read_text())
            if mode == "sft":
                assert state["next_step"] == step + 1
            else:
                assert (state["cursor"], state["batch"], state["K"], state["seed"]) == (step + 1, 64, 16, 20261031)
                assert state.get("rng_state_version") == 1 and state.get("rng_state"), (
                    "Native data RNG snapshot missing; history reconstruction forbidden"
                )
            return dict(step=step, completed=step + 1, load=str(view), original=str(root), rejected=rejected)
        except (OSError, ValueError, AssertionError, KeyError, RuntimeError) as exc:
            rejected.append(dict(checkpoint=str(path), reason=str(exc)))
    assert not candidates and not rejected, "Committed checkpoints exist but none accepted: " + json.dumps(rejected)
    # plan() separately rejects fresh RL if any actual STEP_END was written.
    return dict(
        step=None,
        completed=0,
        load=None,
        original=None,
        rejected=rejected,
        initialization_only_files=[],
        directory_enumeration=False,
    )


def plan(parent, output, sft_target, rl_target, validator=validate_native, hf_validator=hf_identity):
    parent, output = Path(parent), Path(output)
    assert not (parent / "HUMAN_STOP").exists(), "Explicit human stop cannot be revived"
    contract = json.loads((parent / "test_contract.json").read_text())
    assert contract.get("recovery_compatible_parent_commit", contract["source_commit"]) == COMMIT
    assert (
        contract["sft_updates"],
        contract["rl_updates"],
        contract["sft_pool"],
        contract["rl_pool"],
        contract["rl_batch"],
        contract["K"],
    ) == (sft_target, rl_target, 2000, 17903, 64, 16)
    old = json.loads((parent / "RECOVERY_PLAN.json").read_text()) if (parent / "RECOVERY_PLAN.json").is_file() else {}
    roots = {m: [str(parent / "checkpoints" / m), *old.get(m + "_ancestors", [])] for m in ("sft", "rl")}
    selected = {
        m: select_checkpoint(roots[m], m, t, output / "resume_inputs", validator)
        for m, t in [("sft", sft_target), ("rl", rl_target)]
    }
    sft = selected["sft"]
    assert sft["completed"] > 0, "Recovery parent has no valid SFT state; explicit clean-start classification required"
    if sft["completed"] == sft_target:
        require_training_success(Path(sft["original"]).parents[1], "sft", sft_target)
        sft["hf"] = hf_validator(Path(sft["original"]) / f"hf_{sft_target - 1}")
        sft["action"] = "SKIP_COMPLETE"
    else:
        sft["action"] = "RESUME_NATIVE"
    rl = selected["rl"]
    if not rl["completed"]:
        for old_root in roots["rl"]:
            require_zero_update_failure(Path(old_root))
    assert not rl["completed"] or sft["action"] == "SKIP_COMPLETE", "RL checkpoint without complete SFT lineage"
    if rl["completed"] == rl_target:
        require_training_success(Path(rl["original"]).parents[1], "rl", rl_target)
        rl["hf"] = hf_validator(Path(rl["original"]) / f"hf_{rl_target - 1}")
        rl["action"] = "SKIP_COMPLETE"
    else:
        rl["action"] = "RESUME_NATIVE" if rl["completed"] else "FRESH_FROM_SFT"
    return dict(
        version=4,
        parent=str(parent),
        sft=sft,
        rl=rl,
        sft_ancestors=roots["sft"],
        rl_ancestors=roots["rl"],
        total_targets=dict(sft=sft_target, rl=rl_target),
        history_replay=False,
    )


def require_zero_update_failure(root):
    """Missing state is unknown, not evidence that an existing run made zero
    updates."""
    run = root.parents[1] / "gpu_rl"
    if not root.exists() and not run.exists():
        return
    wait = json.loads((run / "training_actual_wait.json").read_text())
    assert (
        wait.get("actual_parent_wait") is True and type(wait.get("returncode")) is int and wait["returncode"] != 0
    ), "No failed native exit evidence; refusing an unproven zero-update restart"
    with (run / "train.log").open() as stream:
        saw_log = False
        for line in stream:
            saw_log = True
            assert "STEP_END" not in line, (
                "Uncheckpointed optimizer updates exist; explicit clean-lineage classification required"
            )
    assert saw_log, "Empty native log cannot establish zero optimizer updates"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sft-target", type=int, required=True)
    p.add_argument("--rl-target", type=int, required=True)
    a = p.parse_args()
    result = plan(a.parent, a.output, a.sft_target, a.rl_target)
    with (Path(a.output) / "RECOVERY_PLAN.json").open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
