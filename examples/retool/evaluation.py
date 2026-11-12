# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native evaluation-only before/after comparison, never optimizer work."""

import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from examples.retool.execution_service import put
from examples.retool.lifecycle import close_owned_children, require_job_success, require_training_success


def summarize(rollout_id, args, data, extra_metrics=None):
    config = json.loads(Path(os.environ["RETOOL_EVAL_MANIFEST"]).read_text())
    expected = config["cases"]
    assert set(data) == set(expected), "Missing or unexpected benchmark"
    panels = {}
    for name, entry in data.items():
        samples = sorted(entry["samples"], key=lambda sample: sample.index)
        n = expected[name]
        assert len(samples) == n * 8 and [sample.index for sample in samples] == list(range(n * 8))
        assert all(sample.status.value in ("completed", "truncated") for sample in samples)
        correct = [bool(sample.reward["acc"]) for sample in samples]
        assert len(entry["rewards"]) == len(samples)
        called = [sample.metadata["tool_calls"] > 0 for sample in samples]
        pass8 = sum(any(correct[i : i + 8]) for i in range(0, len(samples), 8))
        counts = {}
        for sample in samples:
            for status, count in sample.metadata["tool_feedback_counts"].items():
                counts[status] = counts.get(status, 0) + count
        panels[name] = dict(
            questions=n,
            trajectories=len(samples),
            correct_trajectories=sum(correct),
            mean_pass1=sum(correct) / len(samples),
            pass8_correct_questions=pass8,
            pass8=pass8 / n,
            trajectories_calling_tools=sum(called),
            tool_call_rate=sum(called) / len(samples),
            tool_submissions=sum(sample.metadata["tool_calls"] for sample in samples),
            tool_feedback=counts,
        )
        directory = Path(args.save) / name
        directory.mkdir()
        for sample in samples:
            put(
                directory / f"case{sample.index:05d}.json",
                dict(
                    question_index=sample.index // 8,
                    sample_id=sample.index % 8,
                    sample=sample.to_dict(),
                    original_native_score=sample.reward,
                ),
            )
    put(
        Path(args.save) / "evaluation_complete.json",
        dict(
            status="ALL_NATIVE_K8_PANELS_ACCEPTED",
            model=args.hf_checkpoint,
            K=8,
            panels=panels,
            training_updates=0,
            rescoring=False,
            smoke=config["smoke"],
            scoring="unchanged native ReTool acc, penalty separate",
        ),
    )
    from relax.utils import tracking_utils

    metrics = {
        f"eval/{name}/{key}": panel[key]
        for name, panel in panels.items()
        for key in ("pass8", "mean_pass1", "tool_call_rate")
    }
    tracking_utils.log(args, dict(metrics, **{"eval/step": rollout_id}), step_key="eval/step")
    return True


def trained_checkpoint(root: Path, updates: int, sft_updates: int = 50) -> Path:
    """Resolve completed recovery stages without expecting a copied
    checkpoint."""
    source = sft_source = root
    final = root / "checkpoints/rl" / f"hf_{updates - 1}"
    plan = root / "RECOVERY_PLAN.json"
    if plan.is_file():
        selected = json.loads(plan.read_text())
        sft = selected["sft"]
        if sft["action"] == "SKIP_COMPLETE":
            assert sft["completed"] == sft_updates
            sft_source = Path(sft["original"]).parents[1]
        rl = selected["rl"]
        if rl["action"] == "SKIP_COMPLETE":
            assert rl["completed"] == updates
            final = Path(rl["hf"]["path"])
            source = Path(rl["original"]).parents[1]
    require_training_success(sft_source, "sft", sft_updates)
    require_training_success(source, "rl", updates)
    return final


def completed_evaluation(output: Path, model: str, cases: dict[str, int], smoke: bool) -> dict | None:
    """Only reuse full panels with successful native execution and cleanup."""
    try:
        result = json.loads((output / "evaluation_complete.json").read_text())
        wait = json.loads((output / "driver_wait.json").read_text())
        job = json.loads((output / "ray_job_status.json").read_text())
        closed = json.loads((output / "closed.json").read_text())
        assert result["status"] == "ALL_NATIVE_K8_PANELS_ACCEPTED"
        assert Path(result["model"]).resolve() == Path(model).resolve()
        assert result["K"] == 8 and result["smoke"] == smoke
        assert set(result["panels"]) == set(cases)
        assert all(
            result["panels"][name]["questions"] == n and result["panels"][name]["trajectories"] == n * 8
            for name, n in cases.items()
        )
        assert wait["actual_parent_wait"] and wait["returncode"] == 0
        assert job["status"] == "SUCCEEDED"
        assert closed["owned_PG_empty"] and closed["children"]
        assert all(child["actual_parent_wait"] for child in closed["children"])
        return result
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return None


def evaluation_attempt(root: Path, temp: Path, sandbox_base: Path, arm: str) -> tuple[Path, Path, Path]:
    """Preserve partial attempts; reserve fresh output, Ray and sandbox
    paths."""
    base = root / "evaluation" / arm
    attempt = 0
    while True:
        suffix = "" if attempt == 0 else f"_attempt{attempt:04d}"
        output = base if attempt == 0 else base / f"attempt{attempt:04d}"
        runtime = temp / (arm + suffix)
        sandbox = sandbox_base / ("eval_" + arm + suffix)
        if not any(path.exists() for path in (output, runtime, sandbox)):
            output.mkdir(parents=True)
            runtime.mkdir()
            return output, runtime, sandbox
        attempt += 1


def run_comparison(a, env, container, run, reader, helpers):
    from examples.retool.recovery import hf_identity

    root = Path(a.root)
    config = json.loads(a.eval_config.read_text())
    assert set(config["datasets"]) == {"gsm8k", "math500", "aime2025", "aime2026"}
    expected = {"gsm8k": 1319, "math500": 500, "aime2025": 30, "aime2026": 30}
    prepared = root / "evaluation_data"
    prepared.mkdir(exist_ok=True)
    for name, source in config["datasets"].items():
        rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
        assert len(rows) == expected[name]
        assert len({row["metadata"]["question"] for row in rows}) == expected[name]
        if a.smoke:
            rows = rows[:1]
        target = prepared / f"{name}.jsonl"
        if target.exists():
            assert [json.loads(line) for line in target.read_text().splitlines()] == rows
        else:
            with target.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
    count = {name: (1 if a.smoke else n) for name, n in expected.items()}
    manifest = root / "evaluation_manifest.json"
    value = dict(cases=count, smoke=a.smoke, K=8, official_model=str(Path(config["official_model"]).resolve()))
    if manifest.exists():
        assert json.loads(manifest.read_text()) == value
    else:
        put(manifest, value)
    final = trained_checkpoint(root, a.stop_after_updates or a.rl_updates, a.sft_updates)
    models = {"official": config["official_model"], "trained": str(final)}
    summaries = {}
    for arm, model in models.items():
        hf_identity(model)
        base = root / "evaluation" / arm
        candidates = [base, *sorted(base.glob("attempt[0-9][0-9][0-9][0-9]"))]
        accepted = next(
            (result for path in candidates if (result := completed_evaluation(path, model, count, a.smoke))),
            None,
        )
        if accepted is not None:
            summaries[arm] = accepted
            continue
        output, runtime, sandbox = evaluation_attempt(root, Path(a.temp), a.sandbox_base, arm)
        assert (
            len(os.fsencode(str(runtime / "ray/session_2026-11-05_15-00-00_123456_1234567/sockets/plasma_store")))
            <= 107
        )
        import yaml

        cfg = {
            "eval": {
                "defaults": dict(
                    input_key="prompt",
                    label_key="label",
                    metadata_key="metadata",
                    n_samples_per_eval_prompt=8,
                    temperature=1,
                    top_p=1,
                    top_k=-1,
                    max_response_len=8192,
                ),
                "datasets": [
                    dict(
                        name=name,
                        path=str(prepared / f"{name}.jsonl"),
                        metadata_overrides={"tool_enabled": True},
                        custom_generate_function_path="examples.retool.rollout.generate",
                    )
                    for name in count
                ],
            }
        }
        yaml_path = output / "datasets.yaml"
        with yaml_path.open("x") as stream:
            yaml.safe_dump(cfg, stream)
        g = dict(
            env,
            RETOOL_EVAL_GPUS="1" if a.smoke else "8",
            CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
            MODEL_DIR=model,
            RETOOL_RAY_SUBMISSION_ID="retool-eval-" + uuid.uuid4().hex,
            SAVE_DIR=str(output),
            RETOOL_EVAL_MANIFEST=str(manifest),
            TMPDIR=str(runtime),
            RAY_TMPDIR=str(runtime / "ray"),
            SANDBOX_FUSION_URL="http://127.0.0.1:29080/run_code",
            RAY_ADDRESS="http://127.0.0.1:29265",
            RELAX_ENTRYPOINT_MODE="ray-job",
            MASTER_ADDR="127.0.0.1",
        )
        keys = [
            "PYTHONPATH",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "WANDB_MODE",
            "WANDB_DISABLED",
            "CLEARML_OFFLINE_MODE",
            "OMP_NUM_THREADS",
            "TMPDIR",
            "RAY_TMPDIR",
            "NCCL_SOCKET_IFNAME",
            "DATA_DIR",
            "PATH",
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "MEGATRON",
            "RAY_OVERRIDE_JOB_RUNTIME_ENV",
            "RETOOL_EVAL_MANIFEST",
            "RETOOL_EVAL_GPUS",
            "SANDBOX_FUSION_URL",
        ]
        g["RUNTIME_ENV_JSON"] = json.dumps({"env_vars": {key: g[key] for key in keys if key in g}})
        children = []

        def start(name, command, child_env):
            log = (output / f"{name}.log").open("x")
            proc = subprocess.Popen(
                command, env=child_env, cwd=a.source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            children.append((name, proc, helpers.identity(proc.pid), log))
            return proc

        reader.resources()
        try:
            ray = start(
                "ray",
                container(
                    [
                        "ray",
                        "start",
                        "--head",
                        "--num-gpus=8",
                        "--num-cpus=64",
                        "--port=29379",
                        "--dashboard-host=127.0.0.1",
                        "--dashboard-port=29265",
                        "--temp-dir=" + str(runtime / "ray"),
                        "--disable-usage-stats",
                        "--block",
                    ],
                    True,
                ),
                g,
            )
            for _ in range(180):
                assert ray.poll() is None
                try:
                    with urllib.request.urlopen("http://127.0.0.1:29265/api/version", timeout=2):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("Own evaluation Ray head")
            service_env = dict(g, CUDA_VISIBLE_DEVICES="")
            service = start(
                "sandbox",
                [
                    sys.executable,
                    "-m",
                    "examples.retool.execution_service",
                    "--root",
                    str(sandbox),
                    "--port",
                    "29080",
                ],
                service_env,
            )
            for _ in range(90):
                assert service.poll() is None
                try:
                    with urllib.request.urlopen("http://127.0.0.1:29080/health", timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("Own evaluation sandbox")
            process = start(
                "eval", ["bash", "examples/retool/scripts/evaluate.sh", model, str(output), str(yaml_path)], g
            )
            code = process.wait(timeout=max(1, a.deadline - time.time()))
            put(output / "driver_wait.json", dict(returncode=code, actual_parent_wait=True, pid=process.pid))
            assert code == 0, "Native evaluation failed"
            require_job_success(g, output / "ray_job_status.json", a.deadline)
        finally:
            closure = close_owned_children(children, helpers)
            deadline = time.monotonic() + 120
            while (
                any(helpers.group_members(ident["pgid"]) for _, _, ident, _ in children)
                and time.monotonic() < deadline
            ):
                time.sleep(0.5)
            empty = not any(helpers.group_members(ident["pgid"]) for _, _, ident, _ in children)
            put(
                output / "closed.json",
                dict(children=closure, owned_PG_empty=empty, native_Ray_actor_individual_wait="NOT_RECORDED"),
            )
            if not empty or not all(child["actual_parent_wait"] for child in closure):
                raise RuntimeError("Owned evaluation children were not fully reaped")
        summaries[arm] = completed_evaluation(output, model, count, a.smoke)
        assert summaries[arm] is not None, "Incomplete evaluation or missing successful exit receipts"
        for _ in range(120):
            try:
                reader.resources()
                break
            except AssertionError:
                time.sleep(1)
        else:
            raise RuntimeError("Evaluation resources not released")
    comparison = dict(
        status="OFFICIAL_VS_TRAINED_K8_COMPLETE",
        arms=summaries,
        interpretation="SFT plus RL total change, not pure RL gain",
        smoke=a.smoke,
    )
    target = root / "comparison.json"
    if target.exists():
        assert json.loads(target.read_text()) == comparison
    else:
        put(target, comparison)
