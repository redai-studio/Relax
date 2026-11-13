"""Run native ReTool training and K8 evaluation."""

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from examples.retool.lifecycle import close_owned_children, require_job_success


def put(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser()
    for n in ["root", "temp", "model"]:
        p.add_argument("--" + n, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--completed-sft", type=Path)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--sft-updates", type=int, default=50)
    p.add_argument("--rl-updates", type=int, default=200)
    p.add_argument("--sandbox-base", type=Path, required=True)
    p.add_argument("--stop-after-updates", type=int)
    p.add_argument("--eval-config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue-evaluation", action="store_true")
    a = p.parse_args()
    a.source = str(Path(__file__).resolve().parents[2])
    a.helpers = a.reader = str(Path(__file__).with_name("resources.py"))
    a.sandbox_image = sys.executable
    root, rt, source = Path(a.root), Path(a.temp), Path(a.source)
    ray_socket = rt / "ray/session_2026-11-10_00-00-00_123456_1234567/sockets/plasma_store"
    if len(os.fsencode(ray_socket)) > 107:
        p.error("Use a shorter RETOOL_RUNTIME_PARENT for the Ray sockets")
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = a.resume_from or a.completed_sft
    if parent:
        parent_lock = (parent / "pipeline.lock").open("a")
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / "HUMAN_STOP").exists()

    def stopped(sig, frame):
        raise InterruptedError("Owned validation interrupted")

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    reader = load_module("resource_reader", a.reader)
    helpers = load_module("owned_helpers", a.helpers)
    e = {k: v for k, v in os.environ.items() if not k.lower().endswith("proxy")}
    for k in ["RAY_ADDRESS", "RAY_JOB_SUBMISSION_ID", "RELAX_ENTRYPOINT_MODE", "WORKING_DIR", "SSH_AUTH_SOCK"]:
        e.pop(k, None)
    e.update(
        RT=str(rt),
        ROOT=str(root),
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        CLEARML_OFFLINE_MODE="1",
        RAY_USAGE_STATS_ENABLED="0",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        RELAX_SKIP_CLUSTER_CLEANUP="1",
        RELAX=str(source),
        PYTHONPATH=os.pathsep.join(filter(None, (str(source), os.environ.get("PYTHONPATH")))),
        TMPDIR=str(rt / "tmp"),
        RAY_TMPDIR=str(rt / "ray"),
        DATA_DIR=str(root / "data"),
        SAVE_DIR=str(root / "checkpoints"),
    )
    e["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"
    (rt / "tmp").mkdir(exist_ok=True)

    def container(command, gpu=False):
        if command[0] == "python3":
            return [sys.executable] + command[1:]
        if command[0] == "ray":
            return [sys.executable, "-m", "ray.scripts.scripts"] + command[1:]
        return command

    def run(name, command, env):
        start = time.monotonic()
        with (root / (name + ".log")).open("x") as out:
            # Checkpoint CLI stdout is a machine-readable integer; never mix
            # torch/import/deprecation warnings from stderr into that result.
            err = (root / (name + ".stderr.log")).open("x") if name.startswith("checkpoint_") else None
            proc = subprocess.Popen(
                command,
                env=env,
                cwd=source,
                stdout=out,
                stderr=err if err else subprocess.STDOUT,
                start_new_session=True,
            )
            put(
                root / (name + "_launch.json"),
                dict(pid=proc.pid, identity=helpers.identity(proc.pid), command=command),
            )
            try:
                rc = proc.wait(timeout=max(1, a.deadline - time.time()))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=60)
                if err:
                    err.close()
        put(
            root / (name + "_wait.json"),
            dict(returncode=rc, wall_seconds=time.monotonic() - start, actual_parent_wait=True),
        )
        if rc:
            raise RuntimeError(f"{name} failed exit{rc}; see {name}.log")

    smoke = a.smoke
    if a.stop_after_updates is not None:
        assert smoke and 0 < a.stop_after_updates <= a.rl_updates
    if a.continue_evaluation:
        from examples.retool.evaluation import run_comparison

        run_comparison(a, e, container, run, reader, helpers)
        put(
            root / "PIPELINE_COMPLETE_ACTUAL.json",
            dict(status="NATIVE_TRAIN_AND_AUTOMATIC_K8_COMPLETE", not_formal_training=smoke, resources_released=True),
        )
        return
    assert (a.sft_updates, a.rl_updates) == (50, 200)
    put(
        root / "test_contract.json",
        dict(
            source_commit=os.environ["RETOOL_SOURCE_COMMIT"],
            recovery_compatible_parent_commit="2b8f3a9840119d461f983619e9d68a7e3619da3e",
            user_selected_backend="SGLang",
            model="Qwen/Qwen2.5-7B-Instruct",
            bounded_smoke=smoke,
            sft_updates=a.sft_updates,
            sft_batch=80,
            sft_pool=2000,
            rl_updates=a.rl_updates,
            rl_batch=64,
            K=16,
            rl_pool=17903,
            tool_backend="Landlock SandboxFusion-compatible /run_code",
        ),
    )
    from examples.retool.recovery import hf_identity

    model = a.model
    if not parent:
        hf_identity(model)
        if (root / "checkpoints").exists():
            raise FileExistsError("Fresh training requires a new checkpoint root")
    modes = ["sft", "rl"]
    recovery = {}
    if parent:
        run(
            "recovery_inspect",
            container(
                [
                    "python3",
                    str(Path(__file__).with_name("recovery.py")),
                    "--parent",
                    str(parent),
                    "--output",
                    str(root),
                    "--sft-target",
                    str(a.sft_updates),
                    "--rl-target",
                    str(a.rl_updates),
                ]
            ),
            dict(e, CUDA_VISIBLE_DEVICES=""),
        )
        recovery = json.loads((root / "RECOVERY_PLAN.json").read_text())
        initial_root = Path(recovery["sft"]["original"]).parents[1]
        e["DATA_DIR"] = str(initial_root / "data")
        model = str(initial_root / "model")
        if recovery["sft"]["action"] == "SKIP_COMPLETE":
            model = recovery["sft"]["hf"]["path"]
            modes.remove("sft")
            put(
                root / "completed_sft_parent.json",
                dict(parent=str(parent), optimizer_updates=a.sft_updates, weights=model, rerun_sft=False),
            )
        if recovery["rl"]["action"] == "SKIP_COMPLETE":
            modes.remove("rl")
    for mode in modes:
        assert not (root / "HUMAN_STOP").exists()
        updates = a.sft_updates if mode == "sft" else a.rl_updates
        stop_target = (a.stop_after_updates or updates) if mode == "rl" else updates
        resume = recovery.get(mode, {})
        start = resume.get("completed", 0)
        load = resume.get("load")
        resume_args = (
            ["--load", load, "--start-rollout-id", str(start), "--use-checkpoint-opt_param-scheduler"] if load else []
        )
        stage = root / ("gpu_" + mode)
        owner = helpers.identity(os.getpid())
        with (reader.REG / "allocator.lock").open("a") as allocation:
            fcntl.flock(allocation, fcntl.LOCK_EX)
            stage.mkdir()
            claims = helpers.claim_paths(reader.REG, "retool_" + root.name, stage)
            for gpu, path in enumerate(claims):
                put(
                    path,
                    dict(
                        **owner,
                        gpu=gpu,
                        task="retool_training",
                        release_file=str(stage / "resource_release.json"),
                    ),
                )
            put(stage / "admission.json", dict(owner=owner, claims=list(map(str, claims))))
        children = []
        success = False

        def start_child(name, cmd, env, cwd=None):
            out = (stage / (name + ".log")).open("x")
            proc = subprocess.Popen(
                cmd, cwd=cwd or source, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
            )
            children.append((name, proc, helpers.identity(proc.pid), out))
            return proc

        try:
            g = dict(e, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", EXP_NAME=mode, MODEL_DIR=model)
            g["RETOOL_RAY_SUBMISSION_ID"] = "retool-" + uuid.uuid4().hex
            ray = start_child(
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
                        "--temp-dir=" + str(rt / "ray"),
                        "--disable-usage-stats",
                        "--block",
                    ],
                    True,
                ),
                g,
            )
            for _ in range(180):
                if ray.poll() is not None:
                    raise RuntimeError("Owned Ray head exited")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:29265/api/version", timeout=2):
                        break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("Own Ray head startup timeout")
            keys = [
                "PYTHONPATH",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "WANDB_MODE",
                "WANDB_DISABLED",
                "CLEARML_OFFLINE_MODE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TMPDIR",
                "RAY_TMPDIR",
                "NCCL_SOCKET_IFNAME",
                "DATA_DIR",
                "PYTHONHOME",
                "IMAGE_SITE",
                "IMAGE_PYTHON",
                "MEGATRON",
                "PATH",
                "LD_LIBRARY_PATH",
                "CUDA_HOME",
                "RAY_OVERRIDE_JOB_RUNTIME_ENV",
            ]
            runtime_env = {
                "worker_process_setup_hook": "relax.utils.logging_utils.install_asyncio_noise_filter",
                "env_vars": {k: g[k] for k in keys if k in g},
            }
            g.update(
                RAY_ADDRESS="http://127.0.0.1:29265",
                RELAX_ENTRYPOINT_MODE="ray-job",
                RUNTIME_ENV_JSON=json.dumps(runtime_env),
                MASTER_ADDR="127.0.0.1",
            )
            if mode == "rl":
                old = source
                assert a.sandbox_base.stat().st_mode & 0o777 == 0o711
                sb = dict(e, CUDA_VISIBLE_DEVICES="")
                sandbox = start_child(
                    "sandbox",
                    [
                        a.sandbox_image,
                        "-m",
                        "examples.retool.execution_service",
                        "--root",
                        str(a.sandbox_base / "rl_service"),
                        "--port",
                        "29080",
                    ],
                    sb,
                    cwd=old,
                )
                for _ in range(90):
                    if sandbox.poll() is not None:
                        raise RuntimeError("Owned sandbox exited")
                    try:
                        request = urllib.request.Request("http://127.0.0.1:29080/health")
                        with urllib.request.urlopen(request, timeout=1):
                            break
                    except urllib.error.HTTPError:
                        break
                    except Exception:
                        time.sleep(1)
                else:
                    raise RuntimeError("Sandbox startup timeout")
                g["SANDBOX_FUSION_URL"] = "http://127.0.0.1:29080/run_code"
            cmd = container(
                [
                    "bash",
                    f"examples/retool/scripts/run-qwen25-7B-{mode if mode == 'sft' else 'grpo'}-8xgpu.sh",
                    "--num-rollout",
                    str(stop_target),
                    "--save-interval",
                    "1" if smoke else "25",
                    "--use-tensorboard",
                    "--no-use-metrics-service",
                    "--no-use-clearml",
                    "--no-use-wandb",
                ]
                + resume_args,
                True,
            )
            train = start_child("train", cmd, g)
            rc = train.wait(timeout=max(1, a.deadline - time.time()))
            put(stage / "training_actual_wait.json", dict(returncode=rc, actual_parent_wait=True))
            if rc:
                log = stage / "train.log"
                if log.is_file():
                    with log.open("rb") as stream:
                        stream.seek(0, 2)
                        stream.seek(max(0, stream.tell() - 8192))
                        print("TRAIN_ERROR_TAIL_BEGIN\n" + stream.read().decode("utf-8", "replace"), flush=True)
                raise RuntimeError(f"{mode} training failed exit{rc}")
            require_job_success(g, stage / "ray_job_status.json", a.deadline)
            run(
                "checkpoint_" + mode,
                container(
                    ["python3", "-m", "examples.retool.checkpoint", str(root / "checkpoints" / mode)]
                    + (["--sft"] if mode == "sft" else [])
                ),
                dict(e, CUDA_VISIBLE_DEVICES=""),
            )
            assert (root / ("checkpoint_" + mode + ".log")).read_text().strip() == str(stop_target)
            hf_identity(root / "checkpoints" / mode / f"hf_{stop_target - 1}")
            success = True
        finally:
            closed = close_owned_children(children, helpers, timeout=60)
            deadline = time.monotonic() + 120
            while any(helpers.group_members(x[2]["pgid"]) for x in children) and time.monotonic() < deadline:
                time.sleep(0.5)
            empty = not any(helpers.group_members(x[2]["pgid"]) for x in children)
            put(
                stage / "workers_closed.json",
                dict(children=closed, exact_owned_pg_empty=empty, native_actor_individual_parent_wait="NOT_RECORDED"),
            )
            if not empty or not all(x["actual_parent_wait"] for x in closed):
                raise RuntimeError("Owned children were not fully reaped; retain claims")
            put(
                stage / "resource_release.json",
                dict(complete=success, exact_owned_pg_empty=True, actual_driver_wait=True, owner=owner),
            )
            deadline = time.monotonic() + 180
            while True:
                try:
                    fresh = reader.resources()
                    if not fresh["live_unreleased_claims"]:
                        break
                except (AssertionError, RuntimeError, FileNotFoundError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Owned resources were not released")
                time.sleep(0.5)
            put(stage / "post_release_resources.json", fresh)
        if mode == "sft":
            model = str(root / f"checkpoints/sft/hf_{a.sft_updates - 1}")
    from examples.retool.evaluation import run_comparison

    run_comparison(a, e, container, run, reader, helpers)
    put(
        root / "PIPELINE_COMPLETE_ACTUAL.json",
        dict(
            status="NATIVE_TRAIN_AND_AUTOMATIC_K8_COMPLETE",
            not_formal_training=smoke,
            resources_released=True,
            sft_parent=str(parent) if parent else None,
        ),
    )


if __name__ == "__main__":
    main()
