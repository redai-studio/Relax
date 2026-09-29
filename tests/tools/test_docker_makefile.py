# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PREFIX = "registry.example/relax"
VARIANTS = ("cu12", "cu13-blackwell", "cu13-hopper")
MakeResult = tuple[subprocess.CompletedProcess[str], list[list[str]]]


def image_tag(stage: str, variant: str) -> str:
    return f"{IMAGE_PREFIX}:{stage}-20260101-12345678-{variant}"


@pytest.fixture
def docker_make(tmp_path: Path) -> Callable[..., MakeResult]:
    log = tmp_path / "docker.jsonl"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['DOCKER_CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args[:2] in [['image', 'inspect'], ['manifest', 'inspect']]:\n"
        "    sys.exit(0 if args[-1] in json.loads(os.environ['EXISTING_TAGS']) else 1)\n"
        "if args[0] == 'build' and args[args.index('-t') + 1] == os.environ['FAIL_TAG']:\n"
        "    sys.exit(7)\n"
    )
    docker.chmod(0o755)

    def run(
        target: str,
        *overrides: str,
        push: bool = False,
        existing: tuple[str, ...] = (),
        fail_tag: str = "",
        environment: dict[str, str] | None = None,
    ) -> MakeResult:
        log.write_text("")
        env = {
            **os.environ,
            "DOCKER_CALL_LOG": str(log),
            "EXISTING_TAGS": json.dumps(existing),
            "FAIL_TAG": fail_tag,
            "BASE_IMAGE": "cu12-base",
            "CU13_BASE_IMAGE": "cu13-base",
        }
        for name in (
            "MAKEFLAGS",
            "MFLAGS",
            "MAKELEVEL",
            "TRAIN_IMAGE",
            "CU13_BLACKWELL_TRAIN_IMAGE",
            "CU13_HOPPER_TRAIN_IMAGE",
        ):
            env.pop(name, None)
        env.update(environment or {})
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                target,
                f"DOCKER={docker}",
                "REGISTRY=registry.example/",
                "IMAGE_REPOSITORY=relax",
                "BUILD_DATE=20260101",
                "GIT_SHORT_HASH=12345678",
                f"DO_PUSH={int(push)}",
                "DOCKER_BUILD_ARGS=--build-arg PATCH_VERSION=latest",
                *overrides,
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls

    return run


@pytest.mark.parametrize("target", ["train", "docker-train", "docker-dev"])
@pytest.mark.parametrize("push", [False, True])
def test_docker_makefile_pairs_variants(docker_make: Callable[..., MakeResult], target: str, push: bool) -> None:
    result, calls = docker_make(target, push=push)
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [call for call in calls if call[0] == "build"]
    tags = [call[call.index("-t") + 1] for call in builds]
    stages = ("train", "dev") if target == "docker-dev" else ("train",)
    assert tags == [image_tag(stage, variant) for variant in VARIANTS for stage in stages]
    for build, tag in zip(builds, tags):
        is_cu12 = tag.endswith("-cu12")
        assert build[build.index("-f") + 1] == ("docker/Dockerfile" if is_cu12 else "docker/Dockerfile.cu13")
        assert ("BASE_IMAGE=cu12-base" if is_cu12 else "BASE_IMAGE=cu13-base") in build
        assert "PATCH_VERSION=latest" in build
        if not is_cu12:
            assert ("GPU_ARCH=b300" if tag.endswith("-blackwell") else "GPU_ARCH=hopper") in build
        if ":dev-" in tag:
            assert build[build.index("--target") + 1] == "relax"
            assert f"TRAIN_IMAGE={tag.replace(':dev-', ':train-')}" in build
        else:
            assert build[build.index("--target") + 1] == "train"
    assert [call[1] for call in calls if call[0] == "push"] == (tags if push else [])
    assert all(call[0] == ("manifest" if push else "image") for call in calls if "inspect" in call)


@pytest.mark.parametrize("stage", ["train", "dev"])
@pytest.mark.parametrize("push", [False, True])
def test_docker_makefile_reuses_existing_images(
    docker_make: Callable[..., MakeResult], stage: str, push: bool
) -> None:
    existing = tuple(image_tag(stage, variant) for variant in VARIANTS)
    result, calls = docker_make("docker-dev", existing=existing, push=push)
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [call for call in calls if call[0] == "build"]
    assert [call[call.index("-t") + 1] for call in builds] == (
        [image_tag("dev", variant) for variant in VARIANTS] if stage == "train" else []
    )


@pytest.mark.parametrize("push", [False, True])
@pytest.mark.parametrize("custom_tag", [False, True])
def test_docker_makefile_continues_cu13_after_skipping_cu12(
    docker_make: Callable[..., MakeResult], push: bool, custom_tag: bool
) -> None:
    cu12_tag = "registry.example/relax:existing-cu12" if custom_tag else image_tag("train", "cu12")
    result, calls = docker_make(
        "docker-train",
        push=push,
        existing=(cu12_tag,),
        environment={"TRAIN_IMAGE": cu12_tag} if custom_tag else {},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call[-1] for call in calls if "inspect" in call] == [
        cu12_tag,
        image_tag("train", "cu13-blackwell"),
        image_tag("train", "cu13-hopper"),
    ]
    tags = [call[call.index("-t") + 1] for call in calls if call[0] == "build"]
    assert tags == [image_tag("train", variant) for variant in VARIANTS[1:]]
    assert [call[1] for call in calls if call[0] == "push"] == (tags if push else [])


def test_docker_makefile_keeps_cu12_overrides_out_of_cu13(docker_make: Callable[..., MakeResult]) -> None:
    result, calls = docker_make("docker-dev", "TRAIN_IMAGE=custom-train", "DEV_IMAGE=custom-dev")
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [call for call in calls if call[0] == "build"]
    assert len(builds) == 5
    assert "TRAIN_IMAGE=custom-train" in builds[0]
    assert builds[0][builds[0].index("-t") + 1] == "custom-dev"
    for build in builds[1:]:
        assert "TRAIN_IMAGE=custom-train" not in build
        assert "custom-dev" not in build
        assert "BASE_IMAGE=cu13-base" in build


def test_docker_makefile_reads_external_images_from_environment(docker_make: Callable[..., MakeResult]) -> None:
    result, calls = docker_make(
        "docker-dev",
        environment={
            "BASE_IMAGE": "external-cu12-base",
            "CU13_BASE_IMAGE": "external-cu13-base",
            "TRAIN_IMAGE": "external-cu12-train",
            "CU13_BLACKWELL_TRAIN_IMAGE": "external-blackwell-train",
            "CU13_HOPPER_TRAIN_IMAGE": "external-hopper-train",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [call for call in calls if call[0] == "build"]
    assert len(builds) == 3
    for build, variant, train in zip(
        builds, VARIANTS, ("external-cu12-train", "external-blackwell-train", "external-hopper-train")
    ):
        assert build[build.index("--target") + 1] == "relax"
        assert build[build.index("-t") + 1] == image_tag("dev", variant)
        assert f"TRAIN_IMAGE={train}" in build
        assert ("BASE_IMAGE=external-cu12-base" if variant == "cu12" else "BASE_IMAGE=external-cu13-base") in build


@pytest.mark.parametrize("variant", ["cu13-blackwell", "cu13-hopper"])
def test_docker_makefile_external_train_only_overrides_its_variant(
    docker_make: Callable[..., MakeResult], variant: str
) -> None:
    variable = "CU13_BLACKWELL_TRAIN_IMAGE" if variant.endswith("blackwell") else "CU13_HOPPER_TRAIN_IMAGE"
    result, calls = docker_make("docker-dev", environment={variable: "external-train"})
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [call for call in calls if call[0] == "build"]
    assert [call[call.index("-t") + 1] for call in builds] == [
        image_tag(stage, arch)
        for arch in VARIANTS
        for stage in ("train", "dev")
        if stage != "train" or arch != variant
    ]
    for build in builds:
        if build[build.index("-t") + 1] == image_tag("dev", variant):
            assert "TRAIN_IMAGE=external-train" in build
        else:
            assert "TRAIN_IMAGE=external-train" not in build


@pytest.mark.parametrize("stage", ["train", "dev"])
def test_docker_makefile_stops_after_build_failure(docker_make: Callable[..., MakeResult], stage: str) -> None:
    result, calls = docker_make("docker-dev", push=True, fail_tag=image_tag(stage, "cu12"))
    assert result.returncode != 0
    assert not any("cu13" in argument for call in calls for argument in call)
    assert [call[1] for call in calls if call[0] == "push"] == ([image_tag("train", "cu12")] if stage == "dev" else [])


@pytest.mark.parametrize("kind", ["train", "dev"])
@pytest.mark.parametrize("variant,arch", [("cu12", None), ("cu13-blackwell", "b300"), ("cu13-hopper", "hopper")])
def test_docker_makefile_builds_one_variant(
    docker_make: Callable[..., MakeResult], variant: str, arch: str | None, kind: str
) -> None:
    target = f"docker-{kind}-cu13-{arch}" if arch else f"docker-{kind}-cu12"
    result, calls = docker_make(target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call[call.index("-t") + 1] for call in calls if call[0] == "build"] == [
        image_tag(stage, variant) for stage in (("train", "dev") if kind == "dev" else ("train",))
    ]
