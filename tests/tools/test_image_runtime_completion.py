# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise image repairs without modifying the installed runtime."""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "docker/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybridep_validation_preserves_sglang_package(tmp_path):
    import importlib.metadata as md

    validator = load_script("validate_hybridep_install")
    private = tmp_path / "_relax_hybrid_ep"
    public = tmp_path / "deep_ep"
    files = ["__init__.py", "utils.py", "hybrid_ep_buffer.py", "deep_ep_cpp.so", "hybrid_ep_cpp.so"]
    private.mkdir()
    (public / "utils").mkdir(parents=True)
    public_files = [public / "__init__.py", public / "utils/__init__.py"]
    for file in public_files:
        file.write_text("SGLANG = True\n")
    for name in files:
        (private / name).write_text("")
    for name, version, paths in [
        ("relax_hybrid_ep", "1.2.1+eb9cee7", ["_relax_hybrid_ep/" + name for name in files]),
        ("sgl_deep_ep", "0.1.2", [str(file.relative_to(tmp_path)) for file in public_files]),
    ]:
        info = tmp_path / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        (info / "RECORD").write_text("".join(f"{path},,\n" for path in paths))
    distributions = list(md.distributions(path=[str(tmp_path)]))
    hybrid = next(d for d in distributions if d.metadata["Name"] == "relax_hybrid_ep")
    validator.validate(hybrid, distributions)
    assert all(file.read_text() == "SGLANG = True\n" for file in public_files)
    (private / "utils").mkdir()
    (private / "utils/__init__.py").write_text("")
    with pytest.raises(ValueError, match="shadowed"):
        validator.validate(hybrid, distributions)


def test_legacy_fla_rebuilds_complete_validated_closure(tmp_path):
    installer = load_script("install_legacy_fla")
    directory = os.environ.get("BLACKWELL_FLA042_WHEEL_DIR")
    if not directory:
        pytest.skip("Set BLACKWELL_FLA042_WHEEL_DIR to the two checksum-pinned 0.4.2 wheels")
    wheels = [Path(directory) / url.rsplit("/", 1)[1] for url, _ in installer.WHEELS]
    target = tmp_path / "compat"
    report = installer.install(target, ROOT / "docker/patch/fla/0.4.2-legacy-runtime.patch", wheels)
    assert report["source_sha256"] == installer.SOURCE_SHA256
    assert len(list((target / "fla").rglob("*.py"))) == 352
    assert report["gpu_validated"] is False
    with pytest.raises(ValueError, match="overwrite"):
        installer.install(target, ROOT / "docker/patch/fla/0.4.2-legacy-runtime.patch", wheels)
    damaged = tmp_path / "wrong.whl"
    damaged.write_bytes(wheels[0].read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        installer.unpack(damaged, installer.WHEELS[0][1], tmp_path / "bad")


def test_final_dependency_check_rejects_ray_numpy_conflict():
    validate = load_script("record_qwen35_runtime").validate_pip_check
    allowed = "torch 2.13.0 has requirement nvidia-nccl-cu13==2.29.7, but you have nvidia-nccl-cu13 2.30.7.\n"
    validate(subprocess.CompletedProcess([], 1, allowed, ""))
    conflict = "ray 2.56.1.2 has requirement numpy<2.0, but you have numpy 2.2.6.\n"
    with pytest.raises(RuntimeError, match="ray"):
        validate(subprocess.CompletedProcess([], 1, allowed + conflict, ""))


@pytest.mark.parametrize("backend", ["v2", "hybrid"])
def test_deepep_namespace_patch_isolates_real_event_type(tmp_path, backend):
    hybrid = backend == "hybrid"
    source = os.environ.get("BLACKWELL_HYBRIDEP_SOURCE_DIR" if hybrid else "BLACKWELL_DEEPEP_V2_SOURCE_DIR")
    if not source:
        pytest.skip("Set BLACKWELL_DEEPEP_V2_SOURCE_DIR to pinned unpatched v2 sources")
    import sysconfig

    import torch

    pybind_include = Path(torch.__file__).parent / "include"

    pristine = Path(source)
    patched = tmp_path / "patched"
    shutil.copytree(pristine, patched)
    result = subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--fuzz=0",
            "-p1",
            "-i",
            str(
                ROOT
                / "docker/patch/deepep"
                / ("eb9cee7-hybridep-namespace.patch" if hybrid else "2.1.0-relax-namespace.patch")
            ),
        ],
        cwd=patched,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "offset" not in result.stdout and "fuzz" not in result.stdout
    header_path = "csrc/event.hpp" if hybrid else "csrc/utils/event.hpp"
    event = (patched / header_path).read_text()
    expected_type = "RelaxHybridEventHandle" if hybrid else "RelaxV2EventHandle"
    assert f"struct {expected_type}" in event and f"using EventHandle = {expected_type}" in event
    # Compile the real patched EventHandle header with CPU stubs for CUDA/Torch.
    # This verifies RTTI/pybind isolation, not CUDA event behavior or kernel ABI.
    includes = tmp_path / "includes"
    cuda = includes / "ATen/cuda/CUDAContext.h"
    cuda.parent.mkdir(parents=True)
    cuda.write_text("""#pragma once
#include <vector>
#include <optional>
namespace torch {
struct Tensor {};
constexpr int kCUDA = 0;
struct Event { explicit Event(int) {} template <typename T> void record(const T&) {} };
}
namespace at::cuda {
struct CUDAStream {
 int id() const { return 0; }
 const CUDAStream& unwrap() const { return *this; }
 void wait(const torch::Event&) const {}
};
inline CUDAStream getCurrentCUDAStream() { return {}; }
}
""")
    exception = includes / ("kernels/exception.cuh" if hybrid else "deep_ep/common/exception.cuh")
    exception.parent.mkdir(parents=True)
    exception.write_text("#pragma once\n#define EP_HOST_ASSERT(x) ((void)0)\n")
    for name, header, local in (
        ("legacy_event_fixture", pristine / header_path, ""),
        ("modern_event_fixture", patched / header_path, ", pybind11::module_local()"),
    ):
        # Keep the real header bytes, but resolve its CUDA includes against CPU stubs.
        copied_header = tmp_path / f"{name}.hpp"
        shutil.copyfile(header, copied_header)
        header = copied_header
        cpp = tmp_path / f"{name}.cpp"
        cpp.write_text(f'''#include <pybind11/pybind11.h>
#include "{header}"
PYBIND11_MODULE({name}, m) {{
 pybind11::class_<deep_ep::EventHandle>(m, "EventHandle"{local}).def(pybind11::init<>());
 m.def("accept", [](const deep_ep::EventHandle&) {{ return true; }});
}}
''')
        subprocess.run(
            [
                "g++",
                "-O0",
                "-shared",
                "-std=c++17",
                "-fPIC",
                "-fvisibility=hidden",
                *(["-Ddeep_ep=relax_hybrid_deep_ep"] if hybrid and local else []),
                str(cpp),
                "-I" + str(includes),
                "-I" + str(pybind_include),
                "-I" + sysconfig.get_path("include"),
                "-o",
                str(tmp_path / (name + sysconfig.get_config_var("EXT_SUFFIX"))),
            ],
            check=True,
            capture_output=True,
        )
    code = """import importlib, sys
for name in sys.argv[1:]: importlib.import_module(name)
import legacy_event_fixture as a, modern_event_fixture as b
assert a.EventHandle is not b.EventHandle
for own, other in ((a,b), (b,a)):
 assert own.accept(own.EventHandle())
 try: own.accept(other.EventHandle())
 except TypeError: pass
 else: raise AssertionError('foreign EventHandle accepted')
"""
    for order in (("legacy_event_fixture", "modern_event_fixture"), ("modern_event_fixture", "legacy_event_fixture")):
        subprocess.run([sys.executable, "-c", code, *order], cwd=tmp_path, check=True, capture_output=True)


def test_final_check_rejects_platform_ray_override(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "docker/scripts"))
    validator = load_script("validate_final_runtime")
    monkeypatch.setattr(sys, "argv", ["validate_final_runtime.py"])
    monkeypatch.setattr(validator.md, "version", lambda package: "2.56.1.2")
    with pytest.raises(ValueError, match="ray: expected 2.56.1"):
        validator.main()


def test_published_constraints_allow_vendor_ray_without_relaxing_other_pins(tmp_path):
    source = (ROOT / "docker/Dockerfile.cu13").read_text()
    export = next(line.strip() for line in source.splitlines() if "sed 's/^ray==" in line)
    assert source.index(export) > source.rindex("python /opt/relax/validate_final_runtime.py")
    assert source.rstrip().endswith("ENV PIP_CONSTRAINT=/etc/relax-cu130-runtime-constraints.txt")
    build = tmp_path / "build.txt"
    build.write_text("ray==2.56.1\nnumpy==2.2.6\ntorch==2.13.0\nnvidia-nccl-cu13==2.29.7\n")
    runtime = tmp_path / "runtime.txt"
    # Execute the Dockerfile's actual sed program, preserving every other pin.
    with runtime.open("w") as output:
        subprocess.run(["sed", shlex.split(export)[1], str(build)], stdout=output, check=True)
    assert runtime.read_text().splitlines()[1:] == build.read_text().splitlines()[1:]
    assert build.read_text().startswith("ray==2.56.1\n")

    # Local packaging fixture: verify pip constraint resolution, not Ray ABI.
    wheel = tmp_path / "ray-2.56.1.2-py3-none-any.whl"
    info = "ray-2.56.1.2.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{info}/METADATA",
            "Metadata-Version: 2.1\nName: ray\nVersion: 2.56.1.2\nRequires-Dist: numpy<2.0\n",
        )
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--no-index",
        "--no-deps",
        "--find-links",
        str(tmp_path),
        "ray==2.56.1.2",
    ]
    for constraints, expected_success in ((build, False), (runtime, True)):
        result = subprocess.run(
            command,
            env={**os.environ, "PIP_CONSTRAINT": str(constraints)},
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == expected_success, result.stdout + result.stderr


def test_default_dockerfile_installs_shared_repairs_without_cu13_libraries():
    source = (ROOT / "docker/Dockerfile").read_text()
    assert "install_legacy_fla.py --target /opt/relax/compat/fla-0.4.2" in source
    assert "--from=hybridep_builder /opt/ep-wheels" in source
    assert "python /opt/relax/validate_hybridep_install.py" in source
    assert "validate_final_runtime.py --profile cu12" in source
    assert "nvidia-nvshmem-cu12==3.4.5" in source
    assert "nvidia-nvshmem-cu13" not in source
    assert "nvidia-nccl-cu13" not in source
    assert "v0.5.17-cu129" in source
