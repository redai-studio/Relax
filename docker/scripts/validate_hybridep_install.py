# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate the private training wheel without changing SGLang's DeepEP
files."""

import importlib.metadata as md
from importlib.machinery import PathFinder
from pathlib import Path


def validate(hybrid, distributions) -> None:
    if not hybrid.version.startswith("1.2.1+"):
        raise ValueError(f"Unexpected HybridEP distribution: {hybrid.version}")
    files = hybrid.files or ()
    owned = {Path(hybrid.locate_file(path)).resolve() for path in files}
    if any(str(path).startswith(("deep_ep/", "deep_ep_cpp", "hybrid_ep_cpp")) for path in files):
        raise ValueError("HybridEP must not install into a public DeepEP namespace")
    package = Path(hybrid.locate_file("_relax_hybrid_ep"))
    for name in ("__init__.py", "utils.py", "hybrid_ep_buffer.py"):
        path = package / name
        if not path.is_file() or path.resolve() not in owned:
            raise ValueError(f"HybridEP wheel does not own {path}")
    for name in ("utils", "deep_ep_cpp", "hybrid_ep_cpp"):
        spec = PathFinder.find_spec(f"_relax_hybrid_ep.{name}", [str(package)])
        if spec is None or not spec.origin or Path(spec.origin).resolve() not in owned:
            raise ValueError(f"HybridEP module is missing or shadowed: {name}")
        if name == "utils" and Path(spec.origin).resolve() != (package / "utils.py").resolve():
            raise ValueError("HybridEP utils.py is shadowed")
    for distribution in distributions:
        if distribution.metadata["Name"] == hybrid.metadata["Name"]:
            continue
        other = {Path(distribution.locate_file(path)).resolve() for path in distribution.files or ()}
        if owned & other:
            raise ValueError(f"HybridEP files overlap {distribution.metadata['Name']}")


def main() -> None:
    validate(md.distribution("relax-hybrid-ep"), md.distributions())


if __name__ == "__main__":
    main()
