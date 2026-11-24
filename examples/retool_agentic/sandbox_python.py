# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Run payload Python with single-thread HiGHS in the pinned sandbox image."""

import importlib.machinery
import runpy
import sys
from types import ModuleType
from typing import Any


class SingleThreadHiGHS:
    def find_spec(
        self, fullname: str, path: list[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != "scipy.optimize._highs._highs_wrapper":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        original = spec.loader

        class Loader:
            def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
                return original.create_module(spec)

            def exec_module(self, module: ModuleType) -> None:
                original.exec_module(module)
                native = module._highs_wrapper

                def single_thread(*args: Any) -> Any:
                    # SciPy 1.11.4 passes the options dictionary last. HiGHS
                    # otherwise starts native threads despite OMP_NUM_THREADS.
                    return native(*args[:-1], dict(args[-1], threads=1))

                module._highs_wrapper = single_thread

        spec.loader = Loader()
        return spec


if __name__ == "__main__":
    sys.meta_path.insert(0, SingleThreadHiGHS())
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
