# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from pathlib import Path


def adapt_runners(test_dir: Path) -> None:
    anchor = "    result = runner.run(suite)\n"
    export = (
        "    if report_path := os.environ.get('R2E_RESULTS_JSON'):\n"
        "        import json\n"
        "        with open(report_path, 'w') as output:\n"
        "            json.dump({'.'.join(test.split('::')[1:]): outcome.upper() "
        "for test, outcome, _ in result.results if outcome != 'skipped'}, output)\n"
    )
    for name in ("unittest_custom_runner.py", "tornado_unittest_runner.py"):
        runner = test_dir / name
        if not runner.is_file():
            continue
        source = runner.read_text(encoding="utf-8")
        if source.count(anchor) != 1:
            raise RuntimeError(f"Unexpected unittest runner: {runner}")
        runner.write_text(source.replace(anchor, anchor + export, 1), encoding="utf-8")


if __name__ == "__main__":
    adapt_runners(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/r2e_tests"))
