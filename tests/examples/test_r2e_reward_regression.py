# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""JUnit XML scoring regressions without importing Agent Server runtime
deps."""

import ast
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "examples/mini_swe_agent/agent_server.py"


def load_scoring_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"_parse_junit_xml", "_r2e_reward"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"ET": ET, "Path": Path, "json": json, "re": re, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_parse_junit_xml"], namespace["_r2e_reward"]


parse_xml, reward_fn = load_scoring_functions()


def score(expected, xml):
    return reward_fn({"expected_output_json": json.dumps(expected)}, xml)


def _case(name, status="PASSED", classname="r2e_tests.test_calc", file="r2e_tests/test_calc.py", message=""):
    case = ET.Element("testcase", {"name": name, "classname": classname, "file": file})
    if status != "PASSED":
        ET.SubElement(case, {"FAILED": "failure", "ERROR": "error", "SKIPPED": "skipped"}[status], message=message)
    return case


def report(*cases):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    suite.extend(cases)
    return ET.tostring(root, encoding="unicode")


CASES = [
    ("passed", {"test_ok": "PASSED"}, report(_case("test_ok")), 1.0),
    (
        "failed_mentions_passed",
        {"test_ok": "PASSED"},
        report(_case("test_ok", "FAILED", message="assert 'PASSED' == 'FAILED'")),
        0.0,
    ),
    ("expected_failed", {"test_ok": "FAILED"}, report(_case("test_ok", "FAILED")), 1.0),
    ("expected_error", {"test_ok": "ERROR"}, report(_case("test_ok", "ERROR", message="PASSED")), 1.0),
    ("error_mismatch", {"test_ok": "PASSED"}, report(_case("test_ok", "ERROR")), 0.0),
    (
        "class_method",
        {"TestMath.test_add": "PASSED"},
        report(_case("test_add", classname="r2e_tests.test_calc.TestMath")),
        1.0,
    ),
    ("parametrized", {"test_add[1]": "PASSED"}, report(_case("test_add[1]")), 1.0),
    ("skipped_ignored", {"test_ok": "PASSED"}, report(_case("test_skip", "SKIPPED"), _case("test_ok")), 1.0),
    (
        "xfail_ignored",
        {"test_ok": "PASSED"},
        report(_case("test_xfail", "SKIPPED", message="xfail"), _case("test_ok")),
        1.0,
    ),
    ("empty", {}, report(), 0.0),
    ("missing_report", {"test_ok": "PASSED"}, "", 0.0),
    ("malformed_xml", {"test_ok": "PASSED"}, "<testsuites>", 0.0),
    ("wrong_root", {"test_ok": "PASSED"}, "<log><testcase /></log>", 0.0),
    ("missing_file", {"test_ok": "PASSED"}, report(_case("test_ok", file="")), 0.0),
    ("missing_name", {"test_ok": "PASSED"}, report(_case("", "PASSED")), 0.0),
    ("bad_classname", {"test_ok": "PASSED"}, report(_case("test_ok", classname="Unrelated")), 0.0),
    ("duplicate_id", {"test_ok": "PASSED"}, report(_case("test_ok"), _case("test_ok", "FAILED")), 0.0),
    ("missing_expected", {"test_ok": "PASSED", "test_other": "PASSED"}, report(_case("test_ok")), 0.0),
    ("expected_suffix_and_ansi", {"\x1b[1mtest_ok\x1b[0m - detail": "PASSED"}, report(_case("test_ok")), 1.0),
]


@pytest.mark.parametrize("name, expected, xml, wanted", CASES, ids=[case[0] for case in CASES])
def test_xml_reward_regression(name, expected, xml, wanted):
    assert score(expected, xml) == wanted


def test_real_pytest_junit_xml(tmp_path):
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n\n"
        "def test_ok():\n    assert True\n\n"
        "def test_fail():\n    assert 'PASSED' == 'FAILED'\n\n"
        "class TestMath:\n    def test_add(self):\n        assert 1 + 1 == 2\n\n"
        "@pytest.mark.parametrize('value', [1])\n"
        "def test_param(value):\n    assert value == 1\n\n"
        "@pytest.mark.skip(reason='not scored')\n"
        "def test_skip():\n    assert False\n",
        encoding="utf-8",
    )
    script = tmp_path / "run_tests.sh"
    script.write_text(
        f"#!/bin/bash\n{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider {shlex.quote(str(sample))}\n",
        encoding="utf-8",
    )
    xml_path = tmp_path / "results.xml"
    env = os.environ.copy()
    env["PYTEST_ADDOPTS"] = f"--junitxml={xml_path} -o junit_family=xunit1"
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert xml_path.is_file()
    xml = xml_path.read_text(encoding="utf-8")
    expected = {
        "test_ok": "PASSED",
        "test_fail": "FAILED",
        "TestMath.test_add": "PASSED",
        "test_param[1]": "PASSED",
    }
    assert parse_xml(xml) == expected
    assert score(expected, xml) == 1.0
    assert score({**expected, "test_fail": "PASSED"}, xml) == 0.0
