# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Offline plan, numerical and ownership gates; no Ray/network/GPU imports."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import owned_serve
from prepare_experiments import prepare
from run_experiments import numerical_gate, subset
from summarize_plan import summarize, variation


def plan_fixture():
    workload = {
        "model": {"id": "different-model", "path": "/models/second", "content_sha256": "a" * 64},
        "data": {
            "root": "/data",
            "metadata": "/data/train.json",
            "metadata_sha256": "b" * 64,
            "rows": 48,
            "format": "json_array",
            "selection_rule": "first 48 official records",
        },
        "source": {"repo": "/repo", "implementation_sha256": "c" * 64},
        "launcher": "scripts/training/diffusion/second.sh",
        "launcher_sha256": "d" * 64,
        "training": {
            "world_size": 8,
            "global_batch_size": 16,
            "micro_batch_size": 1,
            "seed": 42,
            "semantic_config": {"optimizer": {"lr": 0.001}, "param_dtype": "bf16"},
        },
        "cache": {"root": "/cache", "payload_bytes": 2**30, "prefetch_bytes": 2**20, "prefetch_depth": 0},
        "env": {},
        "extra_args": [],
    }
    return {
        "schema_version": 1,
        "status": "ready",
        "cluster": {
            "ray_address": "node:6379",
            "dashboard": "http://127.0.0.1:8265",
            "expected_gpus": 8,
            "pythonpath": ["/dependency"],
            "lock_file": "/shared/node.lock",
            "serve_deployments": {"sft": "TextSFT"},
        },
        "runs_root": "/runs/new",
        "workloads": {"second": workload},
        "runs": [
            {
                "name": f"run-{repeat}-{enabled}",
                "workload": "second",
                "purpose": "performance",
                "group": "fixed",
                "repeat": repeat,
                "cache": enabled,
                "steps": 12,
                "cold_steps": 3,
            }
            for repeat in range(1, 4)
            for enabled in (False, True)
        ],
    }


def proof_fixture():
    value = {
        "finite": True,
        "passed": True,
        "reference_l2": 1.0,
        "actual_l2": 1.0,
        "error_l2": 0.0,
        "max_absolute_error": 0.0,
    }
    return {
        "passed": True,
        "cache_branch_verified": True,
        "relative_tolerance": 0,
        "loss": {"finite": True, "passed": True, "absolute_error": 0},
        "gradients": {"some.adapter": dict(value)},
        "parameter_updates": {"some.adapter": dict(value)},
    }


class PlanTests(unittest.TestCase):
    def test_empty_cold_window_retains_unknown_throughput(self):
        self.assertEqual(variation([None, None])["mean"], None)

    def test_missing_final_raw_runs_cannot_pass_performance(self):
        repo = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location(
            "test_conditioning_public", repo / "scripts/tools/benchmark_condition_cache.py"
        )
        public = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(public)
        with tempfile.TemporaryDirectory() as folder:
            report = summarize(plan_fixture(), Path(folder), public)
        self.assertFalse(report["performance_gate"])
        self.assertFalse(report["acceptance_complete"])
        self.assertFalse(report["all_planned_runs_complete"])
        self.assertIsNone(report["groups"]["fixed"]["comparison"]["observed_at_least_five_percent"])

    def test_non_magicbrush_rows_accumulation_and_nonstandard_cold_window(self):
        result = prepare(plan_fixture())
        self.assertEqual(len(result["requests"]), 6)
        env = result["requests"][0]["env"]
        self.assertEqual((env["DATASET_SIZE"], env["GLOBAL_BATCH_SIZE"], env["MICRO_BATCH_SIZE"]), ("48", "16", "1"))
        self.assertEqual(env["RELAX_SKIP_CLUSTER_CLEANUP"], "1")
        self.assertFalse(result["execution_supported"])
        self.assertFalse(result["acceptance_passed"])

    def test_draft_is_not_launchable(self):
        plan = plan_fixture()
        plan["status"] = "draft"
        with self.assertRaises(ValueError):
            prepare(plan)

    def test_mixed_training_config_or_source_or_window_is_not_paired(self):
        for change in ("env", "window", "workload"):
            with self.subTest(change=change):
                plan = plan_fixture()
                row = plan["runs"][0]
                if change == "env":
                    row["env"] = {"GRADIENT_CHECKPOINTING": "0"}
                elif change == "window":
                    row["cold_steps"] = 4
                else:
                    plan["workloads"]["different"] = copy.deepcopy(plan["workloads"]["second"])
                    plan["workloads"]["different"]["source"]["implementation_sha256"] = "e" * 64
                    row["workload"] = "different"
                with self.assertRaises(ValueError):
                    prepare(plan)

    def test_missing_repeat_cannot_meet_final_gate(self):
        for count in (5, 4):
            plan = plan_fixture()
            plan["runs"] = plan["runs"][:count]
            with self.assertRaises(ValueError):
                prepare(plan)

    def test_protected_path_or_cache_override_rejected(self):
        for key in (
            "PYTHONPATH",
            "GPUS",
            "RELAX_CONDITION_CACHE",
            "RUNTIME_ENV_JSON",
            "RELAX_CONDITION_CACHE_RUN_DIR",
        ):
            plan = plan_fixture()
            plan["runs"][0]["env"] = {key: "untracked"}
            with self.assertRaises(ValueError):
                prepare(plan)

    def test_profile_not_counted_as_a_missing_performance_pair(self):
        plan = plan_fixture()
        row = plan["runs"][0] | {"name": "trace-only", "purpose": "profile"}
        plan["runs"].append(row)
        self.assertEqual(len(prepare(plan)["requests"]), 7)

    def test_config_subset_requires_observed_values_not_only_requested_env(self):
        subset({"config": {"height": 512}}, {"config": {"height": 512, "width": 512}}, "root")
        with self.assertRaises(ValueError):
            subset({"config": {"height": 512}}, {"config": {"height": 1024}}, "root")


class NumericalTests(unittest.TestCase):
    def test_arbitrary_parameter_count_allowed(self):
        self.assertEqual(numerical_gate(proof_fixture())["tensor_counts"], {"gradients": 1, "parameter_updates": 1})

    def test_zero_reference_requires_exact_zero(self):
        value = proof_fixture()
        value["gradients"]["some.adapter"].update(reference_l2=0, actual_l2=1e-20)
        with self.assertRaises(ValueError):
            numerical_gate(value)

    def test_nonfinite_and_nonexact_quantities_and_unexercised_branch_rejected(self):
        for change in ("nan", "error", "branch", "zero_update"):
            value = proof_fixture()
            if change == "nan":
                value["parameter_updates"]["some.adapter"]["actual_l2"] = float("nan")
            elif change == "error":
                value["gradients"]["some.adapter"]["error_l2"] = 1e-20
            elif change == "branch":
                value["cache_branch_verified"] = False
            else:
                value["parameter_updates"]["some.adapter"].update(reference_l2=0, actual_l2=0)
            with self.assertRaises(ValueError):
                numerical_gate(value)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.proc = Path(temp.name)
        (self.proc / "101").mkdir()
        (self.proc / "101/environ").write_bytes(b"RELAX_CONDITION_CACHE_RUN_DIR=/runs/owned\0")
        self.response = {
            "applications": {
                "sft": {
                    "name": "sft",
                    "last_deployed_time_s": 101.0,
                    "route_prefix": "/sft",
                    "source": "imperative",
                    "deployments": {"TextSFT": {"deployment_config": {"name": "TextSFT"}, "replicas": [{"pid": 101}]}},
                }
            }
        }
        self.owned = owned_serve.claim_owned_serve(
            self.response, 100, Path("/runs/owned"), self.proc, expected_deployments={"sft": "TextSFT"}
        )

    def test_no_assumed_model_or_service_class(self):
        self.assertEqual(list(self.owned), ["sft"])

    def test_foreign_live_replica_is_not_claimed(self):
        (self.proc / "101/environ").write_bytes(b"RELAX_CONDITION_CACHE_RUN_DIR=/other\0")
        with self.assertRaises(RuntimeError):
            owned_serve.claim_owned_serve(
                self.response, 100, Path("/runs/owned"), self.proc, expected_deployments={"sft": "TextSFT"}
            )

    def test_unrecorded_or_replaced_application_is_not_deleted(self):
        for change in ("new", "replace"):
            response = copy.deepcopy(self.response)
            if change == "new":
                response["applications"]["other"] = response["applications"]["sft"]
            else:
                response["applications"]["sft"]["last_deployed_time_s"] = 102
            delete = Mock()
            with self.assertRaises(RuntimeError):
                owned_serve.delete_owned_serve(lambda: response, delete, self.owned)
            delete.assert_not_called()

    def test_transient_missing_config_may_finish_deleting_without_any_delete(self):
        transient = copy.deepcopy(self.response)
        transient["applications"]["sft"]["deployments"]["TextSFT"].pop("deployment_config")
        snapshots = iter([transient, {"applications": {}}])
        delete = Mock()
        result = owned_serve.delete_owned_serve(lambda: next(snapshots), delete, self.owned, retry_interval=0.001)
        self.assertEqual(result, [])
        delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
