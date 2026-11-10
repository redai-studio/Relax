# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import copy
import json
import unittest

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import (
    collective_plan_hashes,
    fingerprint,
    partition_expert_plans,
    validate_collective_plan,
)


def fixture(world=8, replicas=4, tps=2, source_world=None, ranks=None):
    experts, owned = world * 3, 3
    ranks = list(range(world)) if ranks is None else ranks
    order = list(reversed(range(world)))  # Owner rank is deliberately not expert-range order.
    layouts = {
        name: dict(
            shape=[experts, rows, cols],
            stride=[rows * cols, cols, 1],
            dtype="torch.bfloat16",
            nbytes=experts * rows * cols * 2,
        )
        for name, rows, cols in [("gate_up", 4, 3), ("down", 3, 2)]
    }
    groups = [[f"replica-{engine}:tp-{tp}" for engine in range(replicas)] for tp in range(tps)]
    plans, source_blocks, native = {}, {}, {}
    for tp, group in enumerate(groups):
        for name, layout in layouts.items():
            native[tp, name] = bytes((index * 13 + tp * 17 + 3) % 251 for index in range(layout["nbytes"]))
        for key in group:
            writes = []
            for index, owner in enumerate(ranks):
                begin = order[index] * owned
                for name, layout in layouts.items():
                    expert_bytes = layout["nbytes"] // experts
                    for offset, count in ((0, 2), (2, 1)):
                        start = begin + offset
                        source = f"{name}:{start}:{count}"
                        source_blocks[tp, owner, source] = native[tp, name][
                            start * expert_bytes : (start + count) * expert_bytes
                        ]
                        writes.append(
                            dict(
                                id=f"{owner}:{source}:{tp}",
                                source=source,
                                source_rank=owner,
                                target=key,
                                param=name,
                                dst_offset=start * expert_bytes,
                                rows=1,
                                row_bytes=count * expert_bytes,
                                dst_stride=count * expert_bytes,
                                tp_rank=tp,
                            )
                        )
            plans[key] = dict(
                source_world_size=source_world or world,
                receipt_delivery="commit-rpc-v1",
                writes=writes,
            )
    return plans, layouts, groups, source_blocks, native


def pipeline_fixture(stages=2, owners=4, replicas=4, rank_sets=None):
    plans, layouts, blocks, native = {}, {}, {}, {}
    # Global source slots are deliberately interleaved between stages.
    rank_sets = rank_sets or [list(range(stage, stages * owners, stages)) for stage in range(stages)]
    source_world = max(rank for ranks in rank_sets for rank in ranks) + 1
    for stage in range(stages):
        local, specs, groups, values, expected = fixture(owners, replicas, 2, source_world, rank_sets[stage])
        prefix = f"stage-{stage}."
        layouts.update({prefix + name: layout for name, layout in specs.items()})
        for key, plan in local.items():
            for write in plan["writes"]:
                write["id"] = prefix + write["id"]
                write["param"] = prefix + write["param"]
                write["source"] = prefix + write["source"]
            if key in plans:
                plans[key]["writes"].extend(plan["writes"])
            else:
                plans[key] = plan
        blocks.update(
            {
                (tp, owner, prefix + source): bytes(v ^ stage for v in value)
                for (tp, owner, source), value in values.items()
            }
        )
        native.update({(tp, prefix + name): bytes(v ^ stage for v in value) for (tp, name), value in expected.items()})
    return plans, layouts, groups, blocks, native


class TestCollectiveExpertPlan(unittest.TestCase):
    def test_owner_block_crossing_seed_boundary_is_rejected(self):
        plans, layouts, groups, _, _ = fixture(world=2, replicas=6, tps=1)
        # The expert axis divides all six seeds, but the original two-expert
        # blocks were not planned at these one-expert seed boundaries.
        with self.assertRaisesRegex(ValueError, "block crosses a collective seed boundary"):
            partition_expert_plans(plans, layouts, groups)

    def test_disjoint_pipeline_parameter_owners_reconstruct_each_fresh_version(self):
        for stages, owners, replicas, rank_sets in (
            (2, 4, 4, None),
            (2, 4, 2, None),
            (4, 2, 2, None),
            (2, 4, 4, [[0, 1, 2, 3], [0, 1, 4, 5]]),
        ):
            with self.subTest(stages=stages, owners=owners, replicas=replicas):
                plans, layouts, groups, blocks, native = pipeline_fixture(stages, owners, replicas, rank_sets)
                participating = {w["source_rank"] for p in plans.values() for w in p["writes"]}
                before = copy.deepcopy(plans)
                result = partition_expert_plans(plans, layouts, groups)
                self.assertEqual(plans, before)
                self.assertEqual(
                    sum(w["row_bytes"] for p in result.values() for w in p["writes"]) * replicas,
                    sum(w["row_bytes"] for p in plans.values() for w in p["writes"]),
                )
                for version in range(3):
                    for tp, keys in enumerate(groups):
                        outputs = {name: bytearray([255] * spec["nbytes"]) for name, spec in layouts.items()}
                        for rank, key in enumerate(keys):
                            plan = json.loads(json.dumps(result[key]))
                            self.assertEqual(validate_collective_plan(plan, layouts, key, tp), set(layouts))
                            self.assertEqual(plan["source_world_size"], max(participating) + 1)
                            self.assertEqual(len(plan["collective"]["owners"]), len(participating))
                            self.assertEqual(
                                collective_plan_hashes(plan), {peer: fingerprint(result[peer]) for peer in keys}
                            )
                            for write in plan["writes"]:
                                value = bytes(
                                    v ^ (version * 32) for v in blocks[tp, write["source_rank"], write["source"]]
                                )
                                width = layouts[write["param"]]["nbytes"] // replicas
                                start = write["dst_offset"]
                                self.assertGreaterEqual(start, rank * width)
                                self.assertLessEqual(start + len(value), (rank + 1) * width)
                                outputs[write["param"]][start : start + len(value)] = value
                        for name, output in outputs.items():
                            self.assertEqual(bytes(output), bytes(v ^ (version * 32) for v in native[tp, name]))

    def test_pipeline_owner_layout_holes_and_changed_intervals_are_rejected(self):
        for fault in ("missing_owner_layer", "mixed_interval", "misaligned_owner", "wrong_stage_chunk"):
            with self.subTest(fault=fault):
                plans, layouts, groups, _, _ = pipeline_fixture()
                if fault == "missing_owner_layer":
                    for plan in plans.values():
                        plan["writes"] = [
                            w for w in plan["writes"] if not (w["param"] == "stage-1.down" and w["source_rank"] == 1)
                        ]
                elif fault in ("mixed_interval", "misaligned_owner"):
                    for plan in plans.values():
                        for write in plan["writes"]:
                            if write["param"] == "stage-1.down" and write["source_rank"] in (1, 3):
                                if fault == "mixed_interval" or write["source"].endswith(":1"):
                                    write["source_rank"] = 4 - write["source_rank"]
                else:
                    result = partition_expert_plans(plans, layouts, groups)
                    key = groups[0][0]
                    plan = copy.deepcopy(result[key])
                    shards = plan["collective"]["shards"]
                    first, second = groups[0][:2]
                    left = next(w for w in shards[first] if w["param"] == "stage-1.down")
                    right = next(w for w in shards[second] if w["param"] == "stage-1.down")
                    shards[first].remove(left)
                    shards[second].remove(right)
                    left["target"], right["target"] = second, first
                    shards[first].append(right)
                    shards[second].append(left)
                    with self.assertRaises(ValueError):
                        validate_collective_plan(plan, layouts, key, 0)
                    continue
                with self.assertRaises(ValueError):
                    partition_expert_plans(plans, layouts, groups)

    def test_partitioned_writes_and_allgather_reconstruct_native_bytes(self):
        for world, replicas, tps, source_world, ranks in [
            (2, 2, 1, None, None),
            (4, 2, 1, None, None),
            (4, 4, 2, None, None),
            (8, 2, 4, None, None),
            (8, 4, 2, None, None),
            (8, 8, 1, None, None),
            (16, 2, 8, None, None),
            (16, 8, 2, None, None),
            (4, 4, 2, 8, [0, 2, 4, 6]),
            (4, 2, 2, 8, [7, 1, 5, 3]),
            (4, 4, 2, 5, [4, 0, 3, 1]),
            (2, 2, 1, 16, [13, 2]),
        ]:
            with self.subTest(world=world, replicas=replicas, tps=tps, source_world=source_world, ranks=ranks):
                plans, layouts, groups, blocks, native = fixture(world, replicas, tps, source_world, ranks)
                before = copy.deepcopy(plans)
                result = partition_expert_plans(plans, layouts, groups)
                self.assertEqual(plans, before)
                original_bytes = sum(w["row_bytes"] for plan in plans.values() for w in plan["writes"])
                sent_bytes = sum(w["row_bytes"] for plan in result.values() for w in plan["writes"])
                self.assertEqual(sent_bytes * replicas, original_bytes)
                for tp, keys in enumerate(groups):
                    outputs = {
                        key: {name: bytearray([255] * spec["nbytes"]) for name, spec in layouts.items()}
                        for key in keys
                    }
                    for rank, key in enumerate(keys):
                        plan = json.loads(json.dumps(result[key]))
                        self.assertEqual(validate_collective_plan(plan, layouts, key, tp), set(layouts))
                        self.assertEqual(len({w["source_rank"] for w in plan["writes"]}), world // replicas)
                        for write in plan["writes"]:
                            value = blocks[tp, write["source_rank"], write["source"]]
                            start = write["dst_offset"]
                            self.assertEqual(len(value), write["row_bytes"])
                            outputs[key][write["param"]][start : start + len(value)] = value
                        self.assertEqual(
                            collective_plan_hashes(plan), {peer: fingerprint(result[peer]) for peer in keys}
                        )
                    for name, layout in layouts.items():
                        width = layout["nbytes"] // replicas
                        assembled = b"".join(
                            bytes(outputs[key][name][rank * width : (rank + 1) * width])
                            for rank, key in enumerate(keys)
                        )
                        self.assertEqual(assembled, native[tp, name])

    def test_whole_expert_alignment_and_owner_consistency_are_required(self):
        for failure in [
            "owner_gap",
            "parameter_ownership",
            "fractional_expert",
            "stride",
            "different_counts",
            "missing_write",
            "duplicate_id",
            "wrong_target",
            "tp",
        ]:
            with self.subTest(failure=failure):
                plans, layouts, groups, _, _ = fixture()
                key = groups[0][0]
                if failure == "owner_gap":
                    # Preserve full byte coverage but split one owner's range.
                    for plan in plans.values():
                        left = next(w for w in plan["writes"] if w["source_rank"] == 0)
                        right = next(w for w in plan["writes"] if w["source_rank"] == 1)
                        left["source_rank"], right["source_rank"] = 1, 0
                elif failure == "parameter_ownership":
                    for plan in plans.values():
                        for write in plan["writes"]:
                            if write["param"] == "down" and write["source_rank"] in (0, 1):
                                write["source_rank"] = 1 - write["source_rank"]
                elif failure == "fractional_expert":
                    plans[key]["writes"][0]["dst_offset"] += 2
                elif failure == "stride":
                    layouts["down"]["stride"][0] += 1
                elif failure == "different_counts":
                    layouts["down"]["shape"][0] -= 1
                elif failure == "missing_write":
                    plans[key]["writes"].pop()
                elif failure == "duplicate_id":
                    plans[key]["writes"][0]["id"] = plans[key]["writes"][1]["id"]
                elif failure == "wrong_target":
                    plans[key]["writes"][0]["target"] = groups[0][1]
                elif failure == "tp":
                    plans[key]["writes"][0]["tp_rank"] = 1
                with self.assertRaises(ValueError):
                    partition_expert_plans(plans, layouts, groups)

    def test_disjoint_expert_replicas_preserve_distinct_source_bytes(self):
        plans, groups, blocks, expected = {}, [], {}, {}
        for replica, owners in (("a", [6, 0, 4, 2]), ("b", [3, 7, 1, 5])):
            local, layouts, local_groups, local_blocks, native = fixture(4, 2, 2, 8, owners)
            for key, plan in local.items():
                key = replica + key
                for write in plan["writes"]:
                    write["target"] = key
                plans[key] = plan
            for tp, keys in enumerate(local_groups):
                keys = [replica + key for key in keys]
                groups.append(keys)
                for name in layouts:
                    value = native[tp, name]
                    expected[keys[0], name] = bytes(v ^ 127 for v in value) if replica == "b" else value
            blocks.update(
                {
                    key: bytes(v ^ 127 for v in value) if replica == "b" else value
                    for key, value in local_blocks.items()
                }
            )
        result = partition_expert_plans(plans, layouts, groups)
        for keys in groups:
            outputs = {name: bytearray(layout["nbytes"]) for name, layout in layouts.items()}
            for key in keys:
                plan = result[key]
                tp = plan["collective"]["tp_rank"]
                self.assertEqual(validate_collective_plan(plan, layouts, key, tp), set(layouts))
                for write in plan["writes"]:
                    value = blocks[tp, write["source_rank"], write["source"]]
                    start = write["dst_offset"]
                    outputs[write["param"]][start : start + len(value)] = value
            for name, value in outputs.items():
                self.assertEqual(bytes(value), expected[keys[0], name])
        self.assertNotEqual(expected[groups[0][0], "down"], expected[groups[2][0], "down"])
        crossed = copy.deepcopy(groups)
        crossed[0][1], crossed[2][1] = crossed[2][1], crossed[0][1]
        with self.assertRaises(ValueError):
            partition_expert_plans(plans, layouts, crossed)

    def test_groups_must_cover_exact_targets_and_even_owner_partitions(self):
        for failure in ["missing", "duplicate", "single", "uneven", "mode"]:
            with self.subTest(failure=failure):
                plans, layouts, groups, _, _ = fixture(replicas=3 if failure == "uneven" else 4)
                if failure == "missing":
                    groups[0].pop()
                elif failure == "duplicate":
                    groups[0][-1] = groups[0][0]
                elif failure == "single":
                    groups = [[key] for key in plans]
                elif failure == "mode":
                    plans[groups[0][-1]]["receipt_delivery"] = "other"
                with self.assertRaises(ValueError):
                    partition_expert_plans(plans, layouts, groups)

    def test_receiver_rejects_changed_schema_local_chunk_or_owner_evidence(self):
        plans, layouts, groups, _, _ = fixture()
        result = partition_expert_plans(plans, layouts, groups)
        key = groups[0][0]
        for failure in [
            "rank",
            "range",
            "keys",
            "peer_target",
            "owner",
            "hole",
            "layout",
            "own_writes",
            "params",
            "tp",
        ]:
            with self.subTest(failure=failure):
                plan = json.loads(json.dumps(result[key]))
                schema = plan["collective"]
                if failure == "rank":
                    plan["collective_rank"] = 1
                elif failure == "range":
                    plan["expert_range"] = [6, 12]
                elif failure == "keys":
                    schema["keys"] = list(reversed(schema["keys"]))
                elif failure == "peer_target":
                    schema["shards"][groups[0][-1]][0]["target"] = key
                elif failure == "owner":
                    schema["owners"][0]["begin"] -= 3
                elif failure == "hole":
                    schema["shards"][groups[0][-1]].pop()
                elif failure == "layout":
                    schema["layouts"]["down"]["stride"][0] += 1
                elif failure == "own_writes":
                    plan["writes"].pop()
                elif failure == "params":
                    plan["params"].pop()
                elif failure == "tp":
                    schema["tp_rank"] = 1
                with self.assertRaises(ValueError):
                    validate_collective_plan(plan, layouts, key, 0)

    def test_equivalent_replica_write_orders_have_identical_generated_plans(self):
        plans, layouts, groups, _, _ = fixture()
        expected = partition_expert_plans(plans, layouts, groups)
        for index, plan in enumerate(plans.values()):
            writes = list(reversed(plan["writes"]))
            plan["writes"] = writes[index:] + writes[:index]
        self.assertEqual(partition_expert_plans(plans, layouts, groups), expected)

    def test_published_metadata_does_not_alias_source_plans_or_layouts(self):
        plans, layouts, groups, _, _ = fixture()
        result = partition_expert_plans(plans, layouts, groups)
        hashes = {key: fingerprint(plan) for key, plan in result.items()}
        for plan in plans.values():
            for write in plan["writes"]:
                write["dst_offset"] = -1
        layouts["down"]["shape"][0] = 1
        self.assertEqual({key: fingerprint(plan) for key, plan in result.items()}, hashes)

    def test_peer_hashes_bind_rpc_mode(self):
        plans, layouts, groups, _, _ = fixture()
        result = partition_expert_plans(plans, layouts, groups)
        key = groups[0][0]
        original = collective_plan_hashes(result[key])
        changed = dict(result[key], receipt_delivery="file")
        self.assertTrue(all(sha != original[peer] for peer, sha in collective_plan_hashes(changed).items()))
