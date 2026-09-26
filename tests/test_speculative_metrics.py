from relax.agentic.session.state import SessionForest, check_messages
from relax.utils.metrics.speculative import aggregate_speculative_metrics
from relax.utils.types import Sample


def _sample(*records, accepted=0, proposed=0, verify=0, completion=0):
    sample = Sample(metadata={})
    if records:
        sample.metadata["spec_generation_nodes"] = list(records)
    sample.spec_info.spec_accept_token_num = accepted
    sample.spec_info.spec_draft_token_num = proposed
    sample.spec_info.spec_verify_ct = verify
    sample.spec_info.completion_token_num = completion
    return sample


def _record(session_id, request_id, accepted, proposed, verify=0, completion=0, **coverage):
    return {
        "session_id": session_id,
        "request_id": request_id,
        "accepted": accepted,
        "proposed": proposed,
        "verify": verify,
        "completion": completion,
        "acceptance_present": coverage.get("acceptance_present", True),
        "length_present": coverage.get("length_present", True),
    }


def test_agentic_generation_nodes_are_deduplicated_before_weighted_aggregation():
    ancestor = _record("session-a", "request-a", 1, 2, 2, 3)
    branch_b = _record("session-a", "request-b", 9, 10, 10, 11)
    branch_c = _record("session-a", "request-c", 4, 5, 5, 6)

    metrics = aggregate_speculative_metrics(
        [_sample(ancestor), _sample(ancestor, branch_b), _sample(ancestor, branch_c)]
    )

    assert metrics["spec_accept_rate"] == 14 / 17
    assert metrics["spec_accept_length"] == 20 / 17
    assert metrics["spec_generation_count"] == 3


def test_same_text_different_requests_and_sessions_are_kept_separate():
    metrics = aggregate_speculative_metrics(
        [
            _sample(_record("s1", "r1", 1, 2)),
            _sample(_record("s1", "r2", 1, 2)),
            _sample(_record("s2", "r1", 9, 10)),
        ]
    )
    assert metrics["spec_accept_rate"] == 11 / 14
    assert metrics["spec_generation_count"] == 3


def test_zero_denominators_and_missing_backend_fields_are_reported_explicitly():
    metrics = aggregate_speculative_metrics(
        [_sample(_record("s", "r", 0, 0, 0, 0, acceptance_present=True, length_present=True))]
    )
    assert metrics["spec_accept_rate"] == 0.0
    assert metrics["spec_accept_length"] == 0.0
    assert metrics["spec_acceptance_coverage"] == 1.0
    assert metrics["spec_length_coverage"] == 1.0


def test_missing_backend_counters_are_not_reported_as_zero_rates():
    metrics = aggregate_speculative_metrics(
        [_sample(_record("s", "r", 0, 0, 0, 0, acceptance_present=False, length_present=False))]
    )
    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics


def test_legacy_samples_do_not_invent_zero_rate_when_no_counters_exist():
    metrics = aggregate_speculative_metrics([_sample()])
    assert metrics["spec_metrics_legacy"] is True
    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics


def test_legacy_samples_use_weighted_totals_when_counters_exist():
    metrics = aggregate_speculative_metrics([_sample(accepted=1, proposed=2), _sample(accepted=9, proposed=10)])
    assert metrics["spec_accept_rate"] == 10 / 12
    assert metrics["spec_metrics_legacy"] is True


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_session_forest_export_carries_shared_generation_identity_into_metrics():
    forest = SessionForest.create_empty(
        session_id="session-a",
        group_index=0,
        index=0,
        label=None,
        train_metadata=None,
        metadata=None,
    )
    observation = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": "start"}]),
        train_token_delta=[ord("p")],
        rollout_token_delta=[ord("p")],
    )

    def response(parent, request_id, token):
        counters = {
            "request-a": (1, 2, 2, 3),
            "request-b": (9, 10, 10, 11),
            "request-c": (4, 5, 5, 6),
        }[request_id]
        return forest.append_resp(
            parent_state_hash=parent,
            rollout_id=0,
            abort_count=0,
            messages_delta=check_messages([{"role": "assistant", "content": token}]),
            token_delta=[ord(token)],
            logprob_delta=[-0.1],
            spec_delta={
                "spec_accept_token_num": counters[0],
                "spec_draft_token_num": counters[1],
                "spec_verify_ct": counters[2],
                "completion_token_num": counters[3],
            },
            export_metadata_patch={
                "spec_generation": {
                    "session_id": "session-a",
                    "request_id": request_id,
                    "accepted": counters[0],
                    "proposed": counters[1],
                    "verify": counters[2],
                    "completion": counters[3],
                    "acceptance_present": True,
                    "length_present": True,
                }
            },
        )

    ancestor = response(observation.state_hash, "request-a", "a")
    branch_b = response(ancestor.state_hash, "request-b", "b")
    branch_c = response(ancestor.state_hash, "request-c", "c")
    sample_b = forest.build_sample(leaf_state_hash=branch_b.state_hash, tokenizer=_Tokenizer())
    sample_c = forest.build_sample(leaf_state_hash=branch_c.state_hash, tokenizer=_Tokenizer())

    metrics = aggregate_speculative_metrics([sample_b, sample_c])

    assert [record["request_id"] for record in sample_b.metadata["spec_generation_nodes"]] == [
        "request-a",
        "request-b",
    ]
    assert all(record["generation_node_id"] for record in sample_b.metadata["spec_generation_nodes"])
    assert metrics["spec_accept_rate"] == 14 / 17
    assert metrics["spec_generation_count"] == 3
