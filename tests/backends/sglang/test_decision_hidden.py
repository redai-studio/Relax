# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from relax.agentic.decision_hidden import write_record
from relax.backends.sglang.decision_hidden import DecisionPool, apply_patch, validate_server


def test_pool_matches_explicit_mean_and_empty() -> None:
    # Rows 0..2 are prompt-only; rows 3..6 predict assistant tokens.
    hidden = torch.arange(35, dtype=torch.float32).reshape(7, 5).to(torch.bfloat16)
    pool = DecisionPool(hidden[3], "worker:0")
    for row, token in zip(hidden[3:], [7, 8, 9, 2], strict=True):
        pool.add(row, include=token != 2, version="worker:0")
    result = pool.finish()
    assert result["token_count"] == 3
    torch.testing.assert_close(torch.tensor(result["vector"]), hidden[3:6].float().mean(0))
    empty = DecisionPool(hidden[0], "worker:0")
    empty.add(hidden[0], include=False, version="worker:0")
    assert empty.finish()["token_count"] == 0
    assert empty.finish()["vector"] == [0.0] * 5
    with pytest.raises(RuntimeError, match="Weights changed"):
        pool.add(hidden[0], include=True, version="worker:1")


def test_sidecars_disabled_and_retry_join(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RELAX_DECISION_HIDDEN_DIR", raising=False)
    write_record("requests", {"invalid": object()})
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("RELAX_DECISION_HIDDEN_DIR", str(tmp_path))
    for turn, request in [(0, "req_s_0:0"), (0, "req_s_1:0"), (1, "req_s_2:0")]:
        write_record("requests", {"task_id": "task", "session_id": "s", "turn_id": turn, "request_id": request})
    write_record("outcomes", {"session_id": "s", "reward": 0.0, "success": False})
    rows = [json.loads(line) for path in tmp_path.glob("requests-*.jsonl") for line in path.read_text().splitlines()]
    outcomes = {
        r["session_id"]: r
        for path in tmp_path.glob("outcomes-*.jsonl")
        for r in map(json.loads, path.read_text().splitlines())
    }
    assert len({row["request_id"] for row in rows}) == 3
    assert [row["turn_id"] for row in rows] == [0, 0, 1]
    assert all(outcomes[row["session_id"]]["reward"] == 0 for row in rows)


@pytest.fixture
def patched_scheduler(monkeypatch):
    monkeypatch.delenv("RELAX_DECISION_HIDDEN_DIR", raising=False)

    class Processor:
        def _get_hidden_states_to_store(self, *args):
            return None

        def forward(self, hidden, indices):
            self._get_hidden_states_to_store(hidden, None, None, hidden, None, None, indices, None)
            return SimpleNamespace()

    class Scheduler:
        def __init__(self):
            self.server_args = SimpleNamespace(model_path="initial-checkpoint")
            self.model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["Qwen3VLForConditionalGeneration"]),
                hf_text_config=SimpleNamespace(hidden_size=3),
            )
            self.tp_rank = 0
            self.device = "cpu"

        def maybe_collect_customized_info(self, *args):
            pass

        def stream_output_generation(self, *args, **kwargs):
            pass

        def update_weights_from_tensor(self, request):
            return SimpleNamespace(success=request.success)

        update_weights_from_disk = update_weights_from_tensor
        update_weights_from_distributed = update_weights_from_tensor
        update_weights_from_ipc = update_weights_from_tensor
        post_process_weights = update_weights_from_tensor

    for name, cls in [("sglang.srt.layers.logits_processor", Processor), ("sglang.srt.managers.scheduler", Scheduler)]:
        module = ModuleType(name)
        setattr(module, "LogitsProcessor" if cls is Processor else "Scheduler", cls)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("relax.backends.sglang.decision_hidden.validate_server", lambda args: None)
    apply_patch()
    return Scheduler(), Processor()


def request(rid="req_session_0:0"):
    return SimpleNamespace(
        rid=rid,
        stream=False,
        return_hidden_states=False,
        grammar=None,
        eos_token_ids={2},
        sampling_params=SimpleNamespace(
            stop_token_ids={3},
            custom_params={"relax_decision_context": {"session_id": "s", "task_id": "t", "turn_id": 0}},
        ),
        tokenizer=SimpleNamespace(eos_token_id=2, additional_stop_token_ids={4}),
        output_ids=[],
        customized_info=None,
        send_token_offset=0,
        finished_reason=None,
        finished=lambda: False,
    )


def test_scheduler_alignment_special_tokens_tp_and_update(patched_scheduler) -> None:
    scheduler, processor = patched_scheduler
    req = request()
    hidden = torch.tensor([[99.0, 99.0, 99.0], [1.0, 2.0, 3.0], [88.0, 88.0, 88.0], [4.0, 5.0, 6.0]])
    # Pruned rows can still include prompt logprob positions. sample_indices
    # is authoritative, and output metadata must keep the tensor on-device.
    output = processor.forward(hidden, torch.tensor([1, 3]))
    torch.testing.assert_close(output._relax_decision_rows, hidden[[1, 3]])
    assert not hasattr(processor, "_relax_decision_rows")
    for token in [10, 11, 2, 3, 4]:
        req.output_ids.append(token)
        scheduler.maybe_collect_customized_info(0, req, output)
    req.finished = lambda: True
    req.output_ids.append(2)
    req.send_token_offset = 5  # SGLang can force intermediate nonstream output.
    scheduler.maybe_collect_customized_info(0, req, output)
    first = req.customized_info["relax_decision_hidden"][req.send_token_offset :][0]
    assert first["token_count"] == 2
    assert first["vector"] == [1.0, 2.0, 3.0]
    assert not hasattr(req, "_relax_decision_pool")
    scheduler.update_weights_from_tensor(SimpleNamespace(success=True))
    second_req = request("req_session_1:0")
    second_req.output_ids = [10]
    second_req.finished = lambda: True
    scheduler.maybe_collect_customized_info(1, second_req, output)
    second = second_req.customized_info["relax_decision_hidden"][-1]
    assert first["policy_version"].rsplit(":", 1)[0] == second["policy_version"].rsplit(":", 1)[0]
    assert first["policy_version"].endswith(":0")
    assert second["policy_version"].endswith(":1")
    scheduler.tp_rank = 1
    other_rank = request()
    scheduler.maybe_collect_customized_info(0, other_rank, output)
    assert other_rank.customized_info is None
    with pytest.raises(RuntimeError, match="Weight update failed"):
        scheduler.update_weights_from_tensor(SimpleNamespace(success=False))


def test_inspected_source_and_mode_guards() -> None:
    if importlib.util.find_spec("sglang") is None:
        pytest.skip("SGLang source not installed")
    args = SimpleNamespace(disable_cuda_graph=True, disable_overlap_schedule=True, pp_size=1, dp_size=1)
    try:
        validate_server(args)
    except RuntimeError as error:
        pytest.skip(f"Different SGLang source snapshot: {error}")
    args.pp_size = 2
    with pytest.raises(RuntimeError, match="pp_size"):
        validate_server(args)
    args.pp_size = 1
    args.disable_cuda_graph = False
    with pytest.raises(RuntimeError, match="disable_cuda_graph"):
        validate_server(args)


async def test_backend_hidden_context_preserves_generation_contract(monkeypatch, tmp_path) -> None:
    from relax.agentic.pipeline import runtime
    from relax.agentic.session.service import _normalized_chat_request

    monkeypatch.setenv("RELAX_DECISION_HIDDEN_DIR", str(tmp_path))
    normalized = _normalized_chat_request(
        {
            "messages": [{"role": "user", "content": "act"}],
            "relax_hidden_context": {"task_id": "task-a", "turn_id": 3},
        }
    )
    assert normalized["hidden_context"] == {"task_id": "task-a", "turn_id": 3}
    adapter = object.__new__(runtime.SGLangBackendAdapter)
    adapter._args = SimpleNamespace(
        sglang_router_ip="localhost",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        sglang_router_policy="cache_aware",
        slime_router_sticky=False,
    )
    adapter._session_lifecycle = False
    adapter.tokenizer = SimpleNamespace(pad_token_id=None)
    adapter.compiler = SimpleNamespace(processor=None)
    sent = []

    async def post(url, payload, **kwargs):
        sent.append(payload)
        return {
            "output_ids": [5],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "relax_decision_hidden": [{"backend_request_id": "req_s_0:0", "vector": [1.0, 2.0]}],
            },
        }

    monkeypatch.setattr(runtime, "post", post)
    params = {"max_new_tokens": 10, "temperature": 0.7, "custom_params": {"existing": True}}
    result = await adapter.generate(
        input_ids=[1, 2, 3],
        sampling_params=params,
        session_id="s",
        request_id="req_s_0:0",
        image_data=["image"],
        hidden_context=normalized["hidden_context"],
    )
    assert result.new_tokens == [5]
    assert "relax_decision_hidden" not in result.meta_info  # Never export vectors as training fields.
    assert sent[0]["input_ids"] == [1, 2, 3]
    assert sent[0]["image_data"] == ["image"]
    assert sent[0]["sampling_params"]["temperature"] == 0.7
    assert sent[0]["sampling_params"]["custom_params"] == {
        "existing": True,
        "relax_decision_context": {"task_id": "task-a", "turn_id": 3, "session_id": "s", "trajectory_id": "s"},
    }
    assert params == {"max_new_tokens": 10, "temperature": 0.7, "custom_params": {"existing": True}}
    assert not list(tmp_path.iterdir())  # Engine owns persistence, not the HTTP adapter.


def test_transport_reexecution_and_abort_saved_once(patched_scheduler, monkeypatch, tmp_path) -> None:
    scheduler, processor = patched_scheduler
    monkeypatch.setenv("RELAX_DECISION_HIDDEN_DIR", str(tmp_path))
    output = processor.forward(torch.ones(1, 3), None)
    for _ in range(2):
        req = request("req_reexecuted:0")
        req.output_ids = [10]
        req.finished = lambda: True
        scheduler.maybe_collect_customized_info(0, req, output)
        scheduler.stream_output_generation([req], False)
    aborted = request("req_aborted:0")
    aborted.finished = lambda: True
    scheduler.stream_output_generation([aborted], False)
    scheduler.stream_output_generation([aborted], False)
    rows = [json.loads(line) for path in tmp_path.glob("requests-*.jsonl") for line in path.read_text().splitlines()]
    assert len(rows) == len({row["request_id"] for row in rows}) == 3
    assert rows[0]["backend_request_id"] == rows[1]["backend_request_id"]
    assert rows[2]["token_count"] == 0
    assert rows[2]["vector"] == [0.0, 0.0, 0.0]
