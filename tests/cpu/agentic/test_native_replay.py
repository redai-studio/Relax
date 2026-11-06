# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.agentic.native_replay import NativeReplayError, NativeReplayRecorder, iter_native_replay_events


def test_native_replay_preserves_multiturn_payloads_and_rejects_corruption(tmp_path):
    recorder = NativeReplayRecorder(tmp_path)
    payloads = [
        {"input_ids": tokens, "rid": f"turn-{index}", "sampling_params": {"max_new_tokens": 8}}
        for index, tokens in enumerate(([1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 9]))
    ]
    for payload in payloads:
        request = recorder.record_request(payload, session_id="session", routing_key="session")
        assert recorder.record_response(request, {"output_ids": [6, 7]}, elapsed_s=0.1)
    assert recorder.close()
    events = list(iter_native_replay_events(recorder.stream_dir))
    assert [event["payload"] for event in events if event["type"] == "request"] == payloads
    with (recorder.stream_dir / "input-deltas.bin").open("ab") as output:
        output.write(b"corruption")
    with pytest.raises(NativeReplayError, match="checksum mismatch"):
        list(iter_native_replay_events(recorder.stream_dir))


def test_native_replay_missing_terminal_remains_incomplete(tmp_path):
    recorder = NativeReplayRecorder(tmp_path)
    recorder.record_request({"input_ids": [1], "rid": "unfinished"}, session_id="session")
    assert not recorder.close()
    assert (recorder.stream_dir / "INCOMPLETE").is_file()
    with pytest.raises(NativeReplayError, match="not finalized"):
        list(iter_native_replay_events(recorder.stream_dir))
