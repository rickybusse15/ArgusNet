from __future__ import annotations

import json

from argusnet.evaluation.recording import RotatingFrameRecorder
from argusnet.evaluation.replay import write_streaming_replay_document


def test_rotating_recorder_indexes_and_assembles_all_segments(tmp_path) -> None:
    source = tmp_path / "frames.jsonl"
    recorder = RotatingFrameRecorder(source, rotate_bytes=40, rotate_seconds=3600)
    for index in range(5):
        recorder.write(json.dumps({"timestamp_s": index, "nodes": []}) + "\n")
    recorder.close()

    session = json.loads(source.with_suffix(".session.json").read_text())
    assert session["active"] is False
    assert len(session["segments"]) > 1

    output = tmp_path / "replay.json"
    write_streaming_replay_document(source, output, meta={}, summary={})
    replay = json.loads(output.read_text())
    assert [frame["timestamp_s"] for frame in replay["frames"]] == list(range(5))
