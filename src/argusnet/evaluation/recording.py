"""Bounded, crash-tolerant JSONL recording for continuous live sessions."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TextIO


class RotatingFrameRecorder:
    """File-like JSONL writer that rotates by elapsed time or encoded bytes."""

    def __init__(
        self,
        path: Path,
        *,
        rotate_seconds: float = 900.0,
        rotate_bytes: int = 1 << 30,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rotate_seconds = rotate_seconds
        self.rotate_bytes = rotate_bytes
        self.index_path = self.path.with_suffix(".session.json")
        self._segment_index = 0
        self._segments: list[dict[str, object]] = []
        self._file: TextIO
        self._opened_at: float
        self._bytes = 0
        self._open_segment()

    def _segment_path(self) -> Path:
        if self._segment_index == 0:
            return self.path
        return self.path.with_name(f"{self.path.stem}.part-{self._segment_index:04d}.jsonl")

    def _write_index(self) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        payload = {
            "format": "argusnet-rotating-jsonl-v1",
            "active": True,
            "segments": self._segments,
        }
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.index_path)

    def _open_segment(self) -> None:
        segment_path = self._segment_path()
        self._file = segment_path.open("w", encoding="utf-8")
        self._opened_at = time.monotonic()
        self._bytes = 0
        self._segments.append({"path": segment_path.name, "index": self._segment_index, "bytes": 0})
        self._write_index()

    def write(self, value: str) -> int:
        encoded_bytes = len(value.encode("utf-8"))
        elapsed = time.monotonic() - self._opened_at
        if self._bytes and (
            self._bytes + encoded_bytes > self.rotate_bytes or elapsed >= self.rotate_seconds
        ):
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._segment_index += 1
            self._open_segment()
        written = self._file.write(value)
        self._file.flush()
        self._bytes += encoded_bytes
        self._segments[-1]["bytes"] = self._bytes
        return written

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._write_index()
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        payload["active"] = False
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.index_path)

    @property
    def closed(self) -> bool:
        return self._file.closed
