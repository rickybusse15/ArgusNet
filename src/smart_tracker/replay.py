from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .coordinates import ENUOrigin
from .models import PlatformFrame, to_jsonable


ReplayDocument = Dict[str, object]

_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "replay-schema.json"


def _load_replay_schema() -> Dict[str, Any]:
    """Load the JSON Schema for replay documents from the docs directory."""
    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_replay_with_schema(document: Mapping[str, Any]) -> List[str]:
    """Validate a replay document dict against the formal JSON Schema.

    Returns a list of human-readable error strings.  An empty list means the
    document is valid.  Uses the ``jsonschema`` library when available and
    falls back to a lightweight manual check otherwise.
    """
    schema = _load_replay_schema()

    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        return _manual_schema_validation(document, schema)

    validator_cls = jsonschema.Draft7Validator
    validator = validator_cls(schema)
    errors: List[str] = []
    for error in sorted(validator.iter_errors(dict(document)), key=lambda e: list(e.absolute_path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


def _manual_schema_validation(
    document: Mapping[str, Any],
    schema: Dict[str, Any],
) -> List[str]:
    """Lightweight fallback validation when jsonschema is not installed."""
    errors: List[str] = []

    # Top-level type check
    if not isinstance(document, Mapping):
        errors.append("(root): document must be an object")
        return errors

    # Required top-level keys
    for key in schema.get("required", []):
        if key not in document:
            errors.append(f"(root): '{key}' is a required property")

    # --- meta ---
    meta = document.get("meta")
    if meta is not None:
        if not isinstance(meta, dict):
            errors.append("meta: must be an object")
        else:
            meta_schema = schema.get("properties", {}).get("meta", {})
            for key in meta_schema.get("required", []):
                if key not in meta:
                    errors.append(f"meta: '{key}' is a required property")
            dt_s = meta.get("dt_s")
            if dt_s is not None and not isinstance(dt_s, (int, float)):
                errors.append("meta.dt_s: must be a number")
            elif isinstance(dt_s, (int, float)) and dt_s <= 0:
                errors.append("meta.dt_s: must be greater than 0")

    # --- frames ---
    frames = document.get("frames")
    if frames is not None:
        if not isinstance(frames, list):
            errors.append("frames: must be an array")
        elif len(frames) < 1:
            errors.append("frames: must contain at least 1 item")
        else:
            frame_schema = schema.get("definitions", {}).get("frame", {})
            required_frame_keys = frame_schema.get("required", [])
            for idx, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    errors.append(f"frames.{idx}: must be an object")
                    continue
                for key in required_frame_keys:
                    if key not in frame:
                        errors.append(f"frames.{idx}: '{key}' is a required property")

                # Type checks for frame fields
                if "timestamp_s" in frame and not isinstance(frame["timestamp_s"], (int, float)):
                    errors.append(f"frames.{idx}.timestamp_s: must be a number")
                for list_field in ("nodes", "observations", "rejected_observations", "tracks", "truths"):
                    if list_field in frame and not isinstance(frame[list_field], list):
                        errors.append(f"frames.{idx}.{list_field}: must be an array")
                if "metrics" in frame and not isinstance(frame["metrics"], dict):
                    errors.append(f"frames.{idx}.metrics: must be an object")

    return errors


def validate_replay_document(document: Mapping[str, object]) -> None:
    errors: List[str] = []
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("Replay document must contain a non-empty frames list.")

    meta = document.get("meta")
    if not isinstance(meta, Mapping):
        errors.append("Replay document must contain a meta object.")
    else:
        dt_s = meta.get("dt_s")
        if not isinstance(dt_s, (int, float)) or not np.isfinite(dt_s) or dt_s <= 0.0:
            errors.append("Replay document meta.dt_s must be a finite value greater than 0.")

        frame_count = meta.get("frame_count")
        if isinstance(frames, list) and frame_count is not None:
            try:
                expected_frame_count = int(frame_count)
            except (TypeError, ValueError):
                expected_frame_count = None
            if expected_frame_count is not None and expected_frame_count != len(frames):
                errors.append("Replay document meta.frame_count must match the number of frames.")

    errors.extend(validate_replay_with_schema(document))
    unique_errors = list(dict.fromkeys(errors))
    if unique_errors:
        preview = unique_errors[:5]
        message = (
            f"Replay document validation failed with {len(unique_errors)} error(s): "
            + "; ".join(preview)
        )
        if len(unique_errors) > len(preview):
            message += f"; ... (+{len(unique_errors) - len(preview)} more)"
        raise ValueError(message)


def build_replay_document(
    frames: Sequence[PlatformFrame],
    scenario_name: str,
    dt_s: float,
    seed: int,
    extra_meta: Optional[Dict[str, object]] = None,
    enu_origin: Optional[ENUOrigin] = None,
) -> ReplayDocument:
    frame_list = list(frames)
    errors = [
        frame.metrics.mean_error_m
        for frame in frame_list
        if frame.metrics.mean_error_m is not None
    ]
    observation_counts = [frame.metrics.observation_count for frame in frame_list]
    accepted_counts = [frame.metrics.accepted_observation_count for frame in frame_list]
    rejected_counts = [frame.metrics.rejected_observation_count for frame in frame_list]
    active_track_counts = [frame.metrics.active_track_count for frame in frame_list]
    rejection_counter: Counter[str] = Counter()
    for frame in frame_list:
        rejection_counter.update(frame.metrics.rejection_counts)

    track_ids = sorted(
        {
            track.track_id
            for frame in frame_list
            for track in frame.tracks
        }
    )
    node_ids = sorted(
        {
            node.node_id
            for frame in frame_list
            for node in frame.nodes
        }
    )

    summary = {
        "mean_error_m": float(np.mean(errors)) if errors else None,
        "peak_error_m": float(np.max(errors)) if errors else None,
        "mean_observations_per_frame": float(np.mean(observation_counts)) if observation_counts else 0.0,
        "mean_active_tracks": float(np.mean(active_track_counts)) if active_track_counts else 0.0,
        "mean_accepted_observations_per_frame": float(np.mean(accepted_counts)) if accepted_counts else 0.0,
        "mean_rejected_observations_per_frame": float(np.mean(rejected_counts)) if rejected_counts else 0.0,
        "total_accepted_observations": int(sum(accepted_counts)),
        "total_rejected_observations": int(sum(rejected_counts)),
        "observation_rejection_rate": (
            float(sum(rejected_counts) / max(sum(accepted_counts) + sum(rejected_counts), 1))
            if (accepted_counts or rejected_counts)
            else 0.0
        ),
        "rejection_counts": dict(sorted(rejection_counter.items())),
    }

    meta: Dict[str, object] = {
        "scenario_name": scenario_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": len(frame_list),
        "dt_s": dt_s,
        "seed": seed,
        "node_ids": node_ids,
        "track_ids": track_ids,
    }
    if enu_origin is not None:
        meta["enu_origin"] = {
            "latitude_deg": enu_origin.latitude_deg,
            "longitude_deg": enu_origin.longitude_deg,
            "altitude_m": enu_origin.altitude_m,
        }
    if extra_meta:
        meta.update(extra_meta)

    document: ReplayDocument = {
        "meta": meta,
        "summary": summary,
        "frames": to_jsonable(frame_list),
    }
    validate_replay_document(document)
    return document


def write_replay_document(path: str, document: ReplayDocument) -> None:
    validate_replay_document(document)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def load_replay_document(path: str) -> ReplayDocument:
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    validate_replay_document(document)
    return document
