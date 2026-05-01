from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

COMMAND_SIM = "sim"
COMMAND_INGEST = "ingest"
COMMAND_EXPORT = "export"
COMMAND_BATCH_EXPORT = "batch-export"
COMMAND_BUILD_SCENE = "build-scene"
COMMAND_BENCHMARK = "benchmark"
COMMAND_VALIDATE_SCENE = "validate-scene"
COMMAND_VALIDATE_REPLAY = "validate-replay"
COMMAND_INFO = "info"
COMMAND_DUMP_CONFIG = "dump-config"

if TYPE_CHECKING:
    from argusnet.core.frames import ENUOrigin

ALL_COMMANDS = frozenset(
    {
        COMMAND_SIM,
        COMMAND_INGEST,
        COMMAND_EXPORT,
        COMMAND_BATCH_EXPORT,
        COMMAND_BUILD_SCENE,
        COMMAND_BENCHMARK,
        COMMAND_VALIDATE_SCENE,
        COMMAND_VALIDATE_REPLAY,
        COMMAND_INFO,
        COMMAND_DUMP_CONFIG,
    }
)

PROJECT_DEPENDENCY_MODULES = frozenset(
    {"grpc", "numpy", "google", "google.protobuf", "protobuf", "pyproj", "tifffile"}
)

PYTHON_DEPENDENCY_INSTALL_HINT = (
    "ArgusNet requires Python dependencies "
    "(`numpy`, `grpcio`, `protobuf`, `pyproj`, and `tifffile`). "
    "Install project dependencies with `python3 -m pip install --user -e .`."
)


def _module_import_hint(error: ModuleNotFoundError, *, command: str) -> RuntimeError:
    missing_name = error.name or "an unknown dependency"
    if missing_name not in PROJECT_DEPENDENCY_MODULES:
        raise error
    message = (
        f"Simulation requires the Python dependency `{missing_name}`. "
        f"{PYTHON_DEPENDENCY_INSTALL_HINT}"
    )
    return RuntimeError(message)


def _import_sim_module():
    try:
        from . import sim
    except ModuleNotFoundError as error:
        raise _module_import_hint(error, command=COMMAND_SIM) from error
    return sim


def _import_scene_module():
    try:
        from . import scene
    except ModuleNotFoundError as error:
        missing_name = error.name or "an unknown dependency"
        if missing_name not in PROJECT_DEPENDENCY_MODULES:
            raise error
        raise RuntimeError(
            f"Scene compilation requires the Python dependency `{missing_name}`. "
            f"{PYTHON_DEPENDENCY_INSTALL_HINT}"
        ) from error
    return scene


def _import_benchmark_module():
    try:
        from . import benchmark
    except ModuleNotFoundError as error:
        raise _module_import_hint(error, command=COMMAND_BENCHMARK) from error
    return benchmark


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add --verbose / --quiet flags to a parser."""
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )
    log_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress all output except errors.",
    )


def _configure_logging(args: argparse.Namespace) -> None:
    """Set up logging based on --verbose / --quiet flags."""
    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.ERROR
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def _iter_with_progress(
    items: Sequence[str],
    *,
    enabled: bool,
    description: str,
) -> Iterable[str]:
    if not enabled or not sys.stderr.isatty():
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=description, unit="format")


def build_parser(command: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argusnet",
        description="Run the ArgusNet simulator, scene compiler, and exporters.",
    )
    _add_logging_args(parser)
    subparsers = parser.add_subparsers(dest="command")

    # --- sim ---
    sim_parser = subparsers.add_parser(COMMAND_SIM, help="Run the simulation.")
    _add_logging_args(sim_parser)
    sim_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print simulation configuration and exit without running.",
    )
    if command in (None, COMMAND_SIM):
        try:
            _import_sim_module().add_cli_arguments(sim_parser)
        except RuntimeError as error:
            sim_parser.set_defaults(_import_error=str(error))
            sim_parser.add_argument("remainder", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    # --- ingest ---
    ingest_parser = subparsers.add_parser(
        COMMAND_INGEST,
        help="Ingest live sensor data via MQTT, or replay a saved replay.json.",
    )
    _add_logging_args(ingest_parser)
    source_group = ingest_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--mqtt-broker", default=None, help="MQTT broker hostname (live ingestion)."
    )
    source_group.add_argument(
        "--replay-file", default=None, help="Path to replay.json for file-based replay ingestion."
    )
    ingest_parser.add_argument(
        "--mqtt-port", type=int, default=1883, help="MQTT broker port (default: 1883)."
    )
    ingest_parser.add_argument(
        "--observation-topic", default="argusnet/observations", help="MQTT topic for observations."
    )
    ingest_parser.add_argument(
        "--node-topic", default="argusnet/nodes", help="MQTT topic for node states."
    )
    ingest_parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for --replay-file (default: 1.0; 0 = as fast as possible).",
    )
    ingest_parser.add_argument(
        "--replay-loop", action="store_true", help="Loop replay file continuously."
    )
    ingest_parser.add_argument(
        "--enu-origin", required=True, help="ENU origin as 'lat,lon,alt' (degrees, meters)."
    )
    ingest_parser.add_argument(
        "--endpoint", default=None, help="gRPC daemon endpoint (default: spawn local)."
    )
    ingest_parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.25,
        help="Frame flush interval in seconds (default: 0.25).",
    )
    ingest_parser.add_argument(
        "--replay-output", default=None, help="Optional path to write replay JSON on exit."
    )

    # --- export ---
    export_parser = subparsers.add_parser(
        COMMAND_EXPORT, help="Export replay data to external formats."
    )
    _add_logging_args(export_parser)
    export_parser.add_argument("--replay", required=True, help="Path to replay JSON file.")
    export_parser.add_argument(
        "--format",
        required=True,
        choices=["geojson", "czml", "foxglove", "kml", "kmz", "gpx", "geopackage", "shapefile"],
        help="Export format.",
    )
    export_parser.add_argument(
        "--enu-origin", required=True, help="ENU origin as 'lat,lon,alt' (degrees, meters)."
    )
    export_parser.add_argument("--output", required=True, help="Output file path.")
    export_parser.add_argument(
        "--include-observations",
        action="store_true",
        help="Include observations in export (GeoJSON only).",
    )
    export_parser.add_argument(
        "--include-nodes",
        action="store_true",
        help="Include node positions in export (GeoJSON only).",
    )
    export_parser.add_argument(
        "--time-range",
        default=None,
        help="Time range filter as 'start_s,end_s' (e.g., '10.0,60.0').",
    )

    batch_export_parser = subparsers.add_parser(
        COMMAND_BATCH_EXPORT,
        help="Export one replay to many formats in a single bounded batch.",
    )
    _add_logging_args(batch_export_parser)
    batch_export_parser.add_argument("--replay", required=True, help="Path to replay JSON file.")
    batch_export_parser.add_argument(
        "--enu-origin", required=True, help="ENU origin as 'lat,lon,alt' (degrees, meters)."
    )
    batch_export_parser.add_argument(
        "--formats",
        required=True,
        help="Comma-separated export formats, for example 'geojson,kml,kmz'.",
    )
    batch_export_parser.add_argument(
        "--output-dir", required=True, help="Directory for exported outputs."
    )
    batch_export_parser.add_argument(
        "--include-observations",
        action="store_true",
        help="Include observations where the format supports them.",
    )
    batch_export_parser.add_argument(
        "--include-nodes", action="store_true", help="Include nodes where the format supports them."
    )
    batch_export_parser.add_argument(
        "--time-range", default=None, help="Time range filter as 'start_s,end_s'."
    )

    # --- build-scene ---
    scene_parser = subparsers.add_parser(
        COMMAND_BUILD_SCENE, help="Compile a smartscene-v1 package."
    )
    _add_logging_args(scene_parser)
    scene_parser.add_argument("--output", required=True, help="Output scene package directory.")
    scene_parser.add_argument(
        "--scene-id", default=None, help="Optional scene identifier override."
    )
    scene_parser.add_argument(
        "--replay", default=None, help="Optional replay JSON file to package."
    )
    scene_parser.add_argument(
        "--environment-bundle", default=None, help="Optional smartmap-v1 environment bundle path."
    )
    scene_parser.add_argument(
        "--dem", default=None, help="Single-band DEM GeoTIFF for GIS scene compilation."
    )
    scene_parser.add_argument(
        "--source-crs",
        default=None,
        help="Optional CRS identifier for GIS inputs, for example EPSG:32611.",
    )
    scene_parser.add_argument(
        "--landcover", nargs="*", default=None, help="Optional GeoJSON land-cover polygons."
    )
    scene_parser.add_argument(
        "--roads", nargs="*", default=None, help="Optional GeoJSON road lines."
    )
    scene_parser.add_argument(
        "--water", nargs="*", default=None, help="Optional GeoJSON water polygons or lines."
    )
    scene_parser.add_argument(
        "--zones", nargs="*", default=None, help="Optional GeoJSON thematic polygons."
    )
    scene_parser.add_argument(
        "--buildings", nargs="*", default=None, help="Optional GeoJSON building polygons."
    )

    # --- benchmark ---
    benchmark_parser = subparsers.add_parser(
        COMMAND_BENCHMARK,
        help="Run canonical ArgusNet performance benchmarks.",
    )
    _add_logging_args(benchmark_parser)
    _import_benchmark_module().add_cli_arguments(benchmark_parser)

    # --- validate-scene ---
    validate_scene_parser = subparsers.add_parser(
        COMMAND_VALIDATE_SCENE,
        help="Validate a smartscene-v1 package (manifest, terrain chunks, metadata).",
    )
    _add_logging_args(validate_scene_parser)
    validate_scene_parser.add_argument("path", help="Path to the scene package directory.")

    # --- validate-replay ---
    validate_replay_parser = subparsers.add_parser(
        COMMAND_VALIDATE_REPLAY,
        help="Validate a replay JSON document.",
    )
    _add_logging_args(validate_replay_parser)
    validate_replay_parser.add_argument("path", help="Path to the replay JSON file.")

    # --- info ---
    info_parser = subparsers.add_parser(
        COMMAND_INFO,
        help="Display metadata, track count, duration, and statistics for a replay file.",
    )
    _add_logging_args(info_parser)
    info_parser.add_argument("path", help="Path to the replay JSON file.")

    # --- dump-config ---
    dump_config_parser = subparsers.add_parser(
        COMMAND_DUMP_CONFIG,
        help="Emit the default simulation configuration as JSON or YAML for editing.",
    )
    _add_logging_args(dump_config_parser)
    dump_config_parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json).",
    )
    dump_config_parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path (default: stdout).",
    )

    return parser


def _parse_enu_origin(value: str) -> ENUOrigin:
    from argusnet.core.frames import ENUOrigin

    parts = value.split(",")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("ENU origin must be 'lat,lon' or 'lat,lon,alt'")
    lat = float(parts[0])
    lon = float(parts[1])
    alt = float(parts[2]) if len(parts) == 3 else 0.0
    return ENUOrigin(lat, lon, alt)


def normalize_argv(argv: Sequence[str]) -> list[str]:
    if not argv:
        return [COMMAND_SIM]
    if argv[0] in {"-h", "--help"}:
        return list(argv)
    if argv[0] in ALL_COMMANDS:
        return list(argv)
    return [COMMAND_SIM, *argv]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    normalized_argv = normalize_argv(sys.argv[1:] if argv is None else argv)
    command = normalized_argv[0] if normalized_argv and normalized_argv[0] == COMMAND_SIM else None
    parser = build_parser(command)
    return parser.parse_args(normalized_argv)


def main(argv: Sequence[str] | None = None) -> None:
    normalized_argv = normalize_argv(sys.argv[1:] if argv is None else argv)
    if normalized_argv and normalized_argv[0] == COMMAND_SIM:
        try:
            _import_sim_module()
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
    args = parse_args(argv)
    _configure_logging(args)

    import_error = getattr(args, "_import_error", None)
    if import_error:
        raise SystemExit(import_error)

    command = args.command

    if command == COMMAND_VALIDATE_SCENE:
        _run_validate_scene(args)
        return
    if command == COMMAND_VALIDATE_REPLAY:
        _run_validate_replay(args)
        return
    if command == COMMAND_INFO:
        _run_info(args)
        return
    if command == COMMAND_DUMP_CONFIG:
        _run_dump_config(args)
        return
    if command == COMMAND_INGEST:
        _run_ingest(args)
        return
    if command == COMMAND_EXPORT:
        _run_export(args)
        return
    if command == COMMAND_BATCH_EXPORT:
        _run_batch_export(args)
        return
    if command == COMMAND_BUILD_SCENE:
        _run_build_scene(args)
        return
    if command == COMMAND_BENCHMARK:
        _import_benchmark_module().run_from_args(args)
        return

    # Default: sim
    if getattr(args, "dry_run", False):
        _run_sim_dry_run(args)
        return
    _import_sim_module().run_from_args(args)


# ---------------------------------------------------------------------------
# New commands
# ---------------------------------------------------------------------------


def _run_validate_scene(args: argparse.Namespace) -> None:
    scene_path = Path(args.path)
    if not scene_path.is_dir():
        raise SystemExit(f"Error: {scene_path} is not a directory.")

    manifest_path = scene_path / "scene_manifest.json"
    if not manifest_path.exists():
        manifest_path = scene_path / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Validation failed: missing scene_manifest.json or manifest.json in {scene_path}."
        )

    from argusnet.world.scene_loader import load_scene_manifest

    try:
        manifest = load_scene_manifest(manifest_path)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Validation failed: {manifest_path.name} is not valid JSON: {exc}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc

    errors: list[str] = []
    for index, layer in enumerate(manifest["layers"]):
        asset_path = layer["asset_path"]
        resolved_path = scene_path / asset_path
        if not resolved_path.exists():
            errors.append(f"Layer {index} references missing asset_path: {asset_path}")

    metadata = manifest["metadata"]
    for key in ("environment", "style"):
        relative_path = metadata[key]
        resolved_path = scene_path / relative_path
        if not resolved_path.exists():
            errors.append(f"Metadata references missing {key} path: {relative_path}")

    replay = manifest.get("replay")
    if isinstance(replay, dict):
        replay_path = replay.get("path")
        if isinstance(replay_path, str) and replay_path:
            resolved_path = scene_path / replay_path
            if not resolved_path.exists():
                errors.append(f"Replay reference is missing: {replay_path}")

    if errors:
        for err in errors:
            print(f"  FAIL: {err}", file=sys.stderr)
        raise SystemExit(f"Validation failed with {len(errors)} error(s).")
    print(f"Scene package at {scene_path} is valid.")


def _run_validate_replay(args: argparse.Namespace) -> None:
    replay_path = Path(args.path)
    if not replay_path.exists():
        raise SystemExit(f"Error: {replay_path} does not exist.")

    try:
        with open(replay_path, encoding="utf-8") as fh:
            document = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: {replay_path} is not valid JSON: {exc}") from exc

    from argusnet.evaluation.replay import validate_replay_document

    try:
        validate_replay_document(document)
    except ValueError as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc

    meta = document.get("meta", {})
    frame_count = meta.get("frame_count", len(document.get("frames", [])))
    print(f"Replay at {replay_path} is valid ({frame_count} frames).")


def _run_info(args: argparse.Namespace) -> None:
    replay_path = Path(args.path)
    if not replay_path.exists():
        raise SystemExit(f"Error: {replay_path} does not exist.")

    try:
        with open(replay_path, encoding="utf-8") as fh:
            document = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: {replay_path} is not valid JSON: {exc}") from exc

    meta = document.get("meta", {})
    summary = document.get("summary", {})
    frames = document.get("frames", [])

    frame_count = meta.get("frame_count", len(frames))
    dt_s = meta.get("dt_s", 0)
    duration_s = frame_count * dt_s if dt_s else 0
    scenario_name = meta.get("scenario_name", "unknown")
    seed = meta.get("seed", "N/A")
    node_ids = meta.get("node_ids", [])
    track_ids = meta.get("track_ids", [])
    generated_at = meta.get("generated_at_utc", "N/A")

    node_preview = f"{', '.join(node_ids[:5])}{'...' if len(node_ids) > 5 else ''}"
    track_preview = f"{', '.join(track_ids[:5])}{'...' if len(track_ids) > 5 else ''}"
    lines = [
        f"Replay: {replay_path.name}",
        f"  Scenario:     {scenario_name}",
        f"  Generated:    {generated_at}",
        f"  Seed:         {seed}",
        f"  Frame count:  {frame_count}",
        f"  Time step:    {dt_s} s",
        f"  Duration:     {duration_s:.1f} s",
        f"  Sensor nodes: {len(node_ids)} ({node_preview})",
        f"  Tracks:       {len(track_ids)} ({track_preview})",
    ]

    if summary:
        mean_err = summary.get("mean_error_m")
        peak_err = summary.get("peak_error_m")
        mean_obs = summary.get("mean_observations_per_frame")
        rejection_rate = summary.get("observation_rejection_rate")
        lines.append("  --- Statistics ---")
        if mean_err is not None:
            lines.append(f"  Mean error:        {mean_err:.2f} m")
        if peak_err is not None:
            lines.append(f"  Peak error:        {peak_err:.2f} m")
        if mean_obs is not None:
            lines.append(f"  Mean obs/frame:    {mean_obs:.1f}")
        if rejection_rate is not None:
            lines.append(f"  Rejection rate:    {rejection_rate:.1%}")

    print("\n".join(lines))


def _run_dump_config(args: argparse.Namespace) -> None:
    from argusnet.core.config import SimulationConstants

    constants = SimulationConstants.default()

    fmt = getattr(args, "format", "json")
    output = constants.to_yaml() if fmt == "yaml" else constants.to_json(indent=2)

    output_path = getattr(args, "output", None)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Default configuration written to {output_path}")
    else:
        print(output)


def _run_sim_dry_run(args: argparse.Namespace) -> None:
    """Print simulation configuration and exit without running."""
    provided = getattr(args, "_provided_sim_args", set()) or set()

    def _provided(dest: str) -> bool:
        return dest in provided

    constants = None
    try:
        from argusnet.core.config import SimulationConstants

        config_file = getattr(args, "config_file", None)
        if config_file:
            lowered = str(config_file).lower()
            if lowered.endswith(".json"):
                constants = SimulationConstants.from_json(config_file)
            else:
                constants = SimulationConstants.from_yaml(config_file)
        else:
            constants = SimulationConstants.default()
    except Exception:
        constants = None

    print("=== Simulation Dry Run ===")
    duration_s = getattr(args, "duration_s", "default")
    dt_s = getattr(args, "dt", "default")
    seed = getattr(args, "seed", "default")
    if constants is not None:
        duration_s = (
            duration_s if _provided("duration_s") else constants.dynamics.default_duration_s
        )
        dt_s = dt_s if _provided("dt") else constants.dynamics.default_dt_s
        seed = seed if _provided("seed") else constants.dynamics.default_seed
    print(f"  Duration:    {duration_s} s")
    print(f"  Time step:   {dt_s} s")
    print(f"  Seed:        {seed}")
    print(f"  Terrain:     {getattr(args, 'terrain_preset', 'default')}")
    print(f"  Weather:     {getattr(args, 'weather_preset', 'default')}")
    print(f"  Map preset:  {getattr(args, 'map_preset', 'default')}")
    print(f"  Targets:     {getattr(args, 'target_count', 'default')}")
    print(f"  Drones:      {getattr(args, 'drone_count', 'default')}")
    if getattr(args, "config_file", None):
        print(f"  Config file: {args.config_file}")

    try:
        if constants is None:
            raise RuntimeError("constants unavailable")
        print("\n  Config summary:")
        print(f"    Sensor bearing std: {constants.sensor.drone_base_bearing_std_rad:.4f} rad")
        print(f"    Drone base AGL:     {constants.dynamics.drone_base_agl_m:.0f} m")
        print(f"    Max stale steps:    {constants.dynamics.default_max_stale_steps}")
    except Exception:
        pass

    print("\nDry run complete — no simulation executed.")


# ---------------------------------------------------------------------------
# Existing commands (unchanged logic)
# ---------------------------------------------------------------------------


def _run_ingest(args: argparse.Namespace) -> None:
    from argusnet.adapters.argusnet_grpc import TrackerConfig, TrackingService
    from argusnet.sensing.ingestion.frame_stream import (
        FileReplayIngestionAdapter,
        LiveIngestionRunner,
        MQTTIngestionAdapter,
    )

    enu_origin = _parse_enu_origin(args.enu_origin)

    if args.replay_file:
        adapter = FileReplayIngestionAdapter(
            replay_path=args.replay_file,
            speed=args.replay_speed,
            loop=args.replay_loop,
        )
    else:
        adapter = MQTTIngestionAdapter(
            broker=args.mqtt_broker,
            port=args.mqtt_port,
            observation_topic=args.observation_topic,
            node_topic=args.node_topic,
            enu_origin=enu_origin,
        )

    service = TrackingService(
        config=TrackerConfig(),
        endpoint=args.endpoint,
    )
    replay_frames: list = [] if args.replay_output else None
    runner = LiveIngestionRunner(
        adapter=adapter,
        service=service,
        frame_interval_s=args.frame_interval,
        replay_frames=replay_frames,
    )
    adapter.start(on_frame=lambda *a: None)
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        adapter.stop()
        if args.replay_output and replay_frames:
            from argusnet.evaluation.replay import build_replay_document, write_replay_document

            doc = build_replay_document(
                replay_frames,
                scenario_name="live-ingest" if not args.replay_file else "file-replay",
                dt_s=args.frame_interval,
                seed=0,
                enu_origin=enu_origin,
            )
            write_replay_document(args.replay_output, doc)
            print(f"Replay written to {args.replay_output}")
        service.close()


def _run_export(args: argparse.Namespace) -> None:
    from argusnet.evaluation.export import export_replay_format
    from argusnet.evaluation.replay import load_replay_document

    enu_origin = _parse_enu_origin(args.enu_origin)
    replay_doc = load_replay_document(args.replay)
    fmt = args.format

    # Parse optional time range
    start_time_s = None
    end_time_s = None
    if args.time_range:
        parts = args.time_range.split(",")
        if len(parts) == 2:
            start_time_s = float(parts[0])
            end_time_s = float(parts[1])

    if fmt == "shapefile" and Path(args.output).suffix:
        raise SystemExit("Shapefile export requires --output to be a directory, not a file path.")
    export_replay_format(
        replay_doc,
        enu_origin,
        fmt,
        args.output,
        include_observations=args.include_observations,
        include_nodes=args.include_nodes,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )
    print(f"Exported {fmt} to {args.output}")


def _run_batch_export(args: argparse.Namespace) -> None:
    from argusnet.evaluation.export import (
        EXPORT_FORMATS,
        export_replay_format,
        suggested_output_path,
    )
    from argusnet.evaluation.replay import load_replay_document

    enu_origin = _parse_enu_origin(args.enu_origin)
    replay_doc = load_replay_document(args.replay)
    requested_formats = [
        value.strip().lower() for value in args.formats.split(",") if value.strip()
    ]
    unknown_formats = [value for value in requested_formats if value not in EXPORT_FORMATS]
    if unknown_formats:
        raise SystemExit(f"Unsupported batch-export formats: {', '.join(sorted(unknown_formats))}")

    start_time_s = None
    end_time_s = None
    if args.time_range:
        parts = args.time_range.split(",")
        if len(parts) == 2:
            start_time_s = float(parts[0])
            end_time_s = float(parts[1])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: list[str] = []
    for export_format in _iter_with_progress(
        requested_formats,
        enabled=not getattr(args, "quiet", False),
        description="Exporting replay",
    ):
        output_path = suggested_output_path(args.replay, export_format, output_dir)
        export_replay_format(
            replay_doc,
            enu_origin,
            export_format,
            str(output_path),
            include_observations=args.include_observations,
            include_nodes=args.include_nodes,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        exported_paths.append(str(output_path))

    if not getattr(args, "quiet", False):
        print("\n".join(exported_paths))


def _run_build_scene(args: argparse.Namespace) -> None:
    scene_module = _import_scene_module()

    overlay_paths = {
        key: value
        for key, value in {
            "landcover": args.landcover,
            "roads": args.roads,
            "water": args.water,
            "zones": args.zones,
            "buildings": args.buildings,
        }.items()
        if value
    }
    scene_module.build_scene_package(
        output_dir=args.output,
        replay=args.replay,
        environment_bundle=args.environment_bundle,
        dem_path=args.dem,
        source_crs=args.source_crs,
        overlay_paths=overlay_paths,
        scene_id=args.scene_id,
    )
    print(f"Scene package written to {args.output}")
