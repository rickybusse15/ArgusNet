"""Export replay data to GIS and visualization formats."""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from argusnet.core.frames import ENUOrigin, enu_to_wgs84

from .replay import ReplayDocument

EXPORT_FORMATS = (
    "geojson",
    "czml",
    "foxglove",
    "kml",
    "kmz",
    "gpx",
    "geopackage",
    "shapefile",
)


def _lazy_import_fiona():
    try:
        import fiona
    except ImportError as exc:
        raise RuntimeError(
            "Shapefile and GeoPackage export require `fiona`. "
            "Install export extras with `python3 -m pip install --user -e .[export]`."
        ) from exc
    return fiona


def _collect_track_lines(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> list[dict[str, Any]]:
    frames = _filter_frames(replay_document["frames"], start_time_s, end_time_s)
    track_positions: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for frame in frames:
        for track in frame.get("tracks", []):
            lat, lon, alt = enu_to_wgs84(np.array(track["position"], dtype=float), enu_origin)
            track_positions[track["track_id"]].append((lon, lat, alt, float(track["timestamp_s"])))

    features: list[dict[str, Any]] = []
    for track_id in sorted(track_positions.keys()):
        entries = track_positions[track_id]
        if len(entries) < 2:
            continue
        features.append(
            {
                "track_id": track_id,
                "coordinates_3d": [[lon, lat, alt] for lon, lat, alt, _ in entries],
                "coordinates_2d": [[lon, lat] for lon, lat, _alt, _ in entries],
                "point_count": len(entries),
                "start_time_s": entries[0][3],
                "end_time_s": entries[-1][3],
                "mean_alt_m": float(np.mean([alt for _lon, _lat, alt, _ in entries])),
            }
        )
    return features


def _collect_node_points(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> list[dict[str, Any]]:
    frames = _filter_frames(replay_document["frames"], start_time_s, end_time_s)
    seen_nodes: dict[str, dict[str, Any]] = {}
    for frame in frames:
        for node in frame.get("nodes", []):
            seen_nodes[node["node_id"]] = node

    features: list[dict[str, Any]] = []
    for node_id in sorted(seen_nodes.keys()):
        node = seen_nodes[node_id]
        lat, lon, alt = enu_to_wgs84(np.array(node["position"], dtype=float), enu_origin)
        features.append(
            {
                "node_id": node_id,
                "coordinate_3d": [lon, lat, alt],
                "coordinate_2d": [lon, lat],
                "is_mobile": int(bool(node.get("is_mobile", False))),
                "alt_m": float(alt),
            }
        )
    return features


def _collect_observation_points(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> list[dict[str, Any]]:
    frames = _filter_frames(replay_document["frames"], start_time_s, end_time_s)
    features: list[dict[str, Any]] = []
    for frame in frames:
        for observation in frame.get("observations", []):
            lat, lon, alt = enu_to_wgs84(np.array(observation["origin"], dtype=float), enu_origin)
            features.append(
                {
                    "node_id": observation.get("node_id", ""),
                    "target_id": observation.get("target_id", ""),
                    "coordinate_3d": [lon, lat, alt],
                    "coordinate_2d": [lon, lat],
                    "timestamp_s": float(observation.get("timestamp_s", 0.0)),
                    "alt_m": float(alt),
                }
            )
    return features


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------


def export_geojson(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_path: str,
    *,
    include_observations: bool = False,
    include_nodes: bool = False,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> None:
    """Export tracks (and optionally observations/nodes) as a GeoJSON FeatureCollection."""
    frames = _filter_frames(replay_document["frames"], start_time_s, end_time_s)

    # Collect track positions over time
    track_positions: dict[str, list[list[float]]] = defaultdict(list)
    track_timestamps: dict[str, list[float]] = defaultdict(list)

    for frame in frames:
        for track in frame.get("tracks", []):
            track_id = track["track_id"]
            pos = track["position"]
            lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
            track_positions[track_id].append([lon, lat, alt])
            track_timestamps[track_id].append(track["timestamp_s"])

    features: list[dict[str, Any]] = []

    # Track LineStrings
    for track_id in sorted(track_positions.keys()):
        coords = track_positions[track_id]
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "track_id": track_id,
                    "point_count": len(coords),
                    "start_time_s": track_timestamps[track_id][0],
                    "end_time_s": track_timestamps[track_id][-1],
                },
            }
        )

    # Observation Points
    if include_observations:
        for frame in frames:
            for obs in frame.get("observations", []):
                origin = obs["origin"]
                lat, lon, alt = enu_to_wgs84(np.array(origin, dtype=float), enu_origin)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat, alt],
                        },
                        "properties": {
                            "type": "observation",
                            "node_id": obs.get("node_id", ""),
                            "target_id": obs.get("target_id", ""),
                            "timestamp_s": obs.get("timestamp_s", 0.0),
                        },
                    }
                )

    # Node Points (latest position)
    if include_nodes:
        seen_nodes: dict[str, dict] = {}
        for frame in frames:
            for node in frame.get("nodes", []):
                seen_nodes[node["node_id"]] = node
        for node_id in sorted(seen_nodes.keys()):
            node = seen_nodes[node_id]
            pos = node["position"]
            lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat, alt],
                    },
                    "properties": {
                        "type": "node",
                        "node_id": node_id,
                        "is_mobile": node.get("is_mobile", False),
                    },
                }
            )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)


# ---------------------------------------------------------------------------
# CZML (Cesium)
# ---------------------------------------------------------------------------


def export_czml(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_path: str,
    *,
    start_time_utc: str | None = None,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> None:
    """Export tracks as CZML packets for Cesium visualization."""
    frames = _filter_frames(replay_document["frames"], start_time_s, end_time_s)
    meta = replay_document.get("meta", {})
    anchor_dt = _resolve_time_anchor(replay_document, explicit_start_time_utc=start_time_utc)

    # Collect track data
    track_data: dict[str, list[tuple]] = defaultdict(list)
    for frame in frames:
        t = frame.get("timestamp_s", 0.0)
        for track in frame.get("tracks", []):
            pos = track["position"]
            lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
            track_data[track["track_id"]].append((t, lon, lat, alt))

    packets: list[dict[str, Any]] = []

    # Document header
    anchor_iso = anchor_dt.isoformat()
    if frames:
        frame_times = [float(frame.get("timestamp_s", 0.0)) for frame in frames]
        interval_start_s = min(frame_times)
        interval_end_s = max(frame_times)
    else:
        interval_start_s = 0.0
        interval_end_s = 0.0
    total_duration = max(interval_end_s - interval_start_s, 0.0)
    start_iso = _iso_from_timestamp(anchor_dt, interval_start_s)
    end_iso = _iso_from_timestamp(anchor_dt, interval_end_s)

    packets.append(
        {
            "id": "document",
            "name": meta.get("scenario_name", "Smart Tracker Export"),
            "version": "1.0",
            "clock": {
                "interval": f"{start_iso}/{end_iso}",
                "currentTime": start_iso,
                "multiplier": 1,
            },
        }
    )

    # Track packets
    for track_id in sorted(track_data.keys()):
        entries = track_data[track_id]
        if not entries:
            continue

        # Build cartographicDegrees array: [time, lon, lat, alt, time, lon, lat, alt, ...]
        cartographic = []
        for t, lon, lat, alt in entries:
            cartographic.extend([t, lon, lat, alt])

        first_t = entries[0][0]
        last_t = entries[-1][0]
        avail_start = _iso_from_timestamp(anchor_dt, first_t)
        avail_end = _iso_from_timestamp(anchor_dt, last_t)

        packets.append(
            {
                "id": track_id,
                "name": track_id,
                "availability": f"{avail_start}/{avail_end}",
                "position": {
                    "epoch": anchor_iso,
                    "cartographicDegrees": cartographic,
                },
                "point": {
                    "color": {"rgba": [255, 165, 0, 255]},
                    "pixelSize": 8,
                },
                "path": {
                    "material": {
                        "solidColor": {
                            "color": {"rgba": [255, 165, 0, 200]},
                        }
                    },
                    "width": 2,
                    "leadTime": 0,
                    "trailTime": total_duration,
                },
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(packets, f, indent=2)


# ---------------------------------------------------------------------------
# Foxglove MCAP
# ---------------------------------------------------------------------------


def export_foxglove(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_path: str,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> None:
    """Export tracks as a Foxglove MCAP file with SceneUpdate messages.

    Requires the ``mcap-protobuf-support`` package::

        pip install mcap-protobuf-support
    """
    try:
        from mcap.writer import Writer as McapWriter
    except ImportError as exc:
        raise ImportError(
            "Foxglove export requires 'mcap'. Install with: pip install mcap-protobuf-support"
        ) from exc

    frames = _filter_frames(replay_document["frames"], start_time_s, end_time_s)
    meta = replay_document.get("meta", {})
    dt_s = float(meta.get("dt_s", 0.25))
    anchor_dt = _resolve_time_anchor(replay_document)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as f:
        writer = McapWriter(f)
        writer.start()

        schema_id = writer.register_schema(
            name="argusnet.TrackPositions",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "timestamp_s": {"type": "number"},
                        "tracks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "track_id": {"type": "string"},
                                    "lat_deg": {"type": "number"},
                                    "lon_deg": {"type": "number"},
                                    "alt_m": {"type": "number"},
                                },
                            },
                        },
                    },
                }
            ).encode("utf-8"),
        )

        channel_id = writer.register_channel(
            schema_id=schema_id,
            topic="/argusnet/tracks",
            message_encoding="json",
        )

        for i, frame in enumerate(frames):
            timestamp_s = frame.get("timestamp_s", i * dt_s)
            timestamp_ns = _timestamp_to_ns(anchor_dt, timestamp_s)

            track_entries = []
            for track in frame.get("tracks", []):
                pos = track["position"]
                lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
                track_entries.append(
                    {
                        "track_id": track["track_id"],
                        "lat_deg": lat,
                        "lon_deg": lon,
                        "alt_m": alt,
                    }
                )

            message_data = json.dumps(
                {
                    "timestamp_s": timestamp_s,
                    "tracks": track_entries,
                }
            ).encode("utf-8")

            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=message_data,
            )

        writer.finish()


# ---------------------------------------------------------------------------
# KML
# ---------------------------------------------------------------------------

_KML_NS = "http://www.opengis.net/kml/2.2"


def export_kml(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_path: str,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> str:
    """Export tracks and sensor nodes as a KML document.

    Parameters
    ----------
    replay_document:
        A replay document dict with ``meta`` and ``frames``.
    enu_origin:
        Reference origin for ENU-to-WGS84 conversion.
    output_path:
        Destination file path.
    start_time_s:
        If given, only include frames whose ``timestamp_s >= start_time_s``.
    end_time_s:
        If given, only include frames whose ``timestamp_s <= end_time_s``.

    Returns
    -------
    str
        The *output_path* that was written to.
    """
    frames = replay_document["frames"]
    meta = replay_document.get("meta", {})
    anchor_dt = _resolve_time_anchor(replay_document)

    # Filter frames by time range
    filtered_frames = _filter_frames(frames, start_time_s, end_time_s)

    # Collect track positions keyed by track_id
    track_positions: dict[str, list[tuple]] = defaultdict(list)
    for frame in filtered_frames:
        for track in frame.get("tracks", []):
            track_id = track["track_id"]
            pos = track["position"]
            lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
            track_positions[track_id].append((lon, lat, alt, track["timestamp_s"]))

    # Collect latest node positions
    seen_nodes: dict[str, dict] = {}
    for frame in filtered_frames:
        for node in frame.get("nodes", []):
            seen_nodes[node["node_id"]] = node

    # Build KML tree
    kml = ET.Element("kml", xmlns=_KML_NS)
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = meta.get("scenario_name", "Smart Tracker Export")

    # Track folder
    tracks_folder = ET.SubElement(document, "Folder")
    ET.SubElement(tracks_folder, "name").text = "Tracks"

    for track_id in sorted(track_positions.keys()):
        entries = track_positions[track_id]
        if len(entries) < 2:
            continue
        placemark = ET.SubElement(tracks_folder, "Placemark")
        ET.SubElement(placemark, "name").text = track_id

        # TimeSpan
        timespan = ET.SubElement(placemark, "TimeSpan")
        ET.SubElement(timespan, "begin").text = _iso_from_timestamp(anchor_dt, entries[0][3])
        ET.SubElement(timespan, "end").text = _iso_from_timestamp(anchor_dt, entries[-1][3])

        linestring = ET.SubElement(placemark, "LineString")
        ET.SubElement(linestring, "altitudeMode").text = "absolute"
        coord_text = " ".join(f"{lon},{lat},{alt}" for lon, lat, alt, _t in entries)
        ET.SubElement(linestring, "coordinates").text = coord_text

    # Nodes folder
    nodes_folder = ET.SubElement(document, "Folder")
    ET.SubElement(nodes_folder, "name").text = "Sensor Nodes"

    for node_id in sorted(seen_nodes.keys()):
        node = seen_nodes[node_id]
        pos = node["position"]
        lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
        placemark = ET.SubElement(nodes_folder, "Placemark")
        ET.SubElement(placemark, "name").text = node_id
        point = ET.SubElement(placemark, "Point")
        ET.SubElement(point, "altitudeMode").text = "absolute"
        ET.SubElement(point, "coordinates").text = f"{lon},{lat},{alt}"

    # Write
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    tree.write(str(output), xml_declaration=True, encoding="UTF-8")

    return str(output_path)


def export_kmz(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_path: str,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> str:
    """Export KML content as a single-document KMZ archive."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="argusnet-kmz-") as temp_dir:
        kml_path = Path(temp_dir) / "doc.kml"
        export_kml(
            replay_document,
            enu_origin,
            str(kml_path),
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(kml_path, arcname="doc.kml")
    return str(output_path)


# ---------------------------------------------------------------------------
# GPX
# ---------------------------------------------------------------------------

_GPX_NS = "http://www.topografix.com/GPX/1/1"


def export_gpx(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_path: str,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> str:
    """Export tracks and sensor nodes as a GPX 1.1 document.

    Parameters
    ----------
    replay_document:
        A replay document dict with ``meta`` and ``frames``.
    enu_origin:
        Reference origin for ENU-to-WGS84 conversion.
    output_path:
        Destination file path.
    start_time_s:
        If given, only include frames whose ``timestamp_s >= start_time_s``.
    end_time_s:
        If given, only include frames whose ``timestamp_s <= end_time_s``.

    Returns
    -------
    str
        The *output_path* that was written to.
    """
    frames = replay_document["frames"]
    replay_document.get("meta", {})
    anchor_dt = _resolve_time_anchor(replay_document)

    # Filter frames by time range
    filtered_frames = _filter_frames(frames, start_time_s, end_time_s)

    # Collect track positions keyed by track_id
    track_positions: dict[str, list[tuple]] = defaultdict(list)
    for frame in filtered_frames:
        for track in frame.get("tracks", []):
            track_id = track["track_id"]
            pos = track["position"]
            lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
            track_positions[track_id].append((lat, lon, alt, track["timestamp_s"]))

    # Collect latest node positions
    seen_nodes: dict[str, dict] = {}
    for frame in filtered_frames:
        for node in frame.get("nodes", []):
            seen_nodes[node["node_id"]] = node

    # Build GPX tree
    gpx = ET.Element(
        "gpx",
        version="1.1",
        creator="argusnet",
        xmlns=_GPX_NS,
    )

    # Waypoints for sensor nodes
    for node_id in sorted(seen_nodes.keys()):
        node = seen_nodes[node_id]
        pos = node["position"]
        lat, lon, alt = enu_to_wgs84(np.array(pos, dtype=float), enu_origin)
        wpt = ET.SubElement(gpx, "wpt", lat=str(lat), lon=str(lon))
        ET.SubElement(wpt, "ele").text = str(alt)
        ET.SubElement(wpt, "name").text = node_id

    # Tracks
    for track_id in sorted(track_positions.keys()):
        entries = track_positions[track_id]
        trk = ET.SubElement(gpx, "trk")
        ET.SubElement(trk, "name").text = track_id
        trkseg = ET.SubElement(trk, "trkseg")
        for lat, lon, alt, t in entries:
            trkpt = ET.SubElement(trkseg, "trkpt", lat=str(lat), lon=str(lon))
            ET.SubElement(trkpt, "ele").text = str(alt)
            ET.SubElement(trkpt, "time").text = _iso_from_timestamp(anchor_dt, t)

    # Write
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(gpx)
    ET.indent(tree, space="  ")
    tree.write(str(output), xml_declaration=True, encoding="UTF-8")

    return str(output_path)


def export_geopackage(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_path: str,
    *,
    include_observations: bool = False,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> str:
    """Export tracks, nodes, and optional observations to a single GeoPackage."""
    fiona = _lazy_import_fiona()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    track_features = _collect_track_lines(
        replay_document,
        enu_origin,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )
    node_features = _collect_node_points(
        replay_document,
        enu_origin,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )
    observation_features = (
        _collect_observation_points(
            replay_document,
            enu_origin,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        if include_observations
        else []
    )

    with fiona.open(
        output,
        mode="w",
        driver="GPKG",
        layer="tracks",
        crs="EPSG:4326",
        schema={
            "geometry": "LineString",
            "properties": {
                "track_id": "str",
                "point_count": "int",
                "start_time_s": "float",
                "end_time_s": "float",
                "mean_alt_m": "float",
            },
        },
    ) as collection:
        for feature in track_features:
            collection.write(
                {
                    "geometry": {"type": "LineString", "coordinates": feature["coordinates_2d"]},
                    "properties": {
                        "track_id": feature["track_id"],
                        "point_count": feature["point_count"],
                        "start_time_s": feature["start_time_s"],
                        "end_time_s": feature["end_time_s"],
                        "mean_alt_m": feature["mean_alt_m"],
                    },
                }
            )

    with fiona.open(
        output,
        mode="a",
        driver="GPKG",
        layer="nodes",
        crs="EPSG:4326",
        schema={
            "geometry": "Point",
            "properties": {"node_id": "str", "is_mobile": "int", "alt_m": "float"},
        },
    ) as collection:
        for feature in node_features:
            collection.write(
                {
                    "geometry": {"type": "Point", "coordinates": feature["coordinate_2d"]},
                    "properties": {
                        "node_id": feature["node_id"],
                        "is_mobile": feature["is_mobile"],
                        "alt_m": feature["alt_m"],
                    },
                }
            )

    if include_observations:
        with fiona.open(
            output,
            mode="a",
            driver="GPKG",
            layer="observations",
            crs="EPSG:4326",
            schema={
                "geometry": "Point",
                "properties": {
                    "node_id": "str",
                    "target_id": "str",
                    "timestamp_s": "float",
                    "alt_m": "float",
                },
            },
        ) as collection:
            for feature in observation_features:
                collection.write(
                    {
                        "geometry": {"type": "Point", "coordinates": feature["coordinate_2d"]},
                        "properties": {
                            "node_id": feature["node_id"],
                            "target_id": feature["target_id"],
                            "timestamp_s": feature["timestamp_s"],
                            "alt_m": feature["alt_m"],
                        },
                    }
                )

    return str(output_path)


def export_shapefile(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    output_dir: str,
    *,
    include_observations: bool = False,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> str:
    """Export tracks, nodes, and optional observations as Shapefile sets."""
    fiona = _lazy_import_fiona()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    track_features = _collect_track_lines(
        replay_document,
        enu_origin,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )
    node_features = _collect_node_points(
        replay_document,
        enu_origin,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )
    observation_features = (
        _collect_observation_points(
            replay_document,
            enu_origin,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        if include_observations
        else []
    )

    with fiona.open(
        output_root / "tracks.shp",
        mode="w",
        driver="ESRI Shapefile",
        crs="EPSG:4326",
        schema={
            "geometry": "LineString",
            "properties": {
                "track_id": "str",
                "point_count": "int",
                "start_t_s": "float",
                "end_t_s": "float",
                "mean_alt_m": "float",
            },
        },
    ) as collection:
        for feature in track_features:
            collection.write(
                {
                    "geometry": {"type": "LineString", "coordinates": feature["coordinates_2d"]},
                    "properties": {
                        "track_id": feature["track_id"],
                        "point_count": feature["point_count"],
                        "start_t_s": feature["start_time_s"],
                        "end_t_s": feature["end_time_s"],
                        "mean_alt_m": feature["mean_alt_m"],
                    },
                }
            )

    with fiona.open(
        output_root / "nodes.shp",
        mode="w",
        driver="ESRI Shapefile",
        crs="EPSG:4326",
        schema={
            "geometry": "Point",
            "properties": {"node_id": "str", "is_mobile": "int", "alt_m": "float"},
        },
    ) as collection:
        for feature in node_features:
            collection.write(
                {
                    "geometry": {"type": "Point", "coordinates": feature["coordinate_2d"]},
                    "properties": {
                        "node_id": feature["node_id"],
                        "is_mobile": feature["is_mobile"],
                        "alt_m": feature["alt_m"],
                    },
                }
            )

    if include_observations:
        with fiona.open(
            output_root / "observations.shp",
            mode="w",
            driver="ESRI Shapefile",
            crs="EPSG:4326",
            schema={
                "geometry": "Point",
                "properties": {
                    "node_id": "str",
                    "target_id": "str",
                    "time_s": "float",
                    "alt_m": "float",
                },
            },
        ) as collection:
            for feature in observation_features:
                collection.write(
                    {
                        "geometry": {"type": "Point", "coordinates": feature["coordinate_2d"]},
                        "properties": {
                            "node_id": feature["node_id"],
                            "target_id": feature["target_id"],
                            "time_s": feature["timestamp_s"],
                            "alt_m": feature["alt_m"],
                        },
                    }
                )

    return str(output_dir)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _filter_frames(
    frames: list[dict[str, Any]],
    start_time_s: float | None,
    end_time_s: float | None,
) -> list[dict[str, Any]]:
    """Return frames within the optional ``[start_time_s, end_time_s]`` window."""
    if start_time_s is None and end_time_s is None:
        return frames
    result = []
    for frame in frames:
        t = frame.get("timestamp_s", 0.0)
        if start_time_s is not None and t < start_time_s:
            continue
        if end_time_s is not None and t > end_time_s:
            continue
        result.append(frame)
    return result


def _parse_iso_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_time_anchor(
    replay_document: ReplayDocument,
    *,
    explicit_start_time_utc: str | None = None,
) -> datetime:
    explicit_anchor = _parse_iso_timestamp(explicit_start_time_utc)
    if explicit_anchor is not None:
        return explicit_anchor

    meta = replay_document.get("meta", {})
    if isinstance(meta, dict):
        generated_at = _parse_iso_timestamp(meta.get("generated_at_utc"))
        if generated_at is not None:
            return generated_at
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _iso_from_timestamp(anchor_dt: datetime, timestamp_s: float) -> str:
    return (anchor_dt + timedelta(seconds=float(timestamp_s))).isoformat()


def _timestamp_to_ns(anchor_dt: datetime, timestamp_s: float) -> int:
    return int((anchor_dt + timedelta(seconds=float(timestamp_s))).timestamp() * 1e9)


def export_replay_format(
    replay_document: ReplayDocument,
    enu_origin: ENUOrigin,
    export_format: str,
    output_path: str,
    *,
    include_observations: bool = False,
    include_nodes: bool = False,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> str:
    """Dispatch one replay export format to the correct implementation."""
    fmt = export_format.lower()
    if fmt == "geojson":
        export_geojson(
            replay_document,
            enu_origin,
            output_path,
            include_observations=include_observations,
            include_nodes=include_nodes,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    elif fmt == "czml":
        export_czml(
            replay_document,
            enu_origin,
            output_path,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    elif fmt == "foxglove":
        export_foxglove(
            replay_document,
            enu_origin,
            output_path,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    elif fmt == "kml":
        export_kml(
            replay_document,
            enu_origin,
            output_path,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    elif fmt == "kmz":
        export_kmz(
            replay_document,
            enu_origin,
            output_path,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    elif fmt == "gpx":
        export_gpx(
            replay_document,
            enu_origin,
            output_path,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    elif fmt == "geopackage":
        export_geopackage(
            replay_document,
            enu_origin,
            output_path,
            include_observations=include_observations,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    elif fmt == "shapefile":
        export_shapefile(
            replay_document,
            enu_origin,
            output_path,
            include_observations=include_observations,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    return output_path


def suggested_output_path(
    replay_path: str | os.PathLike[str],
    export_format: str,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Build the default output path for batch-export."""
    replay_stem = Path(replay_path).stem
    output_root = Path(output_dir)
    fmt = export_format.lower()
    suffix_map = {
        "geojson": ".geojson",
        "czml": ".czml",
        "foxglove": ".mcap",
        "kml": ".kml",
        "kmz": ".kmz",
        "gpx": ".gpx",
        "geopackage": ".gpkg",
    }
    if fmt == "shapefile":
        return output_root / f"{replay_stem}-shapefile"
    return output_root / f"{replay_stem}{suffix_map[fmt]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "export_geojson",
    "export_czml",
    "export_foxglove",
    "export_kml",
    "export_kmz",
    "export_gpx",
    "export_geopackage",
    "export_replay_format",
    "export_shapefile",
    "suggested_output_path",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``argusnet-export``."""
    import argparse

    from .cli import _parse_enu_origin
    from .replay import load_replay_document

    parser = argparse.ArgumentParser(prog="argusnet-export", description="Export replay data.")
    parser.add_argument("--replay", required=True, help="Path to replay JSON.")
    parser.add_argument("--format", required=True, choices=list(EXPORT_FORMATS))
    parser.add_argument("--enu-origin", required=True, help="ENU origin as 'lat,lon,alt'.")
    parser.add_argument("--output", required=True, help="Output file path.")
    parser.add_argument("--include-observations", action="store_true")
    parser.add_argument("--include-nodes", action="store_true")
    parser.add_argument(
        "--time-range", help="Time range as 'start,end' in seconds (e.g. '0.0,10.0')."
    )
    args = parser.parse_args()

    enu_origin = _parse_enu_origin(args.enu_origin)
    replay_doc = load_replay_document(args.replay)

    time_kwargs: dict[str, float | None] = {}
    if args.time_range:
        parts = args.time_range.split(",")
        time_kwargs["start_time_s"] = float(parts[0]) if parts[0] else None
        time_kwargs["end_time_s"] = float(parts[1]) if len(parts) > 1 and parts[1] else None

    export_replay_format(
        replay_doc,
        enu_origin,
        args.format,
        args.output,
        include_observations=args.include_observations,
        include_nodes=args.include_nodes,
        **time_kwargs,
    )

    print(f"Exported {args.format} to {args.output}")
