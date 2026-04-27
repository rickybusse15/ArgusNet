from __future__ import annotations

from collections.abc import Iterable

STYLE_FORMAT_VERSION = "smartstyle-v1"


def hex_to_rgba(value: str, alpha: float = 1.0) -> list[float]:
    encoded = value.lstrip("#")
    if len(encoded) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {value!r}")
    return [
        int(encoded[0:2], 16) / 255.0,
        int(encoded[2:4], 16) / 255.0,
        int(encoded[4:6], 16) / 255.0,
        float(alpha),
    ]


def default_style() -> dict[str, dict[str, object]]:
    return {
        "terrain-base": {
            "id": "terrain-base",
            "semantic_kind": "terrain",
            "color_rgba": hex_to_rgba("#A7B58B", 1.0),
            "opacity": 1.0,
            "elevation_mode": "terrain-draped",
            "draw_order": 0,
            "default_visibility": True,
        },
        "landcover-open": {
            "id": "landcover-open",
            "semantic_kind": "landcover",
            "color_rgba": hex_to_rgba("#C7DE9A", 0.66),
            "opacity": 0.66,
            "elevation_mode": "terrain-draped",
            "draw_order": 10,
            "default_visibility": True,
        },
        "landcover-forest": {
            "id": "landcover-forest",
            "semantic_kind": "landcover",
            "color_rgba": hex_to_rgba("#9FC07A", 0.72),
            "opacity": 0.72,
            "elevation_mode": "terrain-draped",
            "draw_order": 11,
            "default_visibility": True,
        },
        "landcover-water": {
            "id": "landcover-water",
            "semantic_kind": "water",
            "color_rgba": hex_to_rgba("#BFD8E6", 0.8),
            "opacity": 0.8,
            "elevation_mode": "terrain-draped",
            "draw_order": 12,
            "default_visibility": True,
        },
        "landcover-urban": {
            "id": "landcover-urban",
            "semantic_kind": "zones",
            "color_rgba": hex_to_rgba("#E6C5EB", 0.62),
            "opacity": 0.62,
            "elevation_mode": "terrain-draped",
            "draw_order": 13,
            "default_visibility": True,
        },
        "roads": {
            "id": "roads",
            "semantic_kind": "roads",
            "color_rgba": hex_to_rgba("#C9C7C2", 1.0),
            "opacity": 1.0,
            "elevation_mode": "line-clamped",
            "draw_order": 20,
            "default_visibility": True,
            "line_width_m": 6.0,
        },
        "water": {
            "id": "water",
            "semantic_kind": "water",
            "color_rgba": hex_to_rgba("#D7E6F1", 0.85),
            "opacity": 0.85,
            "elevation_mode": "terrain-draped",
            "draw_order": 21,
            "default_visibility": True,
            "line_width_m": 8.0,
        },
        "zones": {
            "id": "zones",
            "semantic_kind": "zones",
            "color_rgba": hex_to_rgba("#EABBD6", 0.72),
            "opacity": 0.72,
            "elevation_mode": "terrain-draped",
            "draw_order": 22,
            "default_visibility": True,
        },
        "buildings": {
            "id": "buildings",
            "semantic_kind": "buildings",
            "color_rgba": hex_to_rgba("#F5EFE4", 0.45),
            "opacity": 0.45,
            "elevation_mode": "extruded",
            "draw_order": 30,
            "default_visibility": True,
        },
        "vegetation": {
            "id": "vegetation",
            "semantic_kind": "landcover",
            "color_rgba": hex_to_rgba("#B7D48C", 0.6),
            "opacity": 0.6,
            "elevation_mode": "extruded",
            "draw_order": 31,
            "default_visibility": True,
        },
        "walls": {
            "id": "walls",
            "semantic_kind": "buildings",
            "color_rgba": hex_to_rgba("#D2D8DF", 0.50),
            "opacity": 0.50,
            "elevation_mode": "extruded",
            "draw_order": 32,
            "default_visibility": True,
        },
        "tracks": {
            "id": "tracks",
            "semantic_kind": "tracks",
            "color_rgba": hex_to_rgba("#FF9A4D", 1.0),
            "opacity": 1.0,
            "elevation_mode": "runtime",
            "draw_order": 100,
            "default_visibility": True,
        },
        "truths": {
            "id": "truths",
            "semantic_kind": "truths",
            "color_rgba": hex_to_rgba("#4DD5E7", 1.0),
            "opacity": 1.0,
            "elevation_mode": "runtime",
            "draw_order": 101,
            "default_visibility": True,
        },
        "nodes": {
            "id": "nodes",
            "semantic_kind": "nodes",
            "color_rgba": hex_to_rgba("#F3F0A5", 1.0),
            "opacity": 1.0,
            "elevation_mode": "runtime",
            "draw_order": 102,
            "default_visibility": True,
        },
    }


def style_document(style_ids: Iterable[str]) -> dict[str, object]:
    base = default_style()
    return {
        "style_version": STYLE_FORMAT_VERSION,
        "layers": [base[style_id] for style_id in style_ids if style_id in base],
    }


__all__ = [
    "STYLE_FORMAT_VERSION",
    "default_style",
    "hex_to_rgba",
    "style_document",
]
