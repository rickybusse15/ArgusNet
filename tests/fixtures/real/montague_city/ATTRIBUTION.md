# Montague City, Massachusetts: the second real-world site

A village on the Connecticut River in Montague, Franklin County,
Massachusetts. Fourteen mapped power lines meet there. The run's medium map
(about 1.26 x 1.21 km) is centred on 42.5860 N, 72.5740 W. Both sources cover
a 1.6 x 1.6 km box around it.

It was chosen to differ from West Pleasant View, before any run had been made
on it: a river, a different building stock, many lines instead of one, and more
relief. Five river-valley places were compared by feature counts alone (one
count query failed); this one had the most transmission structures with a
village under them.

OSM maps no trees, woods or land use that the importer reads here, so the
world has none. Islands in mapped water are drawn as water (the importer
ignores holes, and counts 18).

## Sources

### `dem.tif`: elevation

USGS 3D Elevation Program (3DEP) Bare Earth DEM. This is a U.S. Geological
Survey product, in the public domain. Requested 2026-11-23 from the 3DEP
dynamic image service, as a 320 x 320 float32 GeoTIFF at 5 m in UTM 18N
(EPSG:32618), bilinear:

    https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage
      ?bbox=698267,4716895,699867,4718495&bboxSR=32618&imageSR=32618
      &size=320,320&format=tiff&pixelType=F32&noData=-9999
      &interpolation=RSP_BilinearInterpolation&f=image

SHA-256 `da12f7863ebd9a7030aa9a4c4c3396378d97abf601ad6cf6b946fdd117d5ad16`.

### `osm.json`: buildings, land use, roads, water, trees, power

Map data © OpenStreetMap contributors, available under the
Open Database License (ODbL) 1.0: https://www.openstreetmap.org/copyright.
A database derived from this extract must also be offered under the ODbL.

Requested 2026-11-23 (OSM data as of 2026-11-23T08:31:51Z) from the
Overpass API at overpass-api.de, as one `out geom` query for the box
42.5788,-72.5838,42.5932,-72.5642:

    [out:json][timeout:60];
    (way(42.5788,-72.5838,42.5932,-72.5642);
     relation[type=multipolygon](42.5788,-72.5838,42.5932,-72.5642);
     node[natural=tree](42.5788,-72.5838,42.5932,-72.5642);
     node[power~"^(tower|pole)$"](42.5788,-72.5838,42.5932,-72.5642););
    out geom;

SHA-256 `b35a8233075457e9fc0cabad0a2d92cc863bbc1783214a7b606a626251a7d799`.

## Rebuilding the world

    python3 scripts/import_world.py tests/fixtures/real/montague_city/dem.tif \
        tests/fixtures/real/montague_city/osm.json --map-preset medium \
        --origin 42.586,-72.574 --out worlds/montague_city \
        --argusnet-rs target/release/argusnet-rs

`canonical.json` holds these parameters and the counts the import and the
run are checked against (`tests/test_canonical_world.py`).
