# West Pleasant View, Colorado: the canonical real-world scenario (R5)

A small community at the foothills edge between Golden and Lakewood,
Jefferson County, Colorado. A 230/115 kV double-circuit transmission line
(Public Service Company of Colorado) crosses it. The run's medium map
(about 1.26 x 1.21 km) is centred on the place's OSM node, 39.7340 N,
105.1782 W. Both sources cover a 1.6 x 1.6 km box around it.

## Sources

### `dem.tif`: elevation

USGS 3D Elevation Program (3DEP) Bare Earth DEM. This is a U.S. Geological
Survey product, in the public domain. Requested 2026-10-21 from the 3DEP
dynamic image service, as a 320 x 320 float32 GeoTIFF at 5 m in UTM 13N
(EPSG:32613), bilinear:

    https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage
      ?bbox=483930.0,4397449.7,485530.0,4399049.7&bboxSR=32613&imageSR=32613
      &size=320,320&format=tiff&pixelType=F32&noData=-9999
      &interpolation=RSP_BilinearInterpolation&f=image

SHA-256 `4b06f1f6a48eecb89c9bb24fe4d9b4100b173fdfce68e71ad2fb096964d52daa`.

### `osm.json`: buildings, land use, roads, water, trees, power

Map data © OpenStreetMap contributors, available under the
Open Database License (ODbL) 1.0: https://www.openstreetmap.org/copyright.
A database derived from this extract must also be offered under the ODbL.

Requested 2026-10-21 (OSM data as of 2026-10-21T23:31:50Z) from the
Overpass API at overpass-api.de, as one `out geom` query for the box
39.7268,-105.1875,39.7412,-105.1689:

    [out:json][timeout:120];
    (way(39.7268,-105.1875,39.7412,-105.1689);
     relation[type=multipolygon](39.7268,-105.1875,39.7412,-105.1689);
     node[natural=tree](39.7268,-105.1875,39.7412,-105.1689);
     node[power~"^(tower|pole)$"](39.7268,-105.1875,39.7412,-105.1689););
    out geom;

SHA-256 `463efdc57671b7ed9704cbb25032a3d93edd486888f479d6e19282e9ef675499`.

## Rebuilding the world

    python3 scripts/import_world.py tests/fixtures/real/west_pleasant_view/dem.tif \
        tests/fixtures/real/west_pleasant_view/osm.json --map-preset medium \
        --origin 39.734,-105.1782 --out worlds/west_pleasant_view \
        --argusnet-rs target/release/argusnet-rs

`canonical.json` holds these parameters and the counts the import and the
run are checked against (`tests/test_canonical_world.py`).
