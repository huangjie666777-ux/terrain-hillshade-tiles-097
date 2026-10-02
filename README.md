# Hillshade Tile Service

Pure-backend terrain hillshade tile service. Upload single-band DEM GeoTIFFs
(meters), publish them as an immutable ordered list, and overlay 256x256
grayscale RGBA PNG hillshade tiles (XYZ, EPSG:3857) on engineering maps.

## Run

    .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

## Protocol

### POST /datasets
Multipart upload (file field) of a GeoTIFF. Accepted only if:
- single band, elevation in meters
- CRS is EPSG:4326 or EPSG:3857 with a valid (non-identity, finite) affine transform
- the file is a readable, non-corrupt GeoTIFF

Rejected uploads return 422/413 and leave no partial files. Valid datasets are
persisted under data/datasets/ with a JSON sidecar and survive restarts.
Returns {"id": ..., "meta": {...}}.

### POST /lists   body: {"datasets": ["<id>", ...]}
Publishes an immutable ordered list and returns {"version": n}. Versions are
monotonic and never overwritten; tile requests must name a version explicitly,
so publishing never disturbs concurrent readers. When mosaicking, the first
dataset in the list wins; NoData / masked holes may be filled from
lower-priority datasets.

### GET /lists/{version}
Returns the published list for that version.

### GET /tiles/{version}/{z}/{x}/{y}.png?azimuth=315&altitude=45
Renders a 256x256 grayscale RGBA PNG hillshade tile.
- z in 0..18; x,y within 0..2^z-1 (else 400)
- azimuth: light direction, degrees clockwise from north (0 <= a < 360)
- altitude: light elevation angle above horizon (0 < a <= 90)
- Pixels with no coverage (or an incomplete 3x3 neighborhood) are transparent;
  NoData is never treated as zero elevation nor mixed into interpolation.

## Method

- Tile bounds are expanded by 1 pixel; datasets are mosaicked onto the padded
  258x258 grid of EPSG:3857 pixel centers. Per dataset, only the intersecting
  raster window is read (never the whole file); samples are reprojected to the
  source CRS and bilinearly interpolated, skipping any output whose 2x2
  stencil touches NoData.
- Gradients use Horn's 3x3 method; brightness is Lambert diffuse
  (cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)),
  clipped at 0 on dark faces.
- Horizontal distances are corrected per row by cos(latitude) so mercator
  meters are not mistaken for ground meters.
- The 3x3 gradient consumes the 1px padding, yielding the final 256x256 tile.

## Limits (see app/config.py)

- Upload size: 64 MiB per file (MAX_UPLOAD_BYTES)
- Concurrent tile computations: 4 (MAX_CONCURRENT_TILES)
- Tile cache: 128-entry LRU (TILE_CACHE_CAPACITY); cache keys include list
  version and both lighting parameters.

## Modules

- app/storage.py    dataset registry + immutable versioned lists
- app/reproject.py  windowed reads, reprojection, bilinear mosaic
- app/hillshade.py  Horn gradients + Lambert shading
- app/tiles.py      pad/mosaic/shade/crop/PNG pipeline
- app/main.py       FastAPI HTTP layer, upload limits, concurrency, cache
- app/cache.py      bounded LRU tile cache

## Examples & tests

    .venv/bin/python examples/make_examples.py   # slope + overlapping-hole DEMs
    .venv/bin/python -m pytest tests -q
