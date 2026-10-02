# Terrain Hillshade Tile Service

Pure-backend FastAPI service for uploading single-band elevation GeoTIFFs and serving 256×256 XYZ grayscale RGBA PNG hillshade tiles in EPSG:3857.

## Modules

- `terrain/storage.py` — atomic GeoTIFF ingestion, validation, dataset metadata, and immutable published versions.
- `terrain/mosaic.py` — bounded source-window reads, reprojection to Web Mercator pixel centers, bilinear elevation sampling, and ordered hole filling.
- `terrain/hillshade.py` — Horn 3×3 gradients on the stitched one-pixel-padded grid, Lambert diffuse shading, and latitude-corrected ground spacing.
- `terrain/tiles.py` — XYZ validation, 258×258 render grid, 256×256 crop, bounded render semaphore, and bounded LRU PNG cache.
- `terrain/app.py` — HTTP API.
- `scripts/generate_samples.py` — creates a slope raster with a NoData/explicit-mask hole and an overlapping lower-priority fill raster.

## Run

```bash
.venv/bin/uvicorn terrain.app:app --host 127.0.0.1 --port 8000
```

Persistent files default to `data/`: GeoTIFFs are in `data/datasets/`, temporary uploads in `data/tmp/`, and dataset/version metadata in `data/registry.json`. The registry is read at startup, so datasets and versions survive a process restart.

Generate the sample pair:

```bash
.venv/bin/python scripts/generate_samples.py --out samples
```

## HTTP Protocol

### Upload one raster

```bash
curl -F 'file=@samples/01_slope_with_hole.tif;type=image/tiff' \
  http://127.0.0.1:8000/datasets
```

The response contains `id` and persistent metadata. Uploads must be single-band GeoTIFF elevation rasters in metres, with CRS `EPSG:4326` or `EPSG:3857`, and a valid non-identity affine transform. Unknown CRS, multiple bands, missing georeferencing, zero/non-invertible transforms, empty files, oversize uploads, and corrupt files are rejected. The file is validated in a temporary path and atomically renamed only after success; failed attempts leave no dataset.

### Publish an ordered, immutable version

```bash
curl -sS -X POST http://127.0.0.1:8000/versions \
  -H 'Content-Type: application/json' \
  -d '{"version":"demo-v1","datasets":["PRIMARY_ID","FILL_ID"]}'
```

Dataset order is priority order. For each target pixel, the first valid elevation in the ordered list is used. NoData and mask holes in a higher-priority dataset may therefore be filled by a lower-priority dataset. Duplicate IDs and unknown IDs are rejected. A version cannot be overwritten with a different ordered list; repeating the identical list is an idempotent success. Reads must name an existing version and use the immutable records snapshot for that request.

### Fetch a tile

```bash
curl -fL 'http://127.0.0.1:8000/tiles/demo-v1/12/3184/1730.png?azimuth=315&altitude=45' \
  -o tile.png
```

- Path: `/tiles/{version}/{z}/{x}/{y}.png`.
- Zoom range: `0 <= z <= 18`; `x` and `y` are standard XYZ tile coordinates and are range-checked for the zoom.
- `azimuth`: degrees clockwise from north, `[0, 360)`; default `315`.
- `altitude`: degrees above the horizon, `(0, 90]`; default `45`.
- Output: 256×256, 8-bit grayscale replicated into RGB plus alpha, PNG.
- Outside source coverage the alpha channel is zero. NoData is never treated as zero elevation or allowed into bilinear interpolation: a target center is accepted only when its bilinear source kernel is valid.
- A 1-pixel border is rendered before mosaicking/gradients, then removed. A tile pixel is transparent unless all nine Horn neighbours are valid after priority fill, avoiding artificial cliffs at holes.
- Horizontal spacing uses Web Mercator projected pixel size multiplied by `cos(latitude)`, not raw projected metres.

Other endpoints:

- `GET /health`
- `GET /datasets`
- `GET /datasets/{id}`
- `GET /versions/{version}`

## Limits and Configuration

Environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `TERRAIN_DATA_DIR` | `data` | persistent dataset and registry directory |
| `TERRAIN_MAX_UPLOAD_BYTES` | `67108864` | accepted upload size limit (64 MiB) |
| `TERRAIN_MAX_CONCURRENT_JOBS` | `2` | simultaneous tile render jobs |
| `TERRAIN_CACHE_MAX_TILES` | `256` | in-memory LRU PNG tile capacity |
| `TERRAIN_MAX_ZOOM` | `18` | maximum supported XYZ zoom |

The cache key is `(version, z, x, y, azimuth, altitude)`. Tile rendering never reads a complete source raster: each source is intersected with the expanded tile bounds and read through a Rasterio window.

## Tests and Build Check

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q terrain scripts tests
```

Tests cover persistence/restart, invalid uploads and cleanup, immutable ordered versions, transparent no-coverage output, ordered hole fill, Horn-neighbour transparency, lighting-sensitive output, and cache-key separation.
