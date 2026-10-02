"""Service-wide limits and paths."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "datasets"
LIST_DIR = DATA_DIR / "lists"

MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MiB per GeoTIFF upload
MAX_CONCURRENT_TILES = 4             # concurrent hillshade computations
TILE_CACHE_CAPACITY = 128            # LRU entries for rendered PNG tiles

TILE_SIZE = 256
MIN_ZOOM = 0
MAX_ZOOM = 18

ALLOWED_EPSG = {4326, 3857}
