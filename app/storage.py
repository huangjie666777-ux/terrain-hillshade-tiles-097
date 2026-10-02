"""Dataset registry and immutable versioned dataset lists.

Layout on disk:
    data/datasets/<id>.tif        uploaded GeoTIFF (validated first)
    data/datasets/<id>.json       metadata (crs, transform, bounds, nodata)
    data/lists/v<version>.json    immutable ordered list of dataset ids
    data/lists/state.json         monotonic version counter
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError

from . import config


class DatasetError(ValueError):
    """Raised when an uploaded GeoTIFF fails validation."""


class Storage:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.dataset_dir = self.data_dir / "datasets"
        self.list_dir = self.data_dir / "lists"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.list_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state_path = self.list_dir / "state.json"
        if self._state_path.exists():
            self._version = json.loads(self._state_path.read_text())["version"]
        else:
            self._version = 0
            self._write_state()

    # ---------------- datasets ----------------

    def _write_state(self):
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": self._version}))
        tmp.replace(self._state_path)

    def save_dataset(self, src_path: Path) -> dict:
        """Validate the GeoTIFF at src_path and persist it. Raises DatasetError."""
        try:
            with rasterio.open(src_path) as ds:
                if ds.count != 1:
                    raise DatasetError(
                        f"only single-band rasters accepted, got {ds.count} bands"
                    )
                if ds.crs is None:
                    raise DatasetError("missing CRS; only EPSG:4326/3857 accepted")
                crs = CRS.from_user_input(ds.crs)
                epsg = crs.to_epsg()
                if epsg not in config.ALLOWED_EPSG:
                    raise DatasetError(f"unsupported CRS: {crs.to_string()}")
                transform = ds.transform
                if transform is None or transform.is_identity:
                    raise DatasetError("missing or identity affine transform")
                vals = (transform.a, transform.b, transform.d,
                        transform.e, transform.c, transform.f)
                if not all(v == v and abs(v) != float("inf") for v in vals):
                    raise DatasetError("invalid affine transform")
                # force a real read to catch corrupt payloads
                ds.read(1, window=((0, min(8, ds.height)), (0, min(8, ds.width))))
                meta = {
                    "crs": f"EPSG:{epsg}",
                    "width": ds.width,
                    "height": ds.height,
                    "transform": list(transform)[:6],
                    "bounds": list(ds.bounds),
                    "nodata": ds.nodata,
                    "dtype": ds.dtypes[0],
                }
        except DatasetError:
            raise
        except RasterioIOError as exc:
            raise DatasetError(f"corrupt or unreadable GeoTIFF: {exc}") from exc

        dataset_id = uuid.uuid4().hex[:12]
        tif_path = self.dataset_dir / f"{dataset_id}.tif"
        meta_path = self.dataset_dir / f"{dataset_id}.json"
        try:
            shutil.move(str(src_path), tif_path)
            meta["id"] = dataset_id
            meta["path"] = tif_path.name
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception:
            tif_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            raise
        return meta

    def get_dataset(self, dataset_id: str) -> dict | None:
        meta_path = self.dataset_dir / f"{dataset_id}.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        meta["path"] = str(self.dataset_dir / meta["path"])
        return meta

    def list_datasets(self) -> list[dict]:
        out = []
        for p in sorted(self.dataset_dir.glob("*.json")):
            meta = json.loads(p.read_text())
            meta["path"] = str(self.dataset_dir / meta["path"])
            out.append(meta)
        return out

    # ---------------- versioned lists ----------------

    def publish_list(self, dataset_ids: list[str]) -> int:
        for did in dataset_ids:
            if self.get_dataset(did) is None:
                raise DatasetError(f"unknown dataset id: {did}")
        with self._lock:
            self._version += 1
            version = self._version
            payload = {"version": version, "datasets": list(dataset_ids)}
            tmp = self.list_dir / f"v{version}.tmp"
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.list_dir / f"v{version}.json")
            self._write_state()
        return version

    def get_list(self, version: int) -> list[str] | None:
        p = self.list_dir / f"v{version}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())["datasets"]

    def latest_version(self) -> int:
        return self._version
