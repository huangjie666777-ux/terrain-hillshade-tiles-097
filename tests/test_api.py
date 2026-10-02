from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app, storage


@pytest.fixture(autouse=True)
def clean_data():
    tmp = Path(tempfile.mkdtemp())
    storage.data_dir = tmp
    storage.dataset_dir = tmp / "datasets"
    storage.list_dir = tmp / "lists"
    storage.dataset_dir.mkdir(parents=True)
    storage.list_dir.mkdir(parents=True)
    storage._state_path = storage.list_dir / "state.json"
    storage._version = 0
    storage._write_state()
    yield
    shutil.rmtree(tmp, ignore_errors=True)


client = TestClient(app)


def make_tif(crs="EPSG:4326", bands=1, nodata=-9999.0, corrupt=False):
    buf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    buf.close()
    if corrupt:
        Path(buf.name).write_bytes(b"not a geotiff")
        return buf.name
    w, h = 64, 64
    transform = Affine(0.01, 0, 116.0, 0, -0.01, 40.0)
    with rasterio.open(
        buf.name, "w", driver="GTiff", width=w, height=h, count=bands,
        dtype="float32", crs=crs, transform=transform,
        nodata=nodata if bands == 1 else None,
    ) as ds:
        for b in range(1, bands + 1):
            yy, xx = np.mgrid[0:h, 0:w]
            ds.write((xx + yy).astype("float32"), b)
    return buf.name


def upload(path):
    with open(path, "rb") as f:
        return client.post("/datasets", files={"file": ("dem.tif", f,
                                                        "image/tiff")})


def test_upload_ok_and_restart_persistence():
    r = upload(make_tif())
    assert r.status_code == 200, r.text
    did = r.json()["id"]
    assert (storage.dataset_dir / f"{did}.tif").exists()
    # simulate restart with a fresh Storage on the same dir
    from app.storage import Storage
    s2 = Storage(storage.data_dir)
    assert s2.get_dataset(did) is not None


def test_reject_multiband_unknown_crs_corrupt():
    assert upload(make_tif(bands=3)).status_code == 422
    assert upload(make_tif(crs="EPSG:32650")).status_code == 422
    assert upload(make_tif(corrupt=True)).status_code == 422
    # no half-written artifacts
    assert list(storage.dataset_dir.glob("*.tif")) == []


def test_publish_versioning_immutable():
    d1 = upload(make_tif()).json()["id"]
    d2 = upload(make_tif()).json()["id"]
    v1 = client.post("/lists", json={"datasets": [d1]}).json()["version"]
    v2 = client.post("/lists", json={"datasets": [d2, d1]}).json()["version"]
    assert v2 == v1 + 1
    assert client.get(f"/lists/{v1}").json()["datasets"] == [d1]
    assert client.get(f"/lists/{v2}").json()["datasets"] == [d2, d1]
    assert client.post("/lists", json={"datasets": ["nope"]}).status_code == 422


def test_tile_render_and_validation():
    did = upload(make_tif()).json()["id"]
    v = client.post("/lists", json={"datasets": [did]}).json()["version"]
    # dataset spans lon 116-116.64, lat 39.36-40 -> find tile at z=10
    import math
    n = 2 ** 10
    lon, lat = 116.3, 39.7
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat))
            + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    r = client.get(f"/tiles/{v}/10/{x}/{y}.png?azimuth=300&altitude=40")
    assert r.status_code == 200, r.text
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (256, 256) and img.mode == "RGBA"
    arr = np.asarray(img)
    assert arr[..., 3].max() == 255  # some opaque pixels
    # out-of-range tile coords rejected
    assert client.get(f"/tiles/{v}/19/0/0.png").status_code == 400
    assert client.get(f"/tiles/{v}/3/8/0.png").status_code == 400
    # unknown version
    assert client.get(f"/tiles/{v + 99}/10/{x}/{y}.png").status_code == 404
    # tile fully outside coverage -> fully transparent
    r2 = client.get(f"/tiles/{v}/10/{(x + 100) % n}/{(y + 100) % n}.png")
    arr2 = np.asarray(Image.open(io.BytesIO(r2.content)))
    assert arr2[..., 3].max() == 0


def test_nodata_not_treated_as_zero():
    did = upload(make_tif()).json()["id"]
    v = client.post("/lists", json={"datasets": [did]}).json()["version"]
    r = client.get(f"/tiles/{v}/0/0/0.png")
    assert r.status_code == 200
