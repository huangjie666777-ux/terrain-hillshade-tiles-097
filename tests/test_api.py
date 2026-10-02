from __future__ import annotations

import io
import numpy as np
from PIL import Image

from terrain import app as app_module
from fastapi.testclient import TestClient


def test_api_upload_publish_and_tile(temp_service, make_tif, tmp_path):
    registry, renderer, _tmp = temp_service
    app_module.registry = registry
    app_module.renderer = renderer
    client = TestClient(app_module.app)

    values = (1000 + np.arange(100, dtype=np.float32)[None, :] * 2 + np.arange(100)[:, None]).astype(np.float32)
    tif = tmp_path / "dem.tif"
    make_tif(tif, values, bounds=(99.80, 26.84, 99.92, 26.96))
    response = client.post(
        "/datasets",
        files={"file": ("dem.tif", tif.read_bytes(), "image/tiff")},
    )
    assert response.status_code == 200, response.text
    dataset_id = response.json()["id"]

    bad = client.post("/datasets", files={"file": ("x.tif", b"bad")})
    assert bad.status_code == 400

    response = client.post("/versions", json={"version": "v1", "datasets": [dataset_id]})
    assert response.status_code == 200
    conflict = client.post("/versions", json={"version": "v1", "datasets": []})
    assert conflict.status_code in (400, 409, 422)

    tile_response = client.get("/tiles/v1/12/3184/1730.png?azimuth=315&altitude=45")
    assert tile_response.status_code == 200
    image = Image.open(io.BytesIO(tile_response.content))
    assert image.size == (256, 256) and image.mode == "RGBA"

    assert client.get("/tiles/missing/12/3184/1730.png").status_code == 404
    assert client.get("/tiles/v1/19/0/0.png").status_code == 400
    assert client.get("/tiles/v1/12/3184/1730.png?azimuth=360&altitude=45").status_code == 400
