"""Telling your cats apart: enrolment, identification, storage and the embedder."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.detectors import Detection
from app.identify import CatGallery, InvalidCatNameError
from app.main import create_app
from tests.conftest import make_image
from tests.fakes import FakeDetector, FakeEmbedder

RED, BLUE, GREEN = (200, 30, 30), (30, 30, 200), (30, 200, 30)
ONE_CAT = [Detection("cat", 15, 0.9, (10.0, 10.0, 40.0, 40.0))]
TWO_CATS = ONE_CAT + [Detection("cat", 15, 0.8, (50.0, 10.0, 70.0, 40.0))]
TOKEN = "s3cret"


def photo(color, size=(80, 60)):
    return ("cat.png", make_image(size, color=color), "image/png")


@pytest.fixture
def detector():
    d = FakeDetector()
    d.scene = list(ONE_CAT)
    return d


@pytest.fixture
def embedder():
    return FakeEmbedder()


def make_client(detector, embedder, **settings):
    config = Settings(_env_file=None, warmup=False, admin_token=TOKEN, **settings)
    return TestClient(create_app(config, detector=detector, embedder=embedder))


@pytest.fixture
def client(detector, embedder):
    with make_client(detector, embedder) as c:
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def enroll(client, name, *colors, headers=AUTH):
    files = [("images", photo(c)) for c in colors]
    return client.post(f"/v1/cats/{name}/photos", files=files, headers=headers)


def test_enroll_then_identify(client):
    assert enroll(client, "luna", RED, (190, 40, 30)).json() == {
        "name": "luna",
        "added": 2,
        "duplicates": 0,
        "rejected": [],
        "photos": 2,
    }
    enroll(client, "milo", BLUE)
    assert client.get("/v1/cats").json() == {
        "cats": [{"name": "luna", "photos": 2}, {"name": "milo", "photos": 1}],
        "threshold": 0.8,
        "persistent": False,
    }

    body = client.post("/v1/identify", files={"image": photo(BLUE)}).json()
    assert body["names"] == ["milo"]
    assert body["cats"][0]["name"] == "milo"
    assert body["cats"][0]["similarity"] == pytest.approx(1.0)
    assert body["cats"][0]["box"] == {"x1": 10.0, "y1": 10.0, "x2": 40.0, "y2": 40.0}


def test_unknown_cat_is_not_named(client):
    enroll(client, "luna", RED)
    body = client.post("/v1/identify", files={"image": photo(GREEN)}).json()
    assert body["names"] == []
    assert body["cats"][0]["name"] is None
    assert body["cats"][0]["closest"] == "luna"
    assert body["cats"][0]["similarity"] < 0.8


def test_identify_without_cats_in_image(client, detector):
    enroll(client, "luna", RED)
    detector.scene = []
    assert client.post("/v1/identify", files={"image": photo(RED)}).json()["cats"] == []


def test_identify_before_enrolling(client):
    res = client.post("/v1/identify", files={"image": photo(RED)})
    assert res.status_code == 409
    assert "POST /v1/cats/{name}/photos" in res.json()["detail"]


def test_photos_need_exactly_one_cat(client, detector):
    detector.scene = list(TWO_CATS)
    body = enroll(client, "luna", RED).json()
    assert body["added"] == 0
    assert body["rejected"] == [
        {"file": "cat.png", "reason": "2 cats found; upload photos showing only luna"}
    ]
    detector.scene = []
    assert "no cat found" in enroll(client, "luna", RED).json()["rejected"][0]["reason"]


def test_invalid_and_duplicate_photos(client):
    files = [
        ("images", photo(RED)),
        ("images", photo(RED)),
        ("images", ("x.png", b"not an image", "image/png")),
    ]
    body = client.post("/v1/cats/luna/photos", files=files, headers=AUTH).json()
    assert (body["added"], body["duplicates"], body["photos"]) == (1, 1, 1)
    assert body["rejected"] == [{"file": "x.png", "reason": "Unsupported or corrupt image"}]


def test_admin_token_required(client):
    assert enroll(client, "luna", RED, headers={}).status_code == 401
    assert enroll(client, "luna", RED, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.delete("/v1/cats/luna").status_code == 401


def test_no_admin_token_configured_allows_changes(detector, embedder):
    config = Settings(_env_file=None, warmup=False)
    with TestClient(create_app(config, detector=detector, embedder=embedder)) as c:
        assert enroll(c, "luna", RED, headers={}).json()["added"] == 1


@pytest.mark.parametrize("name", ["Luna", "a" * 33, "-luna", "luna%2F..", "lu na"])
def test_invalid_names(client, name):
    # 404/405: a name with a "/" matches no API route (405 once the web interface is built).
    assert enroll(client, name, RED).status_code in (400, 404, 405)


def test_delete_cat(client):
    enroll(client, "luna", RED)
    assert client.delete("/v1/cats/luna", headers=AUTH).status_code == 204
    assert client.delete("/v1/cats/luna", headers=AUTH).status_code == 404
    assert client.get("/v1/cats").json()["cats"] == []


def test_docs_offer_file_pickers_for_photos(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    images = next(s for name, s in schemas.items() if "add_photos" in name)["properties"]["images"]
    assert images["items"]["format"] == "binary"


def test_too_many_files_per_upload(client):
    res = enroll(client, "luna", *[RED] * 21)
    assert res.status_code == 400


def test_disabled_without_embedder(detector):
    with TestClient(create_app(Settings(_env_file=None, warmup=False), detector=detector)) as c:
        assert c.post("/v1/identify", files={"image": photo(RED)}).status_code == 503
        assert c.get("/v1/cats").status_code == 503
        assert enroll(c, "luna", RED, headers={}).status_code == 503


def test_photos_survive_restart(tmp_path, detector, embedder):
    with make_client(detector, embedder, cats_dir=str(tmp_path)) as c:
        enroll(c, "luna", RED)
        assert c.get("/v1/cats").json()["persistent"] is True
    assert sorted(p.suffix for p in (tmp_path / "luna").iterdir()) == [".jpg", ".npy"]

    calls_before = embedder.calls
    with make_client(detector, embedder, cats_dir=str(tmp_path)) as c:
        assert c.get("/v1/cats").json()["cats"] == [{"name": "luna", "photos": 1}]
        assert c.post("/v1/identify", files={"image": photo(RED)}).json()["names"] == ["luna"]
    assert embedder.calls == calls_before + 1  # cached fingerprints reused; 1 = the query


def test_unwritable_cats_dir_falls_back_to_memory(tmp_path, embedder, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("")
    gallery = CatGallery(embedder, root=blocker / "cats", threshold=0.8)
    assert gallery.root is None
    assert "will be lost on restart" in caplog.text
    assert gallery.add("luna", [Image.new("RGB", (9, 9))]) == 1


def test_cat_photo(tmp_path, detector, embedder):
    with make_client(detector, embedder, cats_dir=str(tmp_path)) as c:
        enroll(c, "luna", RED)
        res = c.get("/v1/cats/luna/photo")
        assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
        assert c.get("/v1/cats/milo/photo").status_code == 404
    with make_client(detector, embedder) as c:  # memory only: nothing stored
        enroll(c, "luna", RED)
        assert c.get("/v1/cats/luna/photo").status_code == 404


def test_fingerprints_recomputed_for_a_new_embedder(tmp_path, embedder):
    CatGallery(embedder, root=tmp_path, threshold=0.8).add("luna", [Image.new("RGB", (9, 9))])
    new_model = FakeEmbedder()
    new_model.signature = "v2"
    gallery = CatGallery(new_model, root=tmp_path, threshold=0.8)
    assert gallery.cats() == {"luna": 1} and new_model.calls == 1
    assert len(list((tmp_path / "luna").glob("*.v2.npy"))) == 1


def test_gallery_limits_and_names(embedder, monkeypatch):
    gallery = CatGallery(embedder, root=None, threshold=0.8)
    with pytest.raises(InvalidCatNameError):
        gallery.add("../etc", [Image.new("RGB", (9, 9))])
    monkeypatch.setattr("app.identify.MAX_PHOTOS_PER_CAT", 1)
    gallery.add("luna", [Image.new("RGB", (9, 9))])
    with pytest.raises(ValueError, match="more than 1 photos"):
        gallery.add("luna", [Image.new("RGB", (9, 9), (1, 2, 3))])


def test_onnx_embedder(tmp_path):
    """Real Embedder on a tiny ONNX model whose fingerprint is the mean colour."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    from app.identify import Embedder

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["pooled"], axes=[2, 3], keepdims=0)],
        "mean",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["batch", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    model.metadata_props.add(key="imgsz", value="32")
    path = tmp_path / "embedder.onnx"
    onnx.save(model, path)

    embedder = Embedder(str(path), threads=1)
    vectors = embedder.embed([Image.new("RGB", (40, 40), RED), Image.new("RGB", (64, 32), BLUE)])
    assert vectors.shape == (2, 3)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1, 1])
    assert vectors[0].argmax() == 0 and vectors[1].argmax() == 2  # red, then blue
    assert embedder.embed([]).shape[0] == 0
