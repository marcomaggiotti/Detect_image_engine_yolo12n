from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import make_image
from tests.fakes import FakeDetector


def post(client, data: bytes, query: str = ""):
    return client.post(f"/v1/detect{query}", files={"image": ("img.png", data, "image/png")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/v1/model").json()
    assert body["backend"] == "fake"
    assert body["weights"] == "fake.pt"
    assert body["default_classes"] == ["cat"]
    assert body["labels"]["15"] == "cat"


def test_detects_cats_by_default(client, png, detector):
    res = post(client, png)
    assert res.status_code == 200
    body = res.json()
    assert body["classes"] == ["cat"]
    assert body["found"] is True
    assert body["counts"] == {"cat": 2}
    assert body["image"] == {"width": 80, "height": 60}
    assert body["detections"][0] == {
        "label": "cat",
        "class_id": 15,
        "confidence": 0.88,
        "box": {"x1": 10.0, "y1": 10.0, "x2": 40.0, "y2": 40.0},
    }
    assert detector.calls[-1]["class_ids"] == frozenset({15})
    assert detector.calls[-1]["confidence"] == 0.25


def test_cat_endpoint_finds_cats(client, png, detector):
    res = client.post("/v1/cat", files={"image": ("img.png", png, "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["cat"] is True
    assert body["count"] == 2
    assert body["confidence"] == 0.88
    assert body["model"]["weights"] == "fake.pt"
    assert detector.calls[-1]["class_ids"] == frozenset({15})


def test_cat_endpoint_no_cat(client, png):
    res = client.post("/v1/cat?confidence=0.95", files={"image": ("img.png", png, "image/png")})
    assert res.json() | {"inference_ms": 0} == {
        "model": {"backend": "fake", "weights": "fake.pt", "version": None},
        "cat": False,
        "count": 0,
        "confidence": None,
        "inference_ms": 0,
    }


def test_cat_endpoint_ignores_configured_classes(detector, png):
    settings = Settings(_env_file=None, warmup=False, target_classes="person")
    with TestClient(create_app(settings, detector=detector)) as c:
        body = c.post("/v1/cat", files={"image": ("img.png", png, "image/png")}).json()
    assert body["count"] == 2


def test_cat_endpoint_invalid_image(client):
    res = client.post("/v1/cat", files={"image": ("img.png", b"nope", "image/png")})
    assert res.status_code == 400


@pytest.mark.parametrize("query", ["?classes=cat,person", "?classes=cat&classes=person"])
def test_request_can_choose_classes(client, png, query):
    body = post(client, png, query).json()
    assert body["classes"] == ["cat", "person"]
    assert body["counts"] == {"person": 1, "cat": 2}


def test_class_names_are_case_insensitive(client, png):
    assert post(client, png, "?classes=Person").json()["counts"] == {"person": 1}


def test_star_returns_all_classes(client, png):
    body = post(client, png, "?classes=*&confidence=0.1").json()
    assert body["classes"] is None
    assert body["counts"] == {"person": 1, "cat": 2, "dog": 1}


def test_confidence_override(client, png):
    assert post(client, png, "?confidence=0.5").json()["counts"] == {"cat": 1}


def test_not_found(client, png):
    body = post(client, png, "?classes=chair").json()
    assert body["found"] is False
    assert body["detections"] == []


def test_unknown_class_is_rejected(client, png):
    res = post(client, png, "?classes=cat,unicorn")
    assert res.status_code == 400
    assert "unicorn" in res.json()["detail"]


def test_invalid_confidence(client, png):
    assert post(client, png, "?confidence=1.5").status_code == 422


def test_invalid_image(client):
    res = post(client, b"definitely not an image")
    assert res.status_code == 400


def test_upload_too_large(detector):
    settings = Settings(_env_file=None, warmup=False, max_upload_bytes=100)
    with TestClient(create_app(settings, detector=detector)) as c:
        assert post(c, make_image()).status_code == 413


def test_configured_target_classes(detector, png):
    settings = Settings(_env_file=None, warmup=False, target_classes="person, dog")
    with TestClient(create_app(settings, detector=detector)) as c:
        body = post(c, png, "?confidence=0.1").json()
    assert body["classes"] == ["person", "dog"]
    assert body["counts"] == {"person": 1, "dog": 1}


def test_startup_fails_if_model_lacks_target_class():
    settings = Settings(_env_file=None, warmup=False, target_classes="giraffe")
    with pytest.raises(ValueError, match="giraffe"):
        with TestClient(create_app(settings, detector=FakeDetector())):
            pass


def test_custom_backend_loaded_from_settings(png):
    settings = Settings(_env_file=None, backend="tests.fakes:FakeDetector")
    with TestClient(create_app(settings)) as c:
        assert c.get("/v1/model").json()["backend"] == "fake"
        assert post(c, png).json()["counts"] == {"cat": 2}


APP = "https://image-detector-manager.onrender.com"


def test_cors_allows_configured_web_app(detector):
    settings = Settings(_env_file=None, warmup=False, cors_origins=f"{APP}, http://localhost:5173")
    with TestClient(create_app(settings, detector=detector)) as c:
        preflight = c.options(
            "/v1/cats/luna/photos",
            headers={
                "Origin": APP,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == APP
        assert "authorization" in preflight.headers["access-control-allow-headers"].lower()
        res = c.get("/health", headers={"Origin": APP})
        assert res.headers["access-control-allow-origin"] == APP
        other = c.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers


def test_no_cors_by_default(client):
    res = client.get("/health", headers={"Origin": APP})
    assert "access-control-allow-origin" not in res.headers


def test_cors_origins_from_env(monkeypatch):
    monkeypatch.setenv("DETECT_CORS_ORIGINS", f"{APP} , http://localhost:5173")
    assert Settings(_env_file=None).cors_origins == [APP, "http://localhost:5173"]
