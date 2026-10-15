"""Smoke test of the YOLO12 backend with a randomly initialised model (no downloads)."""

from __future__ import annotations

import pytest
from PIL import Image

from app.detectors import DetectorConfig, create_detector

IMAGE = Image.new("RGB", (128, 96), (30, 60, 90))


def test_ultralytics_backend_from_architecture_yaml():
    pytest.importorskip("ultralytics")
    detector = create_detector("ultralytics", DetectorConfig(weights="yolo12n.yaml", device="cpu"))
    assert len(detector.labels) == 80  # untrained yaml models have numeric names
    detections = detector.detect(IMAGE, confidence=0.0, class_ids={15})
    assert all(d.class_id == 15 for d in detections)
