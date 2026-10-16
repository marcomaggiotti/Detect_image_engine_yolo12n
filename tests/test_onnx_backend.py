"""ONNX Runtime backend tests on tiny hand-built graphs with known outputs."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from PIL import Image

from app.detectors import DetectorConfig, create_detector
from app.detectors.onnxruntime_backend import _nms, _resize_bilinear

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

NAMES = "{0: 'cat', 1: 'dog'}"


def make_model(path, output, *, input_shape=(1, 3, 64, 64), metadata=None):
    """A model that ignores its pixels and always returns ``output``."""
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["total"], keepdims=0),
        helper.make_node("Mul", ["total", "zero"], ["nothing"]),
        helper.make_node("Add", ["preds", "nothing"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fixed",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(output.shape))],
        initializer=[
            numpy_helper.from_array(np.float32(0), "zero"),
            numpy_helper.from_array(output.astype(np.float32), "preds"),
        ],
    )
    # IR version 9 loads in every onnxruntime >= 1.17.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    meta = {"task": "detect", "stride": "32", "names": NAMES} | (metadata or {})
    for key, value in meta.items():
        if value is not None:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, path)
    return str(path)


def raw_output(*rows):
    """Raw YOLO head (1, 4 + classes, anchors) from (cx, cy, w, h, cat, dog) rows."""
    return np.array(rows, dtype=np.float32).T[None]


RAW = raw_output(
    (16, 32, 20, 20, 0.90, 0.01),  # cat
    (17, 33, 20, 20, 0.80, 0.02),  # same cat, lower score: removed by NMS
    (48, 32, 10, 10, 0.05, 0.60),  # dog
    (40, 40, 4, 4, 0.10, 0.00),  # below threshold
)
# 128x64 image -> scaled by 0.5 to 64x32, padded 16 px top and bottom.
IMAGE = Image.new("RGB", (128, 64))


def detector(weights, **config):
    return create_detector("onnxruntime", DetectorConfig(weights=weights, device="cpu", **config))


def test_raw_output_letterbox_and_nms(tmp_path):
    d = detector(make_model(tmp_path / "m.onnx", RAW))
    assert d.labels == {0: "cat", 1: "dog"}
    detections = d.detect(IMAGE, confidence=0.25)
    assert [(x.label, x.class_id, round(x.confidence, 2)) for x in detections] == [
        ("cat", 0, 0.9),
        ("dog", 1, 0.6),
    ]
    assert detections[0].box == pytest.approx((12.0, 12.0, 52.0, 52.0))
    assert detections[1].box == pytest.approx((86.0, 22.0, 106.0, 42.0))


def test_class_filter_and_confidence(tmp_path):
    d = detector(make_model(tmp_path / "m.onnx", RAW))
    assert [x.label for x in d.detect(IMAGE, confidence=0.25, class_ids={1})] == ["dog"]
    assert d.detect(IMAGE, confidence=0.95) == []


def test_max_detections(tmp_path):
    d = detector(make_model(tmp_path / "m.onnx", RAW), max_detections=1)
    assert [x.label for x in d.detect(IMAGE, confidence=0.25)] == ["cat"]


def test_end_to_end_output(tmp_path):
    # (1, max_det, 6): x1, y1, x2, y2, score, class; zero rows are padding.
    output = np.array([[[6, 22, 26, 42, 0.9, 0], [38, 22, 58, 42, 0.7, 1], [0] * 6]])
    d = detector(make_model(tmp_path / "m.onnx", output))
    detections = d.detect(IMAGE, confidence=0.25)
    assert [x.label for x in detections] == ["cat", "dog"]
    assert detections[0].box == pytest.approx((12.0, 12.0, 52.0, 52.0))


def test_dynamic_input_uses_configured_size(tmp_path):
    weights = make_model(tmp_path / "m.onnx", RAW, input_shape=(1, 3, "height", "width"))
    d = detector(weights, image_size=100)
    assert d.dynamic and d.input_size == (128, 128)  # rounded up to the stride
    tensor, scale, pad = d._preprocess(Image.new("RGB", (256, 100)))
    assert tensor.shape == (1, 3, 64, 128)  # only padded to the next multiple of 32
    assert scale == (0.5, 0.5) and pad == (0, 7)


def test_fixed_input_warns_when_size_differs(tmp_path, caplog):
    weights = make_model(tmp_path / "m.onnx", RAW)
    with caplog.at_level(logging.WARNING):
        d = detector(weights, image_size=320)
    assert d.input_size == (64, 64)
    assert "image_size=320 is ignored" in caplog.text


def test_labels_file_overrides_metadata(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("kitten\npuppy\n")
    d = detector(make_model(tmp_path / "m.onnx", RAW), labels_file=str(labels))
    assert d.labels == {0: "kitten", 1: "puppy"}


def test_rejects_models_without_names_or_wrong_task(tmp_path):
    with pytest.raises(ValueError, match="DETECT_MODEL_LABELS"):
        detector(make_model(tmp_path / "a.onnx", RAW, metadata={"names": None}))
    weights = make_model(tmp_path / "b.onnx", RAW, metadata={"task": "segment"})
    with pytest.raises(ValueError, match="detection model"):
        detector(weights)
    with pytest.raises(ValueError, match="YOLO models"):
        detector(weights, arch="rtdetr")


def test_service_runs_on_onnxruntime(tmp_path):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    from tests.conftest import make_image

    settings = Settings(
        _env_file=None,
        backend="onnxruntime",
        model_weights=make_model(tmp_path / "m.onnx", RAW),
        threads=1,
    )
    with TestClient(create_app(settings)) as client:
        res = client.post("/v1/cat", files={"image": ("i.png", make_image((128, 64)), "image/png")})
    assert res.json()["cat"] is True
    assert res.json()["model"]["backend"] == "onnxruntime"


def test_nms_suppresses_overlaps_only():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    assert _nms(boxes, scores, 0.45).tolist() == [0, 2]
    assert _nms(boxes, scores, 0.95).tolist() == [0, 1, 2]


def test_resize_matches_opencv_bilinear():
    pixels = np.array([[[0] * 3, [100] * 3], [[200] * 3, [50] * 3]], dtype=np.uint8)
    up = _resize_bilinear(pixels, 4, 4)[..., 0]
    # cv2.resize(pixels, (4, 4), interpolation=cv2.INTER_LINEAR); OpenCV's
    # fixed-point arithmetic can round one level differently.
    opencv = [[0, 25, 75, 100], [50, 59, 78, 88], [150, 128, 84, 63], [200, 162, 87, 50]]
    assert np.abs(up - opencv).max() <= 1
    assert _resize_bilinear(pixels, 2, 2).tolist() == pixels.astype(float).tolist()
