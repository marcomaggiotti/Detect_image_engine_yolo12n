"""ONNX Runtime backend: serve YOLO models exported to ONNX, without PyTorch.

Only ``onnxruntime``, NumPy and Pillow are needed at runtime, which keeps the
service small enough for 512 MB instances. Export a model with
``python -m training.export`` (or any Ultralytics ONNX export of a YOLOv8 to
YOLO12 detection model). Class names, stride and input size are read from the
metadata Ultralytics stores in the file.

Two output layouts are understood:

* raw head ``(1, 4 + classes, anchors)``: boxes as centre/width/height, NMS is
  done here;
* end-to-end ``(1, max_det, 6)``: ``x1, y1, x2, y2, score, class`` from a model
  exported with ``nms=True``.

Export with ``dynamic=True`` to choose ``DETECT_IMAGE_SIZE`` at runtime; a
fixed-size export always runs at the size it was exported with.
"""

from __future__ import annotations

import ast
import logging
import math
from collections.abc import Collection

import numpy as np
from PIL import Image

from app.detectors.base import Detection, Detector, DetectorConfig, load_labels
from app.imaging import resize_bilinear

logger = logging.getLogger(__name__)

_PAD_VALUE = 114.0  # letterbox grey used by Ultralytics
_CLASS_OFFSET = 7680.0  # shifts boxes per class so one NMS pass is class-aware
_MAX_NMS_CANDIDATES = 30_000


class OnnxRuntimeDetector(Detector):
    backend_name = "onnxruntime"

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        if (config.arch or "yolo").lower() != "yolo":
            raise ValueError(f"The onnxruntime backend serves YOLO models, not {config.arch!r}")
        import onnxruntime as ort

        options = ort.SessionOptions()
        if config.threads:
            options.intra_op_num_threads = config.threads
            options.inter_op_num_threads = 1
        providers = list(config.providers or _providers_for(config.device, ort))
        self._session = ort.InferenceSession(config.weights, options, providers=providers)

        meta = self._session.get_modelmeta().custom_metadata_map
        task = meta.get("task", "detect")
        if task != "detect":
            raise ValueError(
                f"{config.weights!r} is a {task!r} model; a detection model is required"
            )

        if config.labels_file:
            self._labels = load_labels(config.labels_file)
        elif "names" in meta:
            names = ast.literal_eval(meta["names"])
            self._labels = {int(k): str(v) for k, v in names.items()}
        else:
            raise ValueError(f"{config.weights!r} has no class names; set DETECT_MODEL_LABELS")

        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._dtype = np.float16 if "float16" in model_input.type else np.float32
        self._stride = int(meta.get("stride", 32))
        height, width = model_input.shape[2:4]
        self.dynamic = not (isinstance(height, int) and isinstance(width, int))
        if self.dynamic:
            size = math.ceil(config.image_size / self._stride) * self._stride
            self.input_size = (size, size)
        else:
            self.input_size = (height, width)
            if self.input_size != (config.image_size, config.image_size):
                logger.warning(
                    "%s has a fixed %dx%d input, so image_size=%d is ignored. "
                    "Export it with --dynamic to choose the size at runtime.",
                    config.weights,
                    width,
                    height,
                    config.image_size,
                )
        self._end2end = self._session.get_outputs()[0].shape[-1] == 6
        logger.info(
            "ONNX Runtime %s with %s, input %dx%d (%s)",
            ort.__version__,
            self._session.get_providers()[0],
            self.input_size[1],
            self.input_size[0],
            "dynamic" if self.dynamic else "fixed",
        )

    @property
    def labels(self) -> dict[int, str]:
        return self._labels

    def detect(
        self,
        image: Image.Image,
        *,
        confidence: float,
        class_ids: Collection[int] | None = None,
    ) -> list[Detection]:
        tensor, (scale_x, scale_y), (pad_x, pad_y) = self._preprocess(image)
        output = self._session.run(None, {self._input_name: tensor})[0][0].astype(np.float32)

        if self._end2end:
            boxes, scores, classes = output[:, :4], output[:, 4], output[:, 5].astype(int)
        else:
            preds = output.T  # (anchors, 4 + classes)
            classes = preds[:, 4:].argmax(axis=1)
            scores = preds[np.arange(len(preds)), 4 + classes]
            boxes = _xywh_to_xyxy(preds[:, :4])

        keep = scores >= confidence
        if class_ids is not None:
            keep &= np.isin(classes, list(class_ids))
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        order = scores.argsort()[::-1]
        if not self._end2end:
            order = order[:_MAX_NMS_CANDIDATES]
            shifted = boxes[order] + classes[order, None] * _CLASS_OFFSET
            order = order[_nms(shifted, scores[order], self.config.iou_threshold)]
        order = order[: self.config.max_detections]

        width, height = image.size
        boxes = (boxes[order] - [pad_x, pad_y, pad_x, pad_y]) / [scale_x, scale_y, scale_x, scale_y]
        boxes = boxes.clip(0, [width, height, width, height])
        return [
            Detection(
                label=self._labels.get(int(cls), str(int(cls))),
                class_id=int(cls),
                confidence=float(score),
                box=tuple(float(v) for v in box),
            )
            for box, score, cls in zip(boxes, scores[order], classes[order], strict=True)
        ]

    def _preprocess(
        self, image: Image.Image
    ) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
        """Letterbox to the model input: resize keeping aspect ratio, then pad."""
        width, height = image.size
        in_height, in_width = self.input_size
        scale = min(in_width / width, in_height / height)
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        if self.dynamic:
            # Pad only to the next stride multiple: fewer pixels, faster inference.
            in_width = math.ceil(new_width / self._stride) * self._stride
            in_height = math.ceil(new_height / self._stride) * self._stride

        pixels = resize_bilinear(np.asarray(image), new_width, new_height)
        pad_x, pad_y = (in_width - new_width) // 2, (in_height - new_height) // 2
        canvas = np.full((in_height, in_width, 3), _PAD_VALUE, dtype=np.float32)
        canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = pixels

        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None] / 255.0, dtype=self._dtype)
        return tensor, (new_width / width, new_height / height), (pad_x, pad_y)


def _providers_for(device: str | None, ort) -> list:
    available = ort.get_available_providers()
    if device and device.startswith("cuda"):
        if "CUDAExecutionProvider" in available:
            index = int(device.partition(":")[2] or 0)
            return [("CUDAExecutionProvider", {"device_id": index}), "CPUExecutionProvider"]
        logger.warning("CUDAExecutionProvider unavailable (install onnxruntime-gpu); using CPU")
    elif device == "mps" and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    centre, half = xywh[:, :2], xywh[:, 2:] / 2
    return np.concatenate([centre - half, centre + half], axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, best first."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(best)
        top_left = np.maximum(boxes[best, :2], boxes[rest, :2])
        bottom_right = np.minimum(boxes[best, 2:], boxes[rest, 2:])
        inter = np.prod((bottom_right - top_left).clip(0), axis=1)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=int)
