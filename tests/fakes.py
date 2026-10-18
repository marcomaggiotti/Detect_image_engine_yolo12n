from __future__ import annotations

from collections.abc import Collection

import numpy as np
from PIL import Image

from app.detectors import Detection, Detector, DetectorConfig


class FakeDetector(Detector):
    """Returns a fixed scene: one person, two cats and a low-confidence dog."""

    backend_name = "fake"
    SCENE = [
        Detection("person", 0, 0.92, (1.0, 2.0, 30.0, 60.0)),
        Detection("cat", 15, 0.88, (10.0, 10.0, 40.0, 40.0)),
        Detection("cat", 15, 0.41, (50.0, 5.0, 70.0, 25.0)),
        Detection("dog", 16, 0.20, (0.0, 0.0, 20.0, 20.0)),
    ]

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config or DetectorConfig(weights="fake.pt"))
        self.calls: list[dict] = []
        self.scene = list(self.SCENE)

    @property
    def labels(self) -> dict[int, str]:
        return {0: "person", 15: "cat", 16: "dog", 56: "chair"}

    def detect(
        self,
        image: Image.Image,
        *,
        confidence: float,
        class_ids: Collection[int] | None = None,
    ) -> list[Detection]:
        self.calls.append({"size": image.size, "confidence": confidence, "class_ids": class_ids})
        return [
            d
            for d in self.scene
            if d.confidence >= confidence and (class_ids is None or d.class_id in class_ids)
        ]


class FakeEmbedder:
    """Fingerprint = average colour, so differently coloured images are different cats."""

    signature = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        self.calls += 1
        vectors = np.array([np.asarray(c.convert("RGB"), float).mean((0, 1)) for c in crops])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class NotADetector:
    pass
