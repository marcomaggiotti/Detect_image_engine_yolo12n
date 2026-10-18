"""Detection use case: decode, choose classes, run the model, summarise."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from PIL import Image

from app.detectors import Detection, Detector
from app.identify import CatGallery, Match, crop_box, validate_name
from app.imaging import InvalidImageError, decode_image

ALL_CLASSES = "*"
CAT = "cat"


class IdentificationDisabledError(RuntimeError):
    pass


class UnknownClassError(ValueError):
    def __init__(self, unknown: list[str]) -> None:
        super().__init__(f"Unknown class(es) for this model: {', '.join(unknown)}")
        self.unknown = unknown


@dataclass(frozen=True)
class ClassSelection:
    """Class ids to report; ``ids is None`` means every class."""

    names: list[str] | None
    ids: frozenset[int] | None


@dataclass
class DetectionResult:
    width: int
    height: int
    classes: list[str] | None
    detections: list[Detection]
    inference_ms: float
    counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.counts = dict(Counter(d.label for d in self.detections))


@dataclass
class IdentifyResult:
    width: int
    height: int
    cats: list[tuple[Detection, Match]]
    inference_ms: float


@dataclass
class EnrollResult:
    name: str
    added: int
    duplicates: int
    rejected: list[tuple[int, str]]  # (index of the upload, reason)
    photos: int


class DetectionService:
    def __init__(
        self,
        detector: Detector,
        *,
        default_classes: Iterable[str],
        confidence_threshold: float,
        max_image_pixels: int,
        gallery: CatGallery | None = None,
    ) -> None:
        self.detector = detector
        self.gallery = gallery
        self.confidence_threshold = confidence_threshold
        self.max_image_pixels = max_image_pixels
        self._ids_by_name: dict[str, set[int]] = {}
        for class_id, name in detector.labels.items():
            self._ids_by_name.setdefault(name.lower(), set()).add(class_id)
        # Validate at startup: a model that cannot detect the configured
        # target classes should fail fast, not on the first request.
        self.default_selection = self.select_classes(list(default_classes))
        # Most backends are not safe to call from several threads at once.
        self._lock = threading.Lock()

    def select_classes(self, names: list[str] | None) -> ClassSelection:
        if names is None:
            return self.default_selection
        if not names or ALL_CLASSES in names:
            return ClassSelection(names=None, ids=None)
        unknown = [n for n in names if n.lower() not in self._ids_by_name]
        if unknown:
            raise UnknownClassError(unknown)
        ids = frozenset().union(*(self._ids_by_name[n.lower()] for n in names))
        return ClassSelection(names=list(dict.fromkeys(names)), ids=ids)

    def detect(
        self,
        data: bytes,
        *,
        classes: list[str] | None = None,
        confidence: float | None = None,
    ) -> DetectionResult:
        selection = self.select_classes(classes)
        image = decode_image(data, max_pixels=self.max_image_pixels)
        threshold = self.confidence_threshold if confidence is None else confidence

        start = time.perf_counter()
        with self._lock:
            detections = self.detector.detect(image, confidence=threshold, class_ids=selection.ids)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            width=image.width,
            height=image.height,
            classes=selection.names,
            detections=detections,
            inference_ms=round(elapsed_ms, 2),
        )

    def identify(self, data: bytes, *, confidence: float | None = None) -> IdentifyResult:
        """Find the cats in an image and tell which of the known cats each one is."""
        gallery = self._require_gallery()
        if not gallery.cats():
            raise LookupError("No cats enrolled yet")
        image = decode_image(data, max_pixels=self.max_image_pixels)
        start = time.perf_counter()
        with self._lock:
            cats = self._find_cats(image, confidence)
            crops = [crop_box(image, d.box) for d in cats]
            matches = gallery.identify(gallery.embedder.embed(crops)) if crops else []
        elapsed_ms = (time.perf_counter() - start) * 1000
        return IdentifyResult(
            width=image.width,
            height=image.height,
            cats=list(zip(cats, matches, strict=True)),
            inference_ms=round(elapsed_ms, 2),
        )

    def enroll(self, name: str, images: list[bytes]) -> EnrollResult:
        """Add photos of one of your cats. Each photo must show exactly one cat."""
        gallery = self._require_gallery()
        validate_name(name)
        crops, rejected = [], []
        for index, data in enumerate(images):
            try:
                image = decode_image(data, max_pixels=self.max_image_pixels)
            except InvalidImageError as exc:
                rejected.append((index, str(exc)))
                continue
            with self._lock:
                cats = self._find_cats(image, None)
            if len(cats) != 1:
                found = "no cat" if not cats else f"{len(cats)} cats"
                rejected.append((index, f"{found} found; upload photos showing only {name}"))
                continue
            crops.append(crop_box(image, cats[0].box))
        with self._lock:
            added = gallery.add(name, crops) if crops else 0
        return EnrollResult(
            name=name,
            added=added,
            duplicates=len(crops) - added,
            rejected=rejected,
            photos=gallery.cats().get(name, 0),
        )

    def _find_cats(self, image: Image.Image, confidence: float | None) -> list[Detection]:
        ids = self.select_classes([CAT]).ids
        threshold = self.confidence_threshold if confidence is None else confidence
        return self.detector.detect(image, confidence=threshold, class_ids=ids)

    def _require_gallery(self) -> CatGallery:
        if self.gallery is None:
            raise IdentificationDisabledError(
                "Telling cats apart is not configured: set DETECT_EMBEDDER_WEIGHTS"
            )
        return self.gallery
