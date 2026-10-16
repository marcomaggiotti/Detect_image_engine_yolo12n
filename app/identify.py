"""Tell your own cats apart, without training.

A small ImageNet-pretrained network (exported by
``training/export_embedder.py``) turns a cropped cat into an appearance
fingerprint (an embedding vector). You upload a few photos of each of your
cats; their fingerprints are kept in a :class:`CatGallery`. A detected cat is
then given the name of the cat whose photos it resembles most, or none when
it doesn't resemble any of them enough.

Adding photos is all the "training" there is: they take effect immediately.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.imaging import resize_bilinear

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_PHOTOS_PER_CAT = 200
_PAD_VALUE = 114.0
_CROP_MARGIN = 0.1  # context kept around the detected box, as a fraction of its size


class InvalidCatNameError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"Invalid cat name {name!r}: use 1-32 lowercase letters, digits, '-' or '_'"
        )


@dataclass(frozen=True)
class Match:
    """Closest known cat; ``name`` is None when the similarity is below the threshold."""

    name: str | None
    closest: str
    similarity: float


class Embedder:
    """ONNX Runtime model mapping cat crops to L2-normalised fingerprints."""

    def __init__(
        self,
        weights: str,
        *,
        threads: int | None = None,
        providers: tuple[str, ...] | None = None,
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            weights, options, providers=list(providers or ["CPUExecutionProvider"])
        )
        meta = self._session.get_modelmeta().custom_metadata_map
        self._input_name = self._session.get_inputs()[0].name
        self.size = int(meta.get("imgsz", 224))
        # Identifies the model, so cached fingerprints are recomputed when it changes.
        self.signature = hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:12]

    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        """Fingerprints of shape (len(crops), dim), averaged with mirrored crops."""
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        batch = np.stack([self._prepare(crop) for crop in crops])
        batch = np.concatenate([batch, batch[..., ::-1]])  # mirrored copies
        vectors = self._session.run(None, {self._input_name: np.ascontiguousarray(batch)})[0]
        vectors = vectors[: len(crops)] + vectors[len(crops) :]
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)

    def _prepare(self, crop: Image.Image) -> np.ndarray:
        """Pad to a square (keeps the whole cat), then resize to the input size."""
        pixels = np.asarray(crop.convert("RGB"))
        height, width = pixels.shape[:2]
        side = max(height, width)
        square = np.full((side, side, 3), _PAD_VALUE, dtype=np.float32)
        top, left = (side - height) // 2, (side - width) // 2
        square[top : top + height, left : left + width] = pixels
        resized = resize_bilinear(square.astype(np.uint8), self.size, self.size)
        return (resized / 255.0).astype(np.float32).transpose(2, 0, 1)


def crop_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Crop a detection, with a little context around it."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * _CROP_MARGIN, (y2 - y1) * _CROP_MARGIN
    return image.crop(
        (
            max(0, round(x1 - dx)),
            max(0, round(y1 - dy)),
            min(image.width, round(x2 + dx)),
            min(image.height, round(y2 + dy)),
        )
    )


class CatGallery:
    """Photos (cat crops) of each known cat and their fingerprints.

    With a ``root`` directory, crops are stored as ``<root>/<name>/<id>.jpg``
    with their fingerprint cached next to them, and reloaded at startup.
    Without one, everything is kept in memory and lost on restart.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        root: str | Path | None,
        threshold: float,
        top_k: int = 3,
    ) -> None:
        self.embedder = embedder
        self.root = Path(root) if root else None
        self.threshold = threshold
        self.top_k = top_k
        self._vectors: dict[str, dict[str, np.ndarray]] = {}  # name -> photo id -> vector
        self._lock = threading.Lock()
        if self.root:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                probe = self.root / ".write-test"
                probe.write_bytes(b"")
                probe.unlink()
            except OSError as exc:
                logger.error(
                    "Cannot write to %s (%s): photos of your cats will be lost on restart",
                    self.root,
                    exc,
                )
                self.root = None
        if self.root:
            self._load()

    def cats(self) -> dict[str, int]:
        """Known cats and how many photos each has."""
        with self._lock:
            return {name: len(photos) for name, photos in sorted(self._vectors.items())}

    def add(self, name: str, crops: list[Image.Image]) -> int:
        """Add photos of a cat; returns how many were new (duplicates are skipped)."""
        validate_name(name)
        with self._lock:
            photos = self._vectors.get(name, {})
            new = {}
            for crop in crops:
                data = _jpeg(crop)
                photo_id = hashlib.sha256(data).hexdigest()[:16]
                if photo_id not in photos and photo_id not in new:
                    new[photo_id] = (crop, data)
            room = MAX_PHOTOS_PER_CAT - len(photos)
            if len(new) > room:
                raise ValueError(
                    f"{name} would have more than {MAX_PHOTOS_PER_CAT} photos; "
                    f"there is room for {max(room, 0)} more"
                )
            if not new:
                return 0
            vectors = self.embedder.embed([crop for crop, _ in new.values()])
            for (photo_id, (_, data)), vector in zip(new.items(), vectors, strict=True):
                if self.root:
                    folder = self.root / name
                    folder.mkdir(exist_ok=True)
                    (folder / f"{photo_id}.jpg").write_bytes(data)
                    np.save(folder / f"{photo_id}.{self.embedder.signature}.npy", vector)
                photos[photo_id] = vector
            self._vectors[name] = photos
            return len(new)

    def photo(self, name: str) -> Path | None:
        """One stored photo of a cat (to show it), or None (unknown cat, memory-only)."""
        validate_name(name)
        with self._lock:
            if name not in self._vectors or not self.root:
                return None
            return next(iter(sorted((self.root / name).glob("*.jpg"))), None)

    def remove(self, name: str) -> bool:
        validate_name(name)
        with self._lock:
            if self._vectors.pop(name, None) is None:
                return False
            if self.root:
                shutil.rmtree(self.root / name, ignore_errors=True)
            return True

    def identify(self, vectors: np.ndarray) -> list[Match]:
        """Match fingerprints against the known cats (the gallery must not be empty)."""
        with self._lock:
            names = list(self._vectors)
            galleries = [np.stack(list(self._vectors[n].values())) for n in names]
        if not names:
            raise LookupError("No cats enrolled")
        matches = []
        for vector in vectors:
            # Score of a cat: mean of its top-k most similar photos (robust to one odd photo).
            scores = [
                float(np.sort(gallery @ vector)[-self.top_k :].mean()) for gallery in galleries
            ]
            best = int(np.argmax(scores))
            similarity = round(scores[best], 4)
            name = names[best] if similarity >= self.threshold else None
            matches.append(Match(name=name, closest=names[best], similarity=similarity))
        return matches

    def _load(self) -> None:
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if not NAME_PATTERN.match(folder.name):
                continue
            photos, missing = {}, []
            for jpg in sorted(folder.glob("*.jpg")):
                cached = folder / f"{jpg.stem}.{self.embedder.signature}.npy"
                if cached.exists():
                    photos[jpg.stem] = np.load(cached)
                else:
                    missing.append(jpg)
            if missing:  # new embedder: compute and cache the fingerprints once
                crops = [Image.open(path).convert("RGB") for path in missing]
                for path, vector in zip(missing, self.embedder.embed(crops), strict=True):
                    np.save(folder / f"{path.stem}.{self.embedder.signature}.npy", vector)
                    photos[path.stem] = vector
            if photos:
                self._vectors[folder.name] = photos
        logger.info(
            "Loaded %d known cat(s) from %s: %s", len(self._vectors), self.root, self.cats()
        )


def validate_name(name: str) -> None:
    if not NAME_PATTERN.match(name):
        raise InvalidCatNameError(name)


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
