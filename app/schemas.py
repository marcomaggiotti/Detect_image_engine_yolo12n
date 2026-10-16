from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Bounding box in pixel coordinates of the uploaded image."""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    label: str
    class_id: int
    confidence: float
    box: Box


class ImageInfo(BaseModel):
    width: int
    height: int


class ModelRef(BaseModel):
    backend: str
    weights: str
    version: str | None = None


class DetectResponse(BaseModel):
    model: ModelRef
    image: ImageInfo
    classes: list[str] | None = Field(
        description="Classes that were searched for; null means all model classes."
    )
    found: bool = Field(description="True when at least one requested object was detected.")
    counts: dict[str, int]
    detections: list[DetectionOut]
    inference_ms: float


class CatResponse(BaseModel):
    model: ModelRef
    cat: bool = Field(description="True when at least one cat was detected.")
    count: int = Field(description="Number of cats detected.")
    confidence: float | None = Field(
        description="Score of the most confident cat; null when there is none."
    )
    inference_ms: float


class ModelInfo(ModelRef):
    arch: str | None = None
    default_classes: list[str] | None
    confidence_threshold: float
    labels: dict[int, str]


class ErrorResponse(BaseModel):
    detail: str


class IdentifiedCat(BaseModel):
    name: str | None = Field(
        description="Which of your cats this is; null when it doesn't look enough like any of them."
    )
    closest: str = Field(description="The most similar of your cats, even below the threshold.")
    similarity: float = Field(description="Similarity to the closest cat (0-1).")
    confidence: float = Field(description="Detection score of this cat.")
    box: Box


class IdentifyResponse(BaseModel):
    model: ModelRef
    image: ImageInfo
    names: list[str] = Field(description="Your cats found in the image.")
    cats: list[IdentifiedCat]
    inference_ms: float


class KnownCat(BaseModel):
    name: str
    photos: int


class CatsResponse(BaseModel):
    cats: list[KnownCat]
    threshold: float = Field(description="Minimum similarity to name a cat.")
    persistent: bool = Field(description="False when photos are lost on restart.")


class RejectedPhoto(BaseModel):
    file: str
    reason: str


class EnrollResponse(BaseModel):
    name: str
    added: int
    duplicates: int = Field(description="Photos already known for this cat, skipped.")
    rejected: list[RejectedPhoto]
    photos: int = Field(description="Photos of this cat in total.")
