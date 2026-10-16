"""Service configuration, read from environment variables prefixed with ``DETECT_``."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DETECT_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Model selection -------------------------------------------------
    # Built-in name ("ultralytics", "onnxruntime") or a custom detector given
    # as "package.module:ClassName". "onnxruntime" serves an exported .onnx
    # model without PyTorch (see training/export.py); the Docker image uses it.
    backend: str = "ultralytics"
    # Weights to load. ultralytics: a .pt/.onnx/... file or a known model name.
    # onnxruntime: a .onnx file exported by Ultralytics.
    # Defaults to YOLO12 nano pretrained on COCO (80 classes, "cat" included).
    # Point it at a fine-tuned "best.pt" (or its ONNX export) to serve your own model.
    model_weights: str = "yolo12n.pt"
    # Architecture hint: "yolo" (default) or "rtdetr".
    model_arch: str | None = None
    # Optional label file (one class name per line, line N = class id N),
    # overriding the names stored in the weights.
    model_labels: str | None = None
    # Free-form tag echoed in responses to trace which model answered.
    model_version: str | None = None
    # "cpu", "cuda", "cuda:0", "mps"... None lets the backend decide.
    device: str | None = None
    # CPU threads used by one inference. None lets the runtime decide (all
    # cores of the host, which oversubscribes small cloud instances).
    threads: int | None = Field(None, gt=0)
    # onnxruntime backend only: execution providers in priority order, comma
    # separated, e.g. "CUDAExecutionProvider,CPUExecutionProvider". Empty means
    # derived from DETECT_DEVICE.
    onnx_providers: Annotated[list[str], NoDecode] = []

    # --- Detection defaults ----------------------------------------------
    # Classes reported when the request does not ask for specific ones.
    # Comma separated in the environment, e.g. DETECT_TARGET_CLASSES=cat,dog.
    # Use "*" to report every class the model knows.
    target_classes: Annotated[list[str], NoDecode] = ["cat"]
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    image_size: int = Field(640, gt=0)
    max_detections: int = Field(100, gt=0)

    # --- Input limits ------------------------------------------------------
    max_upload_bytes: int = Field(10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(40_000_000, gt=0)

    # Run one inference on a blank image at startup so the first request
    # does not pay for lazy initialisation.
    warmup: bool = True

    @field_validator("target_classes", "onnx_providers", mode="before")
    @classmethod
    def _split_classes(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
