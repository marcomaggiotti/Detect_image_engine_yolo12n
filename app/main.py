"""FastAPI application entry point: ``uvicorn app.main:app``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __doc__ as description
from app.api.routes import router
from app.config import Settings, get_settings
from app.detectors import Detector, DetectorConfig, create_detector
from app.identify import CatGallery, Embedder
from app.service import DetectionService

logger = logging.getLogger("detect_image_engine_yolo12n")


def build_detector(settings: Settings) -> Detector:
    config = DetectorConfig(
        weights=settings.model_weights,
        arch=settings.model_arch,
        labels_file=settings.model_labels,
        device=settings.device,
        iou_threshold=settings.iou_threshold,
        image_size=settings.image_size,
        max_detections=settings.max_detections,
        threads=settings.threads,
        providers=tuple(settings.onnx_providers) or None,
    )
    return create_detector(settings.backend, config)


def build_gallery(settings: Settings, embedder: Embedder | None = None) -> CatGallery | None:
    if embedder is None:
        if not settings.embedder_weights:
            return None
        embedder = Embedder(
            settings.embedder_weights,
            threads=settings.threads,
            providers=tuple(settings.onnx_providers) or None,
        )
    if not settings.cats_dir:
        logger.warning("DETECT_CATS_DIR is not set: photos of your cats are lost on restart")
    if settings.admin_token is None:
        logger.warning("DETECT_ADMIN_TOKEN is not set: anyone can add or delete cats")
    return CatGallery(embedder, root=settings.cats_dir, threshold=settings.identify_threshold)


def create_app(
    settings: Settings | None = None,
    detector: Detector | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build the app. ``detector`` and ``embedder`` can be injected (tests, embedding)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = detector or build_detector(settings)
        app.state.settings = settings
        app.state.service = DetectionService(
            model,
            default_classes=settings.target_classes,
            confidence_threshold=settings.confidence_threshold,
            max_image_pixels=settings.max_image_pixels,
            gallery=build_gallery(settings, embedder),
        )
        if settings.warmup:
            model.warmup()
        logger.info(
            "Loaded %s model %s (%d classes), default classes: %s",
            model.backend_name,
            settings.model_weights,
            len(model.labels),
            app.state.service.default_selection.names or "all",
        )
        yield

    app = FastAPI(
        title="Detect Image Engine (YOLO12n)",
        description=description,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    _show_file_pickers_for_file_lists(app)
    _serve_web_interface(app, settings)
    return app


def _serve_web_interface(app: FastAPI, settings: Settings) -> None:
    """Serve the built React app at "/" (after the API routes, which take precedence)."""
    web_dir = Path(settings.web_dir) if settings.web_dir else Path(__file__).parent / "web"
    if (web_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    elif settings.web_dir:
        logger.warning("No web interface in %s (run npm run build in frontend/)", web_dir)


def _show_file_pickers_for_file_lists(app: FastAPI) -> None:
    """Make /docs offer file pickers for multi-file fields.

    FastAPI describes uploaded files with ``contentMediaType``; Swagger UI only
    turns that into a file picker for a single file, and shows text boxes for a
    list of files unless its items also say ``format: binary``.
    """
    generate = app.openapi

    def openapi() -> dict:
        schema = generate()
        for component in schema.get("components", {}).get("schemas", {}).values():
            for field in component.get("properties", {}).values():
                items = field.get("items", {})
                if field.get("type") == "array" and "contentMediaType" in items:
                    items["format"] = "binary"
        return schema

    app.openapi = openapi


def _default_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return create_app()


app = _default_app()
