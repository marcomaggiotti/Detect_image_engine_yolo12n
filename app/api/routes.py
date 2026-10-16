from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.identify import NAME_PATTERN, InvalidCatNameError
from app.imaging import InvalidImageError
from app.schemas import (
    Box,
    CatResponse,
    CatsResponse,
    DetectionOut,
    DetectResponse,
    EnrollResponse,
    ErrorResponse,
    IdentifiedCat,
    IdentifyResponse,
    ImageInfo,
    KnownCat,
    ModelInfo,
    ModelRef,
    RejectedPhoto,
)
from app.service import (
    CAT,
    DetectionResult,
    DetectionService,
    IdentificationDisabledError,
    UnknownClassError,
)

MAX_PHOTOS_PER_UPLOAD = 20

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid image or unknown class"},
    413: {"model": ErrorResponse, "description": "Upload too large"},
}

router = APIRouter()


def _service(request: Request) -> DetectionService:
    return request.app.state.service


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _model_ref(request: Request) -> ModelRef:
    settings = _settings(request)
    return ModelRef(
        backend=_service(request).detector.backend_name,
        weights=settings.model_weights,
        version=settings.model_version,
    )


def _parse_classes(raw: list[str] | None) -> list[str] | None:
    """Accept both ?classes=cat,dog and ?classes=cat&classes=dog."""
    if not raw:
        return None
    names = [part.strip() for item in raw for part in item.split(",") if part.strip()]
    return names or None


def _box(xyxy: tuple[float, float, float, float]) -> Box:
    x1, y1, x2, y2 = (round(v, 1) for v in xyxy)
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


async def _read_upload(request: Request, image: UploadFile) -> bytes:
    max_bytes = _settings(request).max_upload_bytes
    data = await image.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Image exceeds {max_bytes} bytes")
    return data


async def _run_detection(
    request: Request,
    image: UploadFile,
    classes: list[str] | None,
    confidence: float | None,
) -> DetectionResult:
    service = _service(request)
    data = await _read_upload(request, image)

    try:
        return await run_in_threadpool(service.detect, data, classes=classes, confidence=confidence)
    except UnknownClassError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{exc}. See GET /v1/model for available labels."
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


_bearer = HTTPBearer(auto_error=False, description="DETECT_ADMIN_TOKEN")


def require_admin(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Changing the known cats needs DETECT_ADMIN_TOKEN, when one is configured."""
    token = _settings(request).admin_token
    if token is None:
        return
    given = credentials.credentials if credentials else ""
    if not secrets.compare_digest(given.encode(), token.get_secret_value().encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or wrong admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _identification_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, IdentificationDisabledError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(
            status.HTTP_409_CONFLICT,
            f"{exc}. Add photos of your cats with POST /v1/cats/{{name}}/photos.",
        )
    return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))


ImageUpload = Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP, ...)")]
CatName = Annotated[
    str,
    Path(description="Your cat's name: lowercase letters, digits, '-' or '_'", examples=["luna"]),
]
Confidence = Annotated[
    float | None,
    Query(ge=0.0, le=1.0, description="Minimum score. Defaults to the configured threshold."),
]


@router.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/model", response_model=ModelInfo, tags=["detection"])
def model_info(request: Request) -> ModelInfo:
    service = _service(request)
    return ModelInfo(
        **_model_ref(request).model_dump(),
        arch=_settings(request).model_arch,
        default_classes=service.default_selection.names,
        confidence_threshold=service.confidence_threshold,
        labels=service.detector.labels,
    )


@router.post(
    "/v1/cat",
    response_model=CatResponse,
    tags=["detection"],
    responses=_ERROR_RESPONSES,
)
async def is_there_a_cat(
    request: Request, image: ImageUpload, confidence: Confidence = None
) -> CatResponse:
    """Answer one question: is there a cat in this image?"""
    result = await _run_detection(request, image, [CAT], confidence)
    scores = [d.confidence for d in result.detections]
    return CatResponse(
        model=_model_ref(request),
        cat=bool(scores),
        count=len(scores),
        confidence=round(max(scores), 4) if scores else None,
        inference_ms=result.inference_ms,
    )


@router.post(
    "/v1/detect",
    response_model=DetectResponse,
    tags=["detection"],
    responses=_ERROR_RESPONSES,
)
async def detect(
    request: Request,
    image: ImageUpload,
    classes: Annotated[
        list[str] | None,
        Query(
            description="Classes to detect, e.g. 'cat' or 'cat,dog,person'. "
            "'*' returns every class. Defaults to DETECT_TARGET_CLASSES."
        ),
    ] = None,
    confidence: Confidence = None,
) -> DetectResponse:
    result = await _run_detection(request, image, _parse_classes(classes), confidence)
    return DetectResponse(
        model=_model_ref(request),
        image=ImageInfo(width=result.width, height=result.height),
        classes=result.classes,
        found=bool(result.detections),
        counts=result.counts,
        detections=[
            DetectionOut(
                label=d.label,
                class_id=d.class_id,
                confidence=round(d.confidence, 4),
                box=_box(d.box),
            )
            for d in result.detections
        ],
        inference_ms=result.inference_ms,
    )


@router.post(
    "/v1/identify",
    response_model=IdentifyResponse,
    tags=["your cats"],
    responses={
        **_ERROR_RESPONSES,
        409: {"model": ErrorResponse, "description": "No cats enrolled yet"},
        503: {"model": ErrorResponse, "description": "Identification not configured"},
    },
)
async def identify(
    request: Request, image: ImageUpload, confidence: Confidence = None
) -> IdentifyResponse:
    """Which of your cats are in this image?"""
    data = await _read_upload(request, image)
    try:
        result = await run_in_threadpool(_service(request).identify, data, confidence=confidence)
    except (IdentificationDisabledError, LookupError, InvalidImageError, UnknownClassError) as exc:
        raise _identification_errors(exc) from exc
    cats = [
        IdentifiedCat(
            name=match.name,
            closest=match.closest,
            similarity=match.similarity,
            confidence=round(detection.confidence, 4),
            box=_box(detection.box),
        )
        for detection, match in result.cats
    ]
    return IdentifyResponse(
        model=_model_ref(request),
        image=ImageInfo(width=result.width, height=result.height),
        names=list(dict.fromkeys(c.name for c in cats if c.name)),
        cats=cats,
        inference_ms=result.inference_ms,
    )


@router.get("/v1/cats", response_model=CatsResponse, tags=["your cats"])
def list_cats(request: Request) -> CatsResponse:
    """Your cats and how many photos each has."""
    gallery = _service(request).gallery
    if gallery is None:
        raise _identification_errors(
            IdentificationDisabledError("Telling cats apart is not configured")
        )
    return CatsResponse(
        cats=[KnownCat(name=n, photos=count) for n, count in gallery.cats().items()],
        threshold=gallery.threshold,
        persistent=gallery.root is not None,
    )


@router.get(
    "/v1/cats/{name}/photo",
    response_class=FileResponse,
    tags=["your cats"],
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "A photo of this cat"},
        404: {"model": ErrorResponse, "description": "Unknown cat, or photos not stored"},
    },
)
def cat_photo(request: Request, name: CatName) -> FileResponse:
    """One photo of this cat (the cropped cat), e.g. for an avatar."""
    gallery = _service(request).gallery
    if gallery is None:
        raise _identification_errors(
            IdentificationDisabledError("Telling cats apart is not configured")
        )
    try:
        path = gallery.photo(name)
    except InvalidCatNameError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No stored photo of {name!r}")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.post(
    "/v1/cats/{name}/photos",
    response_model=EnrollResponse,
    tags=["your cats"],
    dependencies=[Depends(require_admin)],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid name or too many files"},
        401: {"model": ErrorResponse, "description": "Missing or wrong admin token"},
        409: {"model": ErrorResponse, "description": "Too many photos for this cat"},
        503: {"model": ErrorResponse, "description": "Identification not configured"},
    },
)
async def add_photos(
    request: Request,
    name: CatName,
    images: Annotated[
        list[UploadFile],
        File(description=f"Photos showing only this cat (up to {MAX_PHOTOS_PER_UPLOAD})"),
    ],
) -> EnrollResponse:
    """Teach the service one of your cats. 5-10 varied photos are a good start.

    Each photo must show only this cat. Photos without a cat, or with several,
    are rejected (and listed in the response); the rest take effect immediately.
    """
    if not NAME_PATTERN.match(name):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(InvalidCatNameError(name)))
    if len(images) > MAX_PHOTOS_PER_UPLOAD:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"At most {MAX_PHOTOS_PER_UPLOAD} photos per upload"
        )
    max_bytes = _settings(request).max_upload_bytes
    datas, too_big = [], []
    for upload in images:
        data = await upload.read(max_bytes + 1)
        (too_big if len(data) > max_bytes else datas).append((upload.filename or "", data))
    try:
        result = await run_in_threadpool(
            _service(request).enroll, name, [data for _, data in datas]
        )
    except (IdentificationDisabledError, InvalidCatNameError, UnknownClassError) as exc:
        raise _identification_errors(exc) from exc
    except ValueError as exc:  # over the per-cat photo limit
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    rejected = [RejectedPhoto(file=f, reason=f"Over {max_bytes} bytes") for f, _ in too_big]
    rejected += [RejectedPhoto(file=datas[i][0], reason=reason) for i, reason in result.rejected]
    return EnrollResponse(
        name=result.name,
        added=result.added,
        duplicates=result.duplicates,
        rejected=rejected,
        photos=result.photos,
    )


@router.delete(
    "/v1/cats/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["your cats"],
    dependencies=[Depends(require_admin)],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or wrong admin token"},
        404: {"model": ErrorResponse, "description": "Unknown cat"},
    },
)
def delete_cat(request: Request, name: CatName) -> Response:
    """Forget one of your cats and its photos."""
    gallery = _service(request).gallery
    if gallery is None:
        raise _identification_errors(
            IdentificationDisabledError("Telling cats apart is not configured")
        )
    try:
        removed = gallery.remove(name)
    except InvalidCatNameError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not removed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No cat named {name!r}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
