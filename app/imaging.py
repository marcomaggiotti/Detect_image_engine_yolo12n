"""Safe decoding of uploaded images."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # HEIC/HEIF photos (iPhone default), when pi-heif is installed
    from pi_heif import register_heif_opener
except ImportError:
    pass
else:
    register_heif_opener()


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes, *, max_pixels: int) -> Image.Image:
    """Decode bytes into an upright RGB image, rejecting oversized images."""
    if not data:
        raise InvalidImageError("Empty image")
    try:
        image = Image.open(io.BytesIO(data))
        # The header is enough to know the size: refuse before decoding pixels.
        width, height = image.size
        if width * height > max_pixels:
            raise InvalidImageError(
                f"Image is {width}x{height}, above the {max_pixels} pixel limit"
            )
        image.load()
        # Apply camera orientation so boxes match what the client sees.
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except InvalidImageError:
        raise
    # ImportError: some ML libraries patch Image.open to lazily load optional
    # format plugins (e.g. HEIF) when decoding fails.
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ImportError,
    ) as exc:
        raise InvalidImageError("Unsupported or corrupt image") from exc


def resize_bilinear(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an HxWx3 uint8 array like OpenCV's INTER_LINEAR, which the
    models were trained with. Pillow's bilinear filter smooths when shrinking,
    which shifts scores enough to create false positives on large photos."""
    if pixels.shape[:2] == (height, width):
        return pixels.astype(np.float32)
    y0, y1, wy = _sample_coords(pixels.shape[0], height)
    x0, x1, wx = _sample_coords(pixels.shape[1], width)
    rows = pixels[y0] * (1 - wy)[:, None, None] + pixels[y1] * wy[:, None, None]
    out = rows[:, x0] * (1 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]
    return out.round()  # OpenCV returns uint8


def _sample_coords(size: int, new_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = ((np.arange(new_size) + 0.5) * (size / new_size) - 0.5).clip(0, size - 1)
    low = np.floor(src).astype(int)
    return low, np.minimum(low + 1, size - 1), (src - low).astype(np.float32)
