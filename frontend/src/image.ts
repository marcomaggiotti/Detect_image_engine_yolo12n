import type { Box } from "./api";

/** Longest side of photos sent to the server. The detector works at 640 px, so
 * this keeps plenty of detail while making uploads from phones much smaller. */
export const MAX_UPLOAD_SIDE = 1600;

/** A photo ready to upload, and how to map the server's pixels back to the
 * original photo (upright, full size): original = uploaded × scale. */
export type PreparedImage = { file: File; scaleX: number; scaleY: number };

/**
 * Shrink a large photo before upload, with the camera's orientation applied.
 * Returns the original file (scale 1) when it is small enough, or when the
 * browser can't decode it (e.g. HEIC outside Safari): the server handles those
 * itself, applying the orientation too, so its pixels are the original's.
 */
export async function prepareImage(file: File, maxSide = MAX_UPLOAD_SIDE): Promise<PreparedImage> {
  const unchanged = { file, scaleX: 1, scaleY: 1 };
  if (typeof createImageBitmap !== "function") return unchanged;
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    return unchanged;
  }
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  if (scale === 1) {
    bitmap.close();
    return unchanged;
  }
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const scaleX = bitmap.width / canvas.width;
  const scaleY = bitmap.height / canvas.height;
  bitmap.close();
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  if (!blob) return unchanged;
  return { file: new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" }), scaleX, scaleY };
}

// Each original photo is decoded once, however many cats are cropped from it.
const decoded = new WeakMap<File, Promise<ImageBitmap | null>>();

function decode(file: File): Promise<ImageBitmap | null> {
  let bitmap = decoded.get(file);
  if (!bitmap) {
    bitmap =
      typeof createImageBitmap === "function"
        ? createImageBitmap(file, { imageOrientation: "from-image" }).catch(() => null)
        : Promise.resolve(null);
    decoded.set(file, bitmap);
  }
  return bitmap;
}

/** The part of a photo inside ``box`` (pixels of the upright original), as a
 * JPEG; null when the browser can't decode the photo. */
export async function cropImage(file: File, box: Box): Promise<Blob | null> {
  const bitmap = await decode(file);
  if (!bitmap) return null;
  const x = Math.max(0, Math.floor(box.x1));
  const y = Math.max(0, Math.floor(box.y1));
  const width = Math.min(bitmap.width, Math.ceil(box.x2)) - x;
  const height = Math.min(bitmap.height, Math.ceil(box.y2)) - y;
  if (width <= 0 || height <= 0) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")?.drawImage(bitmap, x, y, width, height, 0, 0, width, height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
}
