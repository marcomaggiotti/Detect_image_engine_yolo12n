/** Longest side of photos sent to the server. The detector works at 640 px, so
 * this keeps plenty of detail while making uploads from phones much smaller. */
export const MAX_UPLOAD_SIDE = 1600;

/**
 * Shrink a large photo before upload, with the camera's orientation applied.
 * Returns the original file when it is small enough, or when the browser can't
 * decode it (e.g. HEIC outside Safari): the server handles those itself.
 */
export async function prepareImage(file: File, maxSide = MAX_UPLOAD_SIDE): Promise<File> {
  if (typeof createImageBitmap !== "function") return file;
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    return file;
  }
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  if (scale === 1) {
    bitmap.close();
    return file;
  }
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  if (!blob) return file;
  return new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" });
}
