// Typed client for the service's HTTP API (see the README and /docs).

export type Box = { x1: number; y1: number; x2: number; y2: number };
export type ImageInfo = { width: number; height: number };

export type Detection = { label: string; class_id: number; confidence: number; box: Box };
export type DetectResponse = {
  image: ImageInfo;
  found: boolean;
  counts: Record<string, number>;
  detections: Detection[];
  inference_ms: number;
};

export type IdentifiedCat = {
  name: string | null;
  closest: string;
  similarity: number;
  confidence: number;
  box: Box;
};
export type IdentifyResponse = {
  image: ImageInfo;
  names: string[];
  cats: IdentifiedCat[];
  inference_ms: number;
};

export type KnownCat = { name: string; photos: number };
export type CatsResponse = { cats: KnownCat[]; threshold: number; persistent: boolean };

export type RejectedPhoto = { file: string; reason: string };
export type EnrollResponse = {
  name: string;
  added: number;
  duplicates: number;
  rejected: RejectedPhoto[];
  photos: number;
};

/** The server accepts at most this many photos per upload; more are sent in batches. */
export const MAX_PHOTOS_PER_UPLOAD = 20;

/** Same rule as the server: lowercase letters, digits, "-" and "_", 1 to 32 characters. */
export const CAT_NAME = /^[a-z0-9][a-z0-9_-]{0,31}$/;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }
  if (!response.ok) {
    let detail = response.statusText || `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

function imageForm(field: string, files: File[]): FormData {
  const form = new FormData();
  for (const file of files) form.append(field, file, file.name);
  return form;
}

function auth(token: string): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function detectCats(file: File): Promise<DetectResponse> {
  return request("/v1/detect?classes=cat", { method: "POST", body: imageForm("image", [file]) });
}

export function identify(file: File): Promise<IdentifyResponse> {
  return request("/v1/identify", { method: "POST", body: imageForm("image", [file]) });
}

/** Known cats, or null when the server isn't set up to tell cats apart. */
export async function listCats(): Promise<CatsResponse | null> {
  try {
    return await request<CatsResponse>("/v1/cats");
  } catch (error) {
    if (error instanceof ApiError && error.status === 503) return null;
    throw error;
  }
}

/** Upload photos of one cat, in batches the server accepts, and merge the results. */
export async function addPhotos(name: string, files: File[], token: string): Promise<EnrollResponse> {
  const total: EnrollResponse = { name, added: 0, duplicates: 0, rejected: [], photos: 0 };
  for (let start = 0; start < files.length; start += MAX_PHOTOS_PER_UPLOAD) {
    const batch = files.slice(start, start + MAX_PHOTOS_PER_UPLOAD);
    const result = await request<EnrollResponse>(`/v1/cats/${encodeURIComponent(name)}/photos`, {
      method: "POST",
      headers: auth(token),
      body: imageForm("images", batch),
    });
    total.added += result.added;
    total.duplicates += result.duplicates;
    total.rejected.push(...result.rejected);
    total.photos = result.photos;
  }
  return total;
}

export function deleteCat(name: string, token: string): Promise<void> {
  return request(`/v1/cats/${encodeURIComponent(name)}`, { method: "DELETE", headers: auth(token) });
}

/** URL of one photo of a cat; ``version`` changes it after new photos are added. */
export function catPhotoUrl(name: string, version: number): string {
  return `/v1/cats/${encodeURIComponent(name)}/photo?v=${version}`;
}

/** Turn what someone types into a valid cat name, e.g. "Mr Whiskers" -> "mr-whiskers". */
export function toCatName(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9_-]/g, "")
    .slice(0, 32);
}
