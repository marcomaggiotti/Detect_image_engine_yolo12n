import type { Box, DetectResponse, IdentifyResponse, ImageInfo } from "./api";

/** One cat found in a photo, ready to draw and describe. */
export type Finding = {
  box: Box;
  /** Your cat's name, "Unknown cat", or "Cat" when names aren't checked. */
  title: string;
  /** null: names weren't checked; false: a cat that isn't one of yours. */
  known: boolean | null;
  confidence: number;
  similarity?: number;
  closest?: string;
};

export type CheckResult = { image: ImageInfo; findings: Finding[]; ms: number };

export function fromDetect(response: DetectResponse): CheckResult {
  return {
    image: response.image,
    findings: response.detections.map((d) => ({
      box: d.box,
      title: "Cat",
      known: null,
      confidence: d.confidence,
    })),
    ms: response.inference_ms,
  };
}

export function fromIdentify(response: IdentifyResponse): CheckResult {
  return {
    image: response.image,
    findings: response.cats.map((c) => ({
      box: c.box,
      title: c.name ? displayName(c.name) : "Unknown cat",
      known: c.name !== null,
      confidence: c.confidence,
      similarity: c.similarity,
      closest: c.closest,
    })),
    ms: response.inference_ms,
  };
}

/** "mr-whiskers" -> "Mr whiskers" */
export function displayName(name: string): string {
  const spaced = name.replace(/[-_]+/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function listNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** The one-line answer shown above the photo. */
export function summarize(findings: Finding[]): { headline: string; found: boolean } {
  if (findings.length === 0) return { headline: "No cat in this photo", found: false };
  if (findings.every((f) => f.known === null)) {
    const n = findings.length;
    return { headline: n === 1 ? "Yes, there's a cat!" : `Yes, ${n} cats!`, found: true };
  }
  const names = [...new Set(findings.filter((f) => f.known).map((f) => f.title))];
  const strangers = findings.filter((f) => f.known === false).length;
  if (names.length === 0) {
    return {
      headline: strangers === 1 ? "A cat, but not one of yours" : `${strangers} cats, none of them yours`,
      found: true,
    };
  }
  const others = strangers === 0 ? [] : [strangers === 1 ? "a cat you haven't added" : `${strangers} cats you haven't added`];
  const who = listNames([...names, ...others]);
  return { headline: names.length === 1 && strangers === 0 ? `It's ${who}!` : `${who}`, found: true };
}

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Original pixels per uploaded pixel, on each axis (1 when the photo wasn't shrunk). */
export type Scale = { x: number; y: number };

/** Size of the original photo, from the size the server saw. */
export function originalSize(image: ImageInfo, scale: Scale): ImageInfo {
  return { width: Math.round(image.width * scale.x), height: Math.round(image.height * scale.y) };
}

/** A box from the server, in pixels of the original photo (upright, full size). */
export function toOriginal(box: Box, scale: Scale, size: ImageInfo): Box {
  const clamp = (value: number, max: number) => Math.round(Math.min(Math.max(value, 0), max) * 10) / 10;
  return {
    x1: clamp(box.x1 * scale.x, size.width),
    y1: clamp(box.y1 * scale.y, size.height),
    x2: clamp(box.x2 * scale.x, size.width),
    y2: clamp(box.y2 * scale.y, size.height),
  };
}

