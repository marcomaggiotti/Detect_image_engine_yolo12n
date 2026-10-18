import { useEffect, useRef, useState } from "react";
import type { Message, Photo } from "../agent/types";
import type { Box, ImageInfo } from "../api";
import { cropImage } from "../image";
import { displayName, originalSize, percent, toOriginal, type CheckResult, type Finding, type Scale } from "../results";

type Checked = { id: number; original: Photo; scale: Scale; result: CheckResult; checkedAt: number };

type Props = { messages: Message[]; onCheck: (files: File[]) => void };

/** Every cat found in the photos checked so far, cropped, with its box in
 * pixels of the original photo. */
export function Detections({ messages, onCheck }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const checked: Checked[] = [];
  let checking = 0;
  for (const m of messages) {
    if (m.role !== "assistant") continue;
    if (m.check) checked.push({ id: m.id, ...m.check });
    else if (m.checking) checking++;
  }
  checked.reverse(); // newest first
  const total = checked.reduce((n, c) => n + c.result.findings.length, 0);

  return (
    <div className="detections">
      <div className="detections__top">
        <div>
          <h2 className="detections__title">Detections</h2>
          <p className="muted small">
            {checked.length
              ? `${total} cat${total === 1 ? "" : "s"} in ${checked.length} photo${checked.length === 1 ? "" : "s"}. `
              : ""}
            Boxes are in pixels of each original photo: x to the right, y down, from its top-left corner.
          </p>
        </div>
        <button type="button" className="button" onClick={() => input.current?.click()}>
          Check a photo
        </button>
        <input
          ref={input}
          type="file"
          accept="image/*,.heic,.heif"
          multiple
          className="visually-hidden"
          tabIndex={-1}
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            if (files.length) onCheck(files);
            e.target.value = "";
          }}
        />
      </div>

      {checking > 0 && (
        <p className="detections__checking muted" role="status">
          <span className="spinner spinner--small" aria-hidden="true" /> Checking {checking} photo{checking === 1 ? "" : "s"}…
        </p>
      )}
      {checked.length === 0 && checking === 0 && (
        <p className="detections__empty muted">No photos checked yet. Choose one here, or send one to the assistant.</p>
      )}
      {checked.map((c) => (
        <PhotoDetections key={c.id} checked={c} />
      ))}
    </div>
  );
}

type Row = Finding & { index: number; originalBox: Box };

function PhotoDetections({ checked }: { checked: Checked }) {
  const size = originalSize(checked.result.image, checked.scale);
  const rows: Row[] = checked.result.findings.map((f, i) => ({ ...f, index: i + 1, originalBox: toOriginal(f.box, checked.scale, size) }));
  const name = checked.original.file.name || "photo";
  const [copied, setCopied] = useState("");

  async function copy() {
    try {
      await navigator.clipboard.writeText(toJson(name, size, rows));
      setCopied("Copied");
    } catch {
      setCopied("Couldn't copy: use Download");
    }
    window.setTimeout(() => setCopied(""), 2500);
  }

  function download() {
    const url = URL.createObjectURL(new Blob([toJson(name, size, rows)], { type: "application/json" }));
    const link = Object.assign(document.createElement("a"), { href: url, download: `${stem(name)}-detections.json` });
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="detections__photo" aria-label={`Detections in ${name}`}>
      <header className="detections__photo-header">
        <img className="detections__thumb" src={checked.original.url} alt="" onError={(e) => (e.currentTarget.style.visibility = "hidden")} />
        <div className="detections__photo-info">
          <strong className="detections__file">{name}</strong>
          <span className="muted small">
            {size.width} × {size.height} px · {new Date(checked.checkedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} ·{" "}
            {rows.length ? `${rows.length} cat${rows.length === 1 ? "" : "s"}` : "no cat"}
          </span>
        </div>
        {rows.length > 0 && (
          <div className="detections__export">
            <button type="button" className="chip" onClick={copy}>
              {copied || "Copy JSON"}
            </button>
            <button type="button" className="chip" onClick={download}>
              Download JSON
            </button>
          </div>
        )}
      </header>
      {rows.length > 0 && (
        <ul className="detections__grid">
          {rows.map((row) => (
            <DetectionCard key={row.index} row={row} original={checked.original.file} fileStem={stem(name)} />
          ))}
        </ul>
      )}
    </section>
  );
}

function DetectionCard({ row, original, fileStem }: { row: Row; original: File; fileStem: string }) {
  const crop = useCrop(original, row.originalBox);
  const { x1, y1, x2, y2 } = row.originalBox;
  const round = (v: number) => Math.round(v);
  return (
    <li className="detection">
      <div className="detection__crop">
        {crop === undefined ? (
          <span className="spinner spinner--small" aria-label="Cropping" />
        ) : crop === null ? (
          <span className="muted small">No preview in this browser</span>
        ) : (
          <img src={crop} alt={`${row.title}, cropped`} />
        )}
        <span className="detection__index">#{row.index}</span>
      </div>
      <div className="detection__body">
        <p className="detection__title">
          <span className={`dot ${row.known === false ? "dot--unknown" : "dot--cat"}`} aria-hidden="true" />
          {row.title}
        </p>
        <p className="muted small">
          {percent(row.confidence)} sure it's a cat
          {row.similarity !== undefined && ` · ${percent(row.similarity)} like ${row.known ? "its photos" : displayName(row.closest!)}`}
        </p>
        <dl className="coords">
          <dt>x1</dt>
          <dd>{round(x1)}</dd>
          <dt>y1</dt>
          <dd>{round(y1)}</dd>
          <dt>x2</dt>
          <dd>{round(x2)}</dd>
          <dt>y2</dt>
          <dd>{round(y2)}</dd>
          <dt>size</dt>
          <dd className="coords__wide">
            {round(x2 - x1)} × {round(y2 - y1)}
          </dd>
        </dl>
        {crop && (
          <a className="small" href={crop} download={`${fileStem}-cat-${row.index}.jpg`}>
            Download crop
          </a>
        )}
      </div>
    </li>
  );
}

/** Object URL of the cropped cat; undefined while cropping, null if impossible. */
function useCrop(file: File, box: Box): string | null | undefined {
  const [url, setUrl] = useState<string | null | undefined>(undefined);
  const { x1, y1, x2, y2 } = box;
  useEffect(() => {
    let current: string | null = null;
    let cancelled = false;
    cropImage(file, { x1, y1, x2, y2 }).then((blob) => {
      if (cancelled) return;
      current = blob ? URL.createObjectURL(blob) : null;
      setUrl(current);
    });
    return () => {
      cancelled = true;
      if (current) URL.revokeObjectURL(current);
    };
  }, [file, x1, y1, x2, y2]);
  return url;
}

function stem(name: string): string {
  return name.replace(/\.[^.]+$/, "") || "photo";
}

/** The detections of one photo, coordinates in pixels of the original photo. */
export function toJson(file: string, size: ImageInfo, rows: Row[]): string {
  return JSON.stringify(
    {
      file,
      width: size.width,
      height: size.height,
      detections: rows.map((r) => ({
        index: r.index,
        label: "cat",
        name: r.known ? r.closest ?? null : null,
        confidence: Math.round(r.confidence * 1e4) / 1e4,
        ...(r.similarity !== undefined && { similarity: r.similarity, closest: r.closest }),
        box: r.originalBox,
        width: Math.round((r.originalBox.x2 - r.originalBox.x1) * 10) / 10,
        height: Math.round((r.originalBox.y2 - r.originalBox.y1) * 10) / 10,
      })),
    },
    null,
    2,
  );
}
