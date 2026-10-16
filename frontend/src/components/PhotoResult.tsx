import { useState } from "react";
import { percent, summarize, type CheckResult } from "../results";

type Props = { previewUrl: string | null; result: CheckResult };

/** The answer, the photo with a box around each cat, and the details. */
export function PhotoResult({ previewUrl, result }: Props) {
  const [previewFailed, setPreviewFailed] = useState(false);
  const { headline, found } = summarize(result.findings);
  const { width, height } = result.image;
  const pct = (value: number, total: number) => `${(value / total) * 100}%`;

  return (
    <section className="result" aria-live="polite">
      <h2 className={`result__headline ${found ? "result__headline--yes" : "result__headline--no"}`}>
        <span aria-hidden="true">{found ? "🐱" : "🔍"}</span> {headline}
      </h2>

      {/* Keeps the photo's own shape, at most 70% of the screen height, so boxes line up. */}
      <div className="photo" style={{ aspectRatio: `${width} / ${height}`, width: `min(100%, calc(70vh * ${width / height}))` }}>
        {previewUrl && !previewFailed ? (
          <img src={previewUrl} alt="Your photo" onError={() => setPreviewFailed(true)} />
        ) : (
          <div className="photo__placeholder">This browser can't show this photo, but it was checked.</div>
        )}
        {result.findings.map((f, i) => (
          <div
            key={i}
            className={`box ${f.known === false ? "box--unknown" : "box--cat"}${f.box.y1 / height < 0.08 ? " box--label-inside" : ""}`}
            style={{
              left: pct(f.box.x1, width),
              top: pct(f.box.y1, height),
              width: pct(f.box.x2 - f.box.x1, width),
              height: pct(f.box.y2 - f.box.y1, height),
            }}
          >
            <span className="box__label">{f.title}</span>
          </div>
        ))}
      </div>

      {result.findings.length > 0 && (
        <ul className="findings">
          {result.findings.map((f, i) => (
            <li key={i} className="findings__item">
              <span className={`dot ${f.known === false ? "dot--unknown" : "dot--cat"}`} aria-hidden="true" />
              <strong>{f.title}</strong>
              <span className="muted">
                {f.similarity !== undefined &&
                  (f.known ? `${percent(f.similarity)} similar · ` : `closest: ${f.closest}, ${percent(f.similarity)} similar · `)}
                detected with {percent(f.confidence)} confidence
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="muted small">Checked in {(result.ms / 1000).toFixed(1)} s</p>
    </section>
  );
}
