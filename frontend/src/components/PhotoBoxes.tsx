import { useState } from "react";
import type { CheckResult } from "../results";

type Props = { url: string; result: CheckResult };

/** The photo with a labelled box around each cat. */
export function PhotoBoxes({ url, result }: Props) {
  const [failed, setFailed] = useState(false);
  const { width, height } = result.image;
  const pct = (value: number, total: number) => `${(value / total) * 100}%`;

  return (
    // Keeps the photo's own shape, at most 60% of the screen height, so boxes line up.
    <div className="photo" style={{ aspectRatio: `${width} / ${height}`, width: `min(100%, calc(60vh * ${width / height}))` }}>
      {failed ? (
        <div className="photo__placeholder">This browser can't show this photo, but it was checked.</div>
      ) : (
        <img src={url} alt="The photo you sent" onError={() => setFailed(true)} />
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
  );
}
