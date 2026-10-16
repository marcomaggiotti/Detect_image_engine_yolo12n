import { useEffect, useRef, useState } from "react";
import { detectCats, identify, type CatsResponse } from "../api";
import { prepareImage } from "../image";
import { fromDetect, fromIdentify, type CheckResult } from "../results";
import { DropZone } from "./DropZone";
import { PhotoResult } from "./PhotoResult";

type Props = { cats: CatsResponse | null | undefined };

type State =
  | { kind: "idle" }
  | { kind: "checking"; previewUrl: string | null }
  | { kind: "done"; previewUrl: string | null; result: CheckResult }
  | { kind: "error"; message: string };

export function CheckPhoto({ cats }: Props) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const previewRef = useRef<string | null>(null);
  // Names are checked once you've added cats; otherwise it's "is there a cat?".
  const withNames = (cats?.cats.length ?? 0) > 0;

  async function check(file: File) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const prepared = await prepareImage(file);
    const previewUrl = URL.createObjectURL(prepared);
    previewRef.current = previewUrl;
    setState({ kind: "checking", previewUrl });
    try {
      const result = withNames ? fromIdentify(await identify(prepared)) : fromDetect(await detectCats(prepared));
      setState({ kind: "done", previewUrl, result });
    } catch (error) {
      setState({ kind: "error", message: (error as Error).message });
    }
  }

  // Paste an image (Ctrl/Cmd+V) to check it.
  useEffect(() => {
    function onPaste(event: ClipboardEvent) {
      const file = Array.from(event.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) check(file);
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });

  useEffect(() => () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
  }, []);

  const busy = state.kind === "checking";

  return (
    <div className="stack">
      <DropZone onFiles={([file]) => check(file)} disabled={busy}>
        <span className="dropzone__icon" aria-hidden="true">📷</span>
        <span className="dropzone__title">{busy ? "Checking…" : "Choose or take a photo"}</span>
        <span className="muted">
          {withNames
            ? "We'll tell you which of your cats is in it."
            : "We'll tell you if there's a cat in it. Add your cats to get their names too."}
          <span className="hide-on-touch"> You can also drop or paste an image here.</span>
        </span>
      </DropZone>

      {state.kind === "checking" && (
        <div className="result result--loading" role="status">
          <div className="spinner" aria-hidden="true" /> Looking for cats…
        </div>
      )}
      {state.kind === "error" && (
        <p className="notice notice--error" role="alert">
          {state.message}
        </p>
      )}
      {state.kind === "done" && <PhotoResult previewUrl={state.previewUrl} result={state.result} />}
    </div>
  );
}
