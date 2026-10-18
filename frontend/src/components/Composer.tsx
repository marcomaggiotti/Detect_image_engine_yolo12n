import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";

export type ComposerHandle = { pickPhotos: () => void; addFiles: (files: File[]) => void };

type Props = { onSend: (text: string, files: File[]) => void; knowsCats: boolean };

function isImage(file: File) {
  return file.type.startsWith("image/") || /\.(heic|heif)$/i.test(file.name);
}

/** Text and photos to send to the assistant. */
export const Composer = forwardRef<ComposerHandle, Props>(function Composer({ onSend, knowsCats }, ref) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const input = useRef<HTMLInputElement>(null);
  const textBox = useRef<HTMLInputElement>(null);
  const previews = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => previews.forEach((url) => URL.revokeObjectURL(url)), [previews]);

  function addFiles(more: File[]) {
    const images = more.filter(isImage);
    if (images.length) {
      setFiles((current) => [...current, ...images]);
      textBox.current?.focus();
    }
  }

  useImperativeHandle(ref, () => ({ pickPhotos: () => input.current?.click(), addFiles }));

  function send() {
    if (!text.trim() && files.length === 0) return;
    onSend(text, files);
    setText("");
    setFiles([]);
  }

  const placeholder = files.length ? (knowsCats ? "Who is it? Or “this is Luna”" : "Add a name, e.g. “this is Luna”") : "Send a photo or a message";

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      {files.length > 0 && (
        <ul className="composer__photos" aria-label="Photos to send">
          {files.map((file, i) => (
            <li key={previews[i]}>
              <img src={previews[i]} alt={file.name} onError={(e) => (e.currentTarget.style.visibility = "hidden")} />
              <button type="button" aria-label={`Remove ${file.name}`} onClick={() => setFiles((c) => c.filter((_, j) => j !== i))}>
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="composer__row">
        <button type="button" className="icon-button" aria-label="Add photos" title="Add photos" onClick={() => input.current?.click()}>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <path
              fill="currentColor"
              d="M9 3 7.2 5H4a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-3.2L15 3H9Zm3 5a5 5 0 1 1 0 10 5 5 0 0 1 0-10Zm0 2a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z"
            />
          </svg>
        </button>
        <input
          ref={input}
          type="file"
          accept="image/*,.heic,.heif"
          multiple
          className="visually-hidden"
          tabIndex={-1}
          onChange={(e) => {
            addFiles(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        <input
          ref={textBox}
          className="composer__text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onPaste={(e) => {
            const pasted = Array.from(e.clipboardData.files);
            if (pasted.some(isImage)) {
              e.preventDefault();
              addFiles(pasted);
            }
          }}
          placeholder={placeholder}
          aria-label="Message"
          autoComplete="off"
        />
        <button type="submit" className="icon-button icon-button--send" aria-label="Send" title="Send" disabled={!text.trim() && files.length === 0}>
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <path fill="currentColor" d="M3.4 20.4 21 12 3.4 3.6 3.4 10l12.6 2-12.6 2z" />
          </svg>
        </button>
      </div>
    </form>
  );
});
