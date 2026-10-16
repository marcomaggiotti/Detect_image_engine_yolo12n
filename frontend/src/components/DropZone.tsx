import { useId, useState, type ReactNode } from "react";

type Props = {
  onFiles: (files: File[]) => void;
  multiple?: boolean;
  disabled?: boolean;
  children: ReactNode;
};

/** Click, tap (camera on phones) or drop images here. */
export function DropZone({ onFiles, multiple = false, disabled = false, children }: Props) {
  const id = useId();
  const [dragging, setDragging] = useState(false);

  function accept(list: FileList | null) {
    const files = Array.from(list ?? []).filter((f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name));
    if (files.length) onFiles(multiple ? files : files.slice(0, 1));
  }

  return (
    <label
      htmlFor={id}
      className={`dropzone${dragging ? " dropzone--active" : ""}${disabled ? " dropzone--disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) accept(e.dataTransfer.files);
      }}
    >
      <input
        id={id}
        type="file"
        accept="image/*,.heic,.heif"
        multiple={multiple}
        disabled={disabled}
        className="visually-hidden"
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = ""; // allow choosing the same file again
        }}
      />
      {children}
    </label>
  );
}
