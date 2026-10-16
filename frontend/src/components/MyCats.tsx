import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  CAT_NAME,
  addPhotos,
  catPhotoUrl,
  deleteCat,
  toCatName,
  type CatsResponse,
  type EnrollResponse,
  type KnownCat,
} from "../api";
import { prepareImage } from "../image";
import { displayName, percent } from "../results";
import { DropZone } from "./DropZone";

type Props = {
  cats: CatsResponse | null | undefined;
  refresh: () => Promise<void>;
  token: string;
  setToken: (token: string) => void;
};

export function MyCats({ cats, refresh, token, setToken }: Props) {
  const [version, setVersion] = useState(0); // refreshes the cat photos after uploads
  const [askToken, setAskToken] = useState(!token);

  if (cats === undefined) return <p className="muted">Loading your cats…</p>;
  if (cats === null) {
    return (
      <p className="notice">
        This server isn't set up to tell cats apart. Its administrator needs to set{" "}
        <code>DETECT_EMBEDDER_WEIGHTS</code> (the Docker image does).
      </p>
    );
  }

  async function changed() {
    await refresh();
    setVersion((v) => v + 1);
  }

  // A rejected (or missing) token is forgotten, and asked for again.
  function onAuthError(error: unknown) {
    if (error instanceof ApiError && error.status === 401) {
      setToken("");
      setAskToken(true);
    }
  }

  return (
    <div className="stack">
      {!cats.persistent && (
        <p className="notice notice--warning">
          Photos are only kept in memory on this server: your cats will be forgotten when it restarts.
        </p>
      )}

      <TokenPanel token={token} setToken={setToken} open={askToken} setOpen={setAskToken} />

      {cats.cats.length > 0 && (
        <section aria-label="Your cats" className="cat-grid">
          {cats.cats.map((cat) => (
            <CatCard key={cat.name} cat={cat} version={version} token={token} onChanged={changed} onAuthError={onAuthError} />
          ))}
        </section>
      )}

      <AddCatForm existing={cats.cats.map((c) => c.name)} token={token} onAdded={changed} onAuthError={onAuthError} />

      <p className="muted small">
        A cat gets its name when it looks at least {percent(cats.threshold)} similar to its photos. If your cats show up as
        “Unknown cat”, add a few more photos of them in different poses and light.
      </p>
    </div>
  );
}

type TokenProps = {
  token: string;
  setToken: (t: string) => void;
  open: boolean;
  setOpen: (open: boolean) => void;
};

function TokenPanel({ token, setToken, open, setOpen }: TokenProps) {
  const [draft, setDraft] = useState("");

  if (!open) {
    return (
      <p className="token-line muted small">
        <span aria-hidden="true">🔑</span> {token ? "Admin token saved on this device." : "No admin token set."}{" "}
        <button type="button" className="link" onClick={() => setOpen(true)}>
          {token ? "Change" : "Set one"}
        </button>
        {token && (
          <>
            {" · "}
            <button type="button" className="link" onClick={() => setToken("")}>
              Forget
            </button>
          </>
        )}
      </p>
    );
  }
  return (
    <form
      className="card token"
      onSubmit={(e) => {
        e.preventDefault();
        if (draft.trim()) {
          setToken(draft.trim());
          setDraft("");
          setOpen(false);
        }
      }}
    >
      <label htmlFor="token" className="card__title">
        Admin token
      </label>
      <p className="muted small">
        Needed to add or remove cats (it's the server's <code>DETECT_ADMIN_TOKEN</code>; on Render: your service →
        Environment).
      </p>
      <div className="row">
        <input
          id="token"
          type="password"
          autoComplete="off"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Paste the token"
        />
        <button type="submit" className="button" disabled={!draft.trim()}>
          Save
        </button>
        <button type="button" className="button button--secondary" onClick={() => setOpen(false)}>
          {token ? "Cancel" : "Skip"}
        </button>
      </div>
    </form>
  );
}

type CardProps = {
  cat: KnownCat;
  version: number;
  token: string;
  onChanged: () => Promise<void>;
  onAuthError: (error: unknown) => void;
};

function CatCard({ cat, version, token, onChanged, onAuthError }: CardProps) {
  const [photoFailed, setPhotoFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<EnrollResponse | null>(null);
  const [error, setError] = useState("");
  useEffect(() => setPhotoFailed(false), [version]);

  async function upload(files: File[]) {
    setBusy(true);
    setError("");
    setReport(null);
    try {
      setReport(await addPhotos(cat.name, await Promise.all(files.map((f) => prepareImage(f))), token));
      await onChanged();
    } catch (e) {
      onAuthError(e);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!window.confirm(`Forget ${displayName(cat.name)} and all its photos?`)) return;
    setBusy(true);
    setError("");
    try {
      await deleteCat(cat.name, token);
      await onChanged();
    } catch (e) {
      onAuthError(e);
      setError((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <article className="card cat-card">
      <div className="cat-card__header">
        {photoFailed ? (
          <div className="avatar avatar--initial" aria-hidden="true">
            {cat.name.charAt(0).toUpperCase()}
          </div>
        ) : (
          <img className="avatar" src={catPhotoUrl(cat.name, version)} alt="" onError={() => setPhotoFailed(true)} />
        )}
        <div>
          <h3 className="cat-card__name">{displayName(cat.name)}</h3>
          <p className="muted small">
            {cat.photos} photo{cat.photos === 1 ? "" : "s"}
          </p>
        </div>
      </div>
      <div className="row">
        <DropZone onFiles={upload} multiple disabled={busy}>
          <span className="button button--secondary">{busy ? "Working…" : "Add photos"}</span>
        </DropZone>
        <button type="button" className="button button--danger" onClick={remove} disabled={busy}>
          Remove
        </button>
      </div>
      {error && (
        <p className="notice notice--error small" role="alert">
          {error}
        </p>
      )}
      {report && <UploadReport report={report} />}
    </article>
  );
}

type FormProps = {
  existing: string[];
  token: string;
  onAdded: () => Promise<void>;
  onAuthError: (error: unknown) => void;
};

function AddCatForm({ existing, token, onAdded, onAuthError }: FormProps) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<EnrollResponse | null>(null);
  const [error, setError] = useState("");
  const previews = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => previews.forEach((url) => URL.revokeObjectURL(url)), [previews]);

  const validName = CAT_NAME.test(name);
  const taken = existing.includes(name);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setReport(null);
    try {
      const result = await addPhotos(name, await Promise.all(files.map((f) => prepareImage(f))), token);
      setReport(result);
      if (result.added > 0) {
        setName("");
        setFiles([]);
      }
      await onAdded();
    } catch (e) {
      onAuthError(e);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card add-cat" onSubmit={submit}>
      <h2 className="card__title">Add a cat</h2>
      <p className="muted small">
        5–10 photos where only this cat is visible work well: different poses, rooms and light.
      </p>
      <label htmlFor="cat-name" className="field-label">
        Name
      </label>
      <input
        id="cat-name"
        value={name}
        onChange={(e) => setName(toCatName(e.target.value))}
        placeholder="luna"
        autoComplete="off"
        aria-describedby="cat-name-hint"
      />
      <p id="cat-name-hint" className={`small ${name && !validName ? "error-text" : "muted"}`}>
        {taken
          ? `${displayName(name)} is already here: the photos will be added to it.`
          : "Lowercase letters, digits, - and _."}
      </p>

      <DropZone onFiles={(more) => setFiles((current) => [...current, ...more])} multiple disabled={busy}>
        <span className="dropzone__icon" aria-hidden="true">🖼️</span>
        <span className="dropzone__title">Choose photos</span>
        <span className="muted small">or drop them here</span>
      </DropZone>

      {files.length > 0 && (
        <ul className="thumbs" aria-label="Chosen photos">
          {files.map((file, i) => (
            <li key={previews[i]} className="thumbs__item">
              <img src={previews[i]} alt={file.name} />
              <button
                type="button"
                className="thumbs__remove"
                aria-label={`Remove ${file.name}`}
                onClick={() => setFiles((current) => current.filter((_, j) => j !== i))}
                disabled={busy}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      <button type="submit" className="button" disabled={busy || !validName || files.length === 0}>
        {busy
          ? "Uploading…"
          : files.length > 0
            ? `Add ${files.length} photo${files.length === 1 ? "" : "s"}${validName ? ` of ${displayName(name)}` : ""}`
            : "Add photos"}
      </button>
      {error && (
        <p className="notice notice--error small" role="alert">
          {error}
        </p>
      )}
      {report && <UploadReport report={report} />}
    </form>
  );
}

function UploadReport({ report }: { report: EnrollResponse }) {
  const parts = [`${report.added} photo${report.added === 1 ? "" : "s"} added`];
  if (report.duplicates) parts.push(`${report.duplicates} already there`);
  return (
    <div className={`notice ${report.added > 0 ? "notice--ok" : "notice--warning"} small`} role="status">
      <p>
        {displayName(report.name)}: {parts.join(", ")}. {report.photos} in total.
      </p>
      {report.rejected.length > 0 && (
        <ul className="rejected">
          {report.rejected.map((r, i) => (
            <li key={i}>
              <strong>{r.file || "photo"}</strong>: {r.reason}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
