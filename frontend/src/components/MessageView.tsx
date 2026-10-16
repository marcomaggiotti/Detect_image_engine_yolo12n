import { useState } from "react";
import type { Action, AssistantMessage, Message, Photo } from "../agent/types";
import { catPhotoUrl, type CatsResponse, type EnrollResponse } from "../api";
import { displayName, percent, type Finding } from "../results";
import { PhotoBoxes } from "./PhotoBoxes";
import { Steps } from "./Steps";

export type Handlers = {
  onAction: (messageId: number, action: Action) => void;
  onName: (messageId: number, name: string, photo: Photo) => Promise<string | null>;
  onToken: (messageId: number, token: string) => void;
};

type ViewProps = { message: Message; handlers: Handlers; catsVersion: number; known: Set<string> };

export function MessageView({ message, handlers, catsVersion, known }: ViewProps) {
  if (message.role === "user") {
    return (
      <div className="msg msg--user">
        <div className="bubble bubble--user">
          {message.photos.length > 0 && (
            <div className={`sent-photos sent-photos--${Math.min(message.photos.length, 3)}`}>
              {message.photos.slice(0, 6).map((p) => (
                <img key={p.url} src={p.url} alt={p.file.name} onError={(e) => (e.currentTarget.style.visibility = "hidden")} />
              ))}
              {message.photos.length > 6 && <span className="sent-photos__more">+{message.photos.length - 6}</span>}
            </div>
          )}
          {message.text && <p>{message.text}</p>}
        </div>
      </div>
    );
  }
  return <AssistantView message={message} handlers={handlers} catsVersion={catsVersion} known={known} />;
}

function AssistantView({ message, handlers, catsVersion, known }: ViewProps & { message: AssistantMessage }) {
  const working = message.steps.some((s) => s.status === "running");
  return (
    <div className="msg msg--assistant" aria-busy={working}>
      <div className="msg__avatar" aria-hidden="true">
        🐱
      </div>
      <div className="msg__body">
        <Steps steps={message.steps} />
        {message.text && <p className={`msg__text${message.failed ? " msg__text--failed" : ""}`}>{message.text}</p>}
        {message.check && (
          <>
            <PhotoBoxes url={message.check.photo.url} result={message.check.result} />
            <Findings findings={message.check.result.findings} />
          </>
        )}
        {message.report && <Rejected report={message.report} />}
        {message.cats && (
          <CatList
            cats={message.cats}
            version={catsVersion}
            known={known}
            onForget={(name) => handlers.onAction(message.id, { kind: "forget-ask", label: "", name })}
          />
        )}
        {message.note && <p className="msg__note">{message.note}</p>}
        {message.askName && <NameForm onSubmit={(name) => handlers.onName(message.id, name, message.askName!)} />}
        {message.askToken && <TokenForm onSubmit={(token) => handlers.onToken(message.id, token)} />}
        {message.actions && message.actions.length > 0 && (
          <div className="chips">
            {message.actions.map((action, i) => (
              <button key={i} type="button" className="chip" onClick={() => handlers.onAction(message.id, action)}>
                {action.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Findings({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) return null;
  return (
    <ul className="findings">
      {findings.map((f, i) => (
        <li key={i}>
          <span className={`dot ${f.known === false ? "dot--unknown" : "dot--cat"}`} aria-hidden="true" />
          <strong>{f.title}</strong>{" "}
          <span className="muted">
            {f.similarity !== undefined &&
              (f.known ? `${percent(f.similarity)} like its photos · ` : `looks most like ${displayName(f.closest!)} (${percent(f.similarity)}) · `)}
            {percent(f.confidence)} sure it's a cat
          </span>
        </li>
      ))}
    </ul>
  );
}

function Rejected({ report }: { report: EnrollResponse }) {
  if (report.rejected.length === 0) return null;
  return (
    <ul className="rejected">
      {report.rejected.map((r, i) => (
        <li key={i}>
          <strong>{r.file || "photo"}</strong>: {r.reason}
        </li>
      ))}
    </ul>
  );
}

type CatListProps = { cats: CatsResponse; version: number; known: Set<string>; onForget: (name: string) => void };

/** Cats as they were when listed; ones forgotten since are shown as such. */
function CatList({ cats, version, known, onForget }: CatListProps) {
  return (
    <ul className="cat-list">
      {cats.cats.map((cat) =>
        known.has(cat.name) ? (
          <li key={cat.name} className="cat-list__item">
            <Avatar name={cat.name} version={version} />
            <span>
              <strong>{displayName(cat.name)}</strong>
              <span className="muted small"> · {cat.photos} photo{cat.photos === 1 ? "" : "s"}</span>
            </span>
            <button type="button" className="link small cat-list__forget" onClick={() => onForget(cat.name)}>
              Forget
            </button>
          </li>
        ) : (
          <li key={cat.name} className="cat-list__item cat-list__item--forgotten">
            <span className="avatar avatar--initial" aria-hidden="true">
              {cat.name.charAt(0).toUpperCase()}
            </span>
            <span>
              <s>{displayName(cat.name)}</s> <span className="muted small">· forgotten</span>
            </span>
          </li>
        ),
      )}
    </ul>
  );
}

function Avatar({ name, version }: { name: string; version: number }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <span className="avatar avatar--initial" aria-hidden="true">
        {name.charAt(0).toUpperCase()}
      </span>
    );
  }
  return <img className="avatar" src={catPhotoUrl(name, version)} alt="" onError={() => setFailed(true)} />;
}

function NameForm({ onSubmit }: { onSubmit: (name: string) => Promise<string | null> }) {
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  return (
    <form
      className="inline-form"
      onSubmit={async (e) => {
        e.preventDefault();
        setError((await onSubmit(name)) ?? "");
      }}
    >
      <label htmlFor="new-cat-name" className="small">
        What's this cat's name?
      </label>
      <div className="row">
        <input id="new-cat-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Luna" autoComplete="off" autoFocus />
        <button type="submit" className="button" disabled={!name.trim()}>
          Save
        </button>
      </div>
      {error && <p className="small error-text">{error}</p>}
    </form>
  );
}

function TokenForm({ onSubmit }: { onSubmit: (token: string) => void }) {
  const [token, setToken] = useState("");
  return (
    <form
      className="inline-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (token.trim()) onSubmit(token.trim());
      }}
    >
      <label htmlFor="admin-token" className="small">
        Admin token (<code>DETECT_ADMIN_TOKEN</code>; on Render: your service → Environment). I'll keep it on this device.
      </label>
      <div className="row">
        <input id="admin-token" type="password" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" autoFocus />
        <button type="submit" className="button" disabled={!token.trim()}>
          Continue
        </button>
      </div>
    </form>
  );
}
