import { useEffect, useRef, useState } from "react";
import type { useAgent } from "../agent/useAgent";
import { Composer, type ComposerHandle } from "./Composer";
import { MessageView, type Handlers } from "./MessageView";

type Agent = ReturnType<typeof useAgent>;

/** The conversation, with drag and drop of photos anywhere on the page. */
export function Chat({ agent }: { agent: Agent }) {
  const composer = useRef<ComposerHandle>(null);
  const end = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const [catsVersion, setCatsVersion] = useState(0);

  // New cat photos after the list of cats changes.
  useEffect(() => setCatsVersion((v) => v + 1), [agent.cats]);

  // Follow the conversation as it grows or steps update.
  useEffect(() => {
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    end.current?.scrollIntoView?.({ behavior: reduced ? "auto" : "smooth", block: "end" });
  }, [agent.messages]);

  const known = new Set(agent.cats?.cats.map((c) => c.name) ?? []);

  const handlers: Handlers = {
    onAction: (id, action) => {
      if (action.kind === "check-another") composer.current?.pickPhotos();
      agent.act(id, action);
    },
    onName: agent.submitName,
    onToken: agent.submitToken,
  };

  return (
    <div
      className="chat"
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("Files")) {
          e.preventDefault();
          setDragging(true);
        }
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        composer.current?.addFiles(Array.from(e.dataTransfer.files));
      }}
    >
      <div className="chat__log" role="log" aria-live="polite" aria-label="Conversation">
        {agent.messages.map((m) => (
          <MessageView key={m.id} message={m} handlers={handlers} catsVersion={catsVersion} known={known} />
        ))}
        <div ref={end} />
      </div>
      <Composer ref={composer} onSend={agent.send} knowsCats={(agent.cats?.cats.length ?? 0) > 0} />
      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          Drop photos to send them
        </div>
      )}
    </div>
  );
}
