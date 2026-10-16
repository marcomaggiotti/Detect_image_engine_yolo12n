import type { Step } from "../agent/types";

const ICON = { running: null, done: "✓", failed: "✕" } as const;

/** What the assistant is doing or did, one line per step. */
export function Steps({ steps }: { steps: Step[] }) {
  if (steps.length === 0) return null;
  return (
    <ol className="steps" aria-label="Steps">
      {steps.map((step, i) => (
        <li key={i} className={`steps__item steps__item--${step.status}`}>
          <span className="steps__icon" aria-hidden="true">
            {ICON[step.status] ?? <span className="spinner spinner--small" />}
          </span>
          <span className="steps__label">
            {step.label}
            {step.status === "running" && "…"}
          </span>
          {step.detail && <span className="steps__detail">{step.detail}</span>}
          <span className="visually-hidden">{step.status === "running" ? "in progress" : step.status}</span>
        </li>
      ))}
    </ol>
  );
}
