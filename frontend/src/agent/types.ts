import type { CatsResponse, EnrollResponse } from "../api";
import type { CheckResult, Scale } from "../results";

export type StepStatus = "running" | "done" | "failed";

/** One thing the assistant does, shown as it happens. */
export type Step = { label: string; status: StepStatus; detail?: string };

export type Photo = { file: File; url: string };

/** Buttons offered under an assistant message. */
export type Action =
  | { kind: "check-another"; label: string }
  | { kind: "save-photo"; label: string; name: string; photo: Photo }
  | { kind: "save-photo-new"; label: string; photo: Photo }
  | { kind: "show-cats"; label: string }
  | { kind: "how-to-add"; label: string }
  | { kind: "forget-ask"; label: string; name: string }
  | { kind: "forget-confirm"; label: string; name: string }
  | { kind: "dismiss"; label: string };

export type UserMessage = { id: number; role: "user"; text: string; photos: Photo[] };

export type AssistantMessage = {
  id: number;
  role: "assistant";
  steps: Step[];
  text?: string;
  /** Smaller text under the main answer. */
  note?: string;
  /** A checked photo: ``photo`` is what was uploaded, ``original`` what was chosen. */
  check?: { photo: Photo; original: Photo; scale: Scale; result: CheckResult; checkedAt: number };
  cats?: CatsResponse;
  report?: EnrollResponse;
  actions?: Action[];
  /** Asks for the admin token inline. */
  askToken?: boolean;
  /** Asks for a new cat's name inline, to save this photo. */
  askName?: Photo;
  failed?: boolean;
  /** A photo check still in progress. */
  checking?: boolean;
};

export type Message = UserMessage | AssistantMessage;
