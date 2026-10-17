import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import * as realApi from "../api";
import { ApiError, CAT_NAME, toCatName, type CatsResponse } from "../api";
import { prepareImage as realPrepareImage, type PreparedImage } from "../image";
import { displayName, fromDetect, fromIdentify, listNames, percent, summarize, type CheckResult } from "../results";
import { parseIntent } from "./intent";
import type { Action, AssistantMessage, Message, Photo, Step } from "./types";

/** What the assistant can call; replaced by fakes in tests. */
export type AgentApi = Pick<typeof realApi, "detectCats" | "identify" | "listCats" | "addPhotos" | "deleteCat"> & {
  prepareImage: (file: File) => Promise<PreparedImage>;
};

const defaultApi: AgentApi = { ...realApi, prepareImage: realPrepareImage };

type Update = Partial<AssistantMessage> | ((message: AssistantMessage) => Partial<AssistantMessage>);
type Event = { type: "add"; message: Message } | { type: "update"; id: number; update: Update };

function reducer(messages: Message[], event: Event): Message[] {
  if (event.type === "add") return [...messages, event.message];
  return messages.map((m) => {
    if (m.id !== event.id || m.role !== "assistant") return m;
    const patch = typeof event.update === "function" ? event.update(m) : event.update;
    return { ...m, ...patch };
  });
}

const CHECK_ANOTHER: Action = { kind: "check-another", label: "Check another photo" };
const SHOW_CATS: Action = { kind: "show-cats", label: "Show my cats" };
const HOW_TO_ADD: Action = { kind: "how-to-add", label: "How do I add a cat?" };

export const HELP_TEXT = [
  "Here's what I can do:",
  "• Send me a photo: I'll tell you if there's a cat in it, and which of your cats it is.",
  "• Send photos of one of your cats and say “this is Luna”: I'll remember Luna.",
  "• Ask me to “show my cats” or to “forget Luna”.",
  "I understand these few requests and photos, nothing more: I'm not a chatbot.",
].join("\n");

function kb(file: File): string {
  return `${Math.max(1, Math.round(file.size / 1024))} KB`;
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

type Options = { token: string; setToken: (token: string) => void; api?: AgentApi };

export function useAgent({ token, setToken, api = defaultApi }: Options) {
  const [messages, dispatch] = useReducer(reducer, []);
  const [cats, setCats] = useState<CatsResponse | null | undefined>(undefined);
  const nextId = useRef(1);
  const tokenRef = useRef(token);
  const catsRef = useRef(cats);
  const retry = useRef<(() => Promise<void>) | null>(null);
  tokenRef.current = token;
  catsRef.current = cats;

  const add = useCallback((message: Omit<UserMessage, "id"> | Omit<AssistantMessage, "id">) => {
    const id = nextId.current++;
    dispatch({ type: "add", message: { ...message, id } as Message });
    return id;
  }, []);
  const update = useCallback((id: number, change: Update) => dispatch({ type: "update", id, update: change }), []);
  const say = useCallback((text: string, extra: Partial<AssistantMessage> = {}) => add({ role: "assistant", steps: [], text, ...extra }), [add]);
  const setStep = useCallback(
    (id: number, index: number, step: Step) =>
      update(id, (m) => {
        const steps = [...m.steps];
        steps[index] = step;
        return { steps };
      }),
    [update],
  );

  const refreshCats = useCallback(async () => {
    const latest = await api.listCats();
    setCats(latest);
    catsRef.current = latest;
    return latest;
  }, [api]);

  const names = () => catsRef.current?.cats.map((c) => c.name) ?? [];

  // ---- What the assistant does ------------------------------------------

  /** A change the server refused for lack of the admin token: ask for it, then carry on. */
  function needToken(id: number, again: () => Promise<void>) {
    retry.current = again;
    update(id, {
      failed: true,
      askToken: true,
      text: tokenRef.current
        ? "That admin token didn't work. Paste the right one and I'll try again."
        : "I need the admin token to change your cats. Paste it below and I'll carry on.",
    });
  }

  async function checkPhoto(original: Photo) {
    const id = add({ role: "assistant", steps: [{ label: "Preparing the photo", status: "running" }], checking: true });
    const prepared = await api.prepareImage(original.file);
    const file = prepared.file;
    const photo = file === original.file ? original : { file, url: URL.createObjectURL(file) };
    setStep(id, 0, { label: "Preparing the photo", status: "done", detail: file === original.file ? kb(file) : `resized, ${kb(file)}` });

    const known = catsRef.current?.cats ?? [];
    const withNames = known.length > 0;
    setStep(id, 1, {
      label: withNames ? `Looking for cats and comparing them with ${listNames(known.map((c) => displayName(c.name)))}` : "Looking for cats",
      status: "running",
    });
    let result: CheckResult;
    try {
      result = withNames ? fromIdentify(await api.identify(file)) : fromDetect(await api.detectCats(file));
    } catch (error) {
      setStep(id, 1, { label: "Looking for cats", status: "failed", detail: "error" });
      update(id, { failed: true, checking: false, text: (error as Error).message, actions: [CHECK_ANOTHER] });
      return;
    }

    const found = result.findings.length;
    const steps: Step[] = [
      { label: "Preparing the photo", status: "done", detail: file === original.file ? kb(file) : `resized, ${kb(file)}` },
      { label: "Looking for cats", status: "done", detail: found ? `${plural(found, "cat")} found` : "none found" },
    ];
    if (withNames && found) {
      const recognised = result.findings.filter((f) => f.known).length;
      steps.push({
        label: `Comparing with your ${plural(known.length, "cat")}`,
        status: "done",
        detail: recognised ? `${recognised} recognised` : "no match",
      });
    }

    const actions: Action[] = [];
    let note: string | undefined;
    if (found === 1 && catsRef.current) {
      const [finding] = result.findings;
      if (finding.known) {
        const name = finding.closest!;
        actions.push({ kind: "save-photo", label: `Add to ${displayName(name)}'s photos`, name, photo });
        actions.push({ kind: "save-photo-new", label: "No, it's another cat…", photo });
      } else {
        const byLikeness = finding.closest ? [finding.closest, ...names().filter((n) => n !== finding.closest)] : names();
        for (const name of byLikeness.slice(0, 3)) {
          actions.push({ kind: "save-photo", label: `It's ${displayName(name)}`, name, photo });
        }
        actions.push({ kind: "save-photo-new", label: withNames ? "It's a new cat…" : "Remember this cat…", photo });
      }
    } else if (found > 1 && catsRef.current) {
      note = "To teach me one of these cats, send a photo where it's alone.";
    }
    actions.push(CHECK_ANOTHER);

    update(id, {
      steps,
      checking: false,
      text: summarize(result.findings).headline,
      check: { photo, original, scale: { x: prepared.scaleX, y: prepared.scaleY }, result, checkedAt: Date.now() },
      note: [note, `Checked in ${(result.ms / 1000).toFixed(1)} s on the server.`].filter(Boolean).join(" "),
      actions,
    });
  }

  async function savePhotos(name: string, photos: Photo[]) {
    if (catsRef.current === null) {
      say("This server can't learn your cats' names, so I can't save them. I can still tell you if there's a cat in a photo.");
      return;
    }
    const wasKnown = names().includes(name);
    const who = displayName(name);
    const first: Step = { label: `Preparing ${photos.length === 1 ? "the photo" : `${photos.length} photos`}`, status: "running" };
    const id = add({ role: "assistant", steps: [first] });
    const files = (await Promise.all(photos.map((p) => api.prepareImage(p.file)))).map((p) => p.file);
    setStep(id, 0, { ...first, status: "done" });
    const finding: Step = { label: `Finding the cat in ${photos.length === 1 ? "the photo" : "each photo"} and saving ${who}`, status: "running" };
    setStep(id, 1, finding);

    let report: realApi.EnrollResponse;
    try {
      report = await api.addPhotos(name, files, tokenRef.current);
    } catch (error) {
      setStep(id, 1, { ...finding, status: "failed", detail: error instanceof ApiError && error.status === 401 ? "admin token needed" : "error" });
      if (error instanceof ApiError && error.status === 401) needToken(id, () => savePhotos(name, photos));
      else update(id, { failed: true, text: (error as Error).message });
      return;
    }

    const usable = report.added + report.duplicates;
    const steps: Step[] = [
      { ...first, status: "done" },
      {
        label: `Finding the cat in ${photos.length === 1 ? "the photo" : "each photo"}`,
        status: usable ? "done" : "failed",
        detail: `${usable} of ${photos.length} usable`,
      },
    ];
    if (report.added) steps.push({ label: `Saving ${who}`, status: "done", detail: `${plural(report.photos, "photo")} in total` });

    let text: string;
    if (report.added === 0) {
      text = usable ? `I already had ${photos.length === 1 ? "this photo" : "these photos"} of ${who}.` : `I couldn't use ${photos.length === 1 ? "this photo" : "these photos"}.`;
    } else if (!wasKnown) {
      text = `Got it! I'll recognise ${who} from now on.`;
    } else {
      text = `Thanks! ${who} now has ${plural(report.photos, "photo")}.`;
    }
    const tip = !wasKnown && report.photos < 5 && report.added ? `A few more photos of ${who} in different poses and light will help me recognise them.` : undefined;
    update(id, { steps, text, note: tip, report, actions: [CHECK_ANOTHER, SHOW_CATS] });
    await refreshCats().catch(() => undefined);
  }

  async function showCats() {
    let latest: CatsResponse | null;
    try {
      latest = await refreshCats();
    } catch (error) {
      say((error as Error).message, { failed: true });
      return;
    }
    if (latest === null) {
      say("This server can't learn your cats' names. I can still tell you if there's a cat in a photo.");
    } else if (latest.cats.length === 0) {
      say("I don't know any of your cats yet.", { actions: [HOW_TO_ADD] });
    } else {
      say(`I know ${plural(latest.cats.length, "cat")}:`, {
        cats: latest,
        note: `I name a cat when it looks at least ${percent(latest.threshold)} like its photos.${latest.persistent ? "" : " This server keeps them in memory only: I'll forget them when it restarts."}`,
      });
    }
  }

  function askForget(name: string) {
    const cat = catsRef.current?.cats.find((c) => c.name === name);
    if (!cat) {
      say(`I don't know a cat called ${displayName(name)}.`, { actions: [SHOW_CATS] });
      return;
    }
    say(`Forget ${displayName(name)} and ${cat.photos === 1 ? "their photo" : `all ${cat.photos} photos`}?`, {
      actions: [
        { kind: "forget-confirm", label: `Yes, forget ${displayName(name)}`, name },
        { kind: "dismiss", label: "Keep" },
      ],
    });
  }

  async function forget(name: string) {
    const step: Step = { label: `Forgetting ${displayName(name)}`, status: "running" };
    const id = add({ role: "assistant", steps: [step] });
    try {
      await api.deleteCat(name, tokenRef.current);
    } catch (error) {
      setStep(id, 0, { ...step, status: "failed" });
      if (error instanceof ApiError && error.status === 401) needToken(id, () => forget(name));
      else update(id, { failed: true, text: (error as Error).message });
      return;
    }
    setStep(id, 0, { ...step, status: "done" });
    update(id, { text: `Done, I've forgotten ${displayName(name)}.` });
    await refreshCats().catch(() => undefined);
  }

  // ---- What the person does ----------------------------------------------

  async function send(text: string, files: File[]) {
    const photos = files.map((file) => ({ file, url: URL.createObjectURL(file) }));
    add({ role: "user", text: text.trim(), photos });
    const intent = parseIntent(text, photos.length > 0, names());
    switch (intent.kind) {
      case "check":
        for (const photo of photos) await checkPhoto(photo);
        return;
      case "save":
        return savePhotos(intent.name, photos);
      case "list":
        return showCats();
      case "forget":
        return askForget(intent.name);
      case "help":
        say(HELP_TEXT, { actions: [SHOW_CATS] });
        return;
      default:
        say("Sorry, I didn't get that. Send me a photo, or try one of these:", { actions: [SHOW_CATS, HOW_TO_ADD] });
    }
  }

  async function act(messageId: number, action: Action) {
    update(messageId, { actions: undefined }); // each set of buttons is used once
    switch (action.kind) {
      case "save-photo":
        add({ role: "user", text: `It's ${displayName(action.name)}`, photos: [] });
        return savePhotos(action.name, [action.photo]);
      case "save-photo-new":
        update(messageId, { askName: action.photo });
        return;
      case "show-cats":
        add({ role: "user", text: "Show my cats", photos: [] });
        return showCats();
      case "how-to-add":
        add({ role: "user", text: "How do I add a cat?", photos: [] });
        say(HELP_TEXT, { actions: [SHOW_CATS] });
        return;
      case "forget-ask":
        add({ role: "user", text: `Forget ${displayName(action.name)}`, photos: [] });
        return askForget(action.name);
      case "forget-confirm":
        add({ role: "user", text: action.label, photos: [] });
        return forget(action.name);
      case "dismiss":
        add({ role: "user", text: action.label, photos: [] });
        say("OK, nothing changed.");
        return;
      case "check-another":
        return; // the chat opens the photo picker
    }
  }

  /** The name typed for a new cat, from a photo the assistant just checked. */
  async function submitName(messageId: number, typed: string, photo: Photo): Promise<string | null> {
    const name = toCatName(typed);
    if (!CAT_NAME.test(name)) return "Use letters, digits, - or _.";
    update(messageId, { askName: undefined });
    add({ role: "user", text: `This is ${displayName(name)}`, photos: [] });
    await savePhotos(name, [photo]);
    return null;
  }

  async function submitToken(messageId: number, value: string) {
    setToken(value);
    tokenRef.current = value;
    update(messageId, { askToken: false });
    add({ role: "user", text: "🔑 (admin token)", photos: [] });
    const again = retry.current;
    retry.current = null;
    if (again) await again();
  }

  // Greeting, once the cats are known.
  const greeted = useRef(false);
  useEffect(() => {
    if (greeted.current) return;
    greeted.current = true;
    refreshCats()
      .then((latest) => {
        const known = latest?.cats.map((c) => displayName(c.name)) ?? [];
        if (latest === null) {
          say("Hi! Send me a photo and I'll tell you if there's a cat in it.");
        } else if (known.length === 0) {
          say("Hi! Send me a photo and I'll tell you if there's a cat in it.", {
            note: "Want me to recognise your own cats? Send me a few photos of one and tell me its name, like “this is Luna”.",
            actions: [HOW_TO_ADD],
          });
        } else {
          say(`Hi! I know ${listNames(known)}. Send me a photo and I'll tell you who's in it.`, { actions: [SHOW_CATS, HOW_TO_ADD] });
        }
      })
      .catch((error: Error) => say(`I can't reach the server right now (${error.message}). Try reloading the page.`, { failed: true }));
  }, [refreshCats, say]);

  return { messages, cats, send, act, submitName, submitToken };
}

type UserMessage = Extract<Message, { role: "user" }>;
