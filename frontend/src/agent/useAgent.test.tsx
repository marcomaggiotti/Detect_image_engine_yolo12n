import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { ApiError, type CatsResponse } from "../api";
import type { AssistantMessage, Message } from "./types";
import { useAgent, type AgentApi } from "./useAgent";

beforeAll(() => {
  URL.createObjectURL = vi.fn(() => "blob:photo");
  URL.revokeObjectURL = vi.fn();
});

const box = { x1: 10, y1: 10, x2: 50, y2: 50 };
const photo = () => new File(["x"], "cat.jpg", { type: "image/jpeg" });

function fakeApi(cats: CatsResponse["cats"] = [{ name: "luna", photos: 5 }]): AgentApi & { known: CatsResponse["cats"] } {
  const known = [...cats];
  return {
    known,
    prepareImage: async (file) => ({ file, scaleX: 1, scaleY: 1 }),
    listCats: vi.fn(async () => ({ cats: [...known], threshold: 0.8, persistent: true })),
    detectCats: vi.fn(async () => ({ image: { width: 100, height: 100 }, found: true, counts: { cat: 1 }, detections: [{ label: "cat", class_id: 15, confidence: 0.9, box }], inference_ms: 200 })),
    identify: vi.fn(async () => ({
      image: { width: 100, height: 100 },
      names: [],
      cats: [{ name: null, closest: "luna", similarity: 0.6, confidence: 0.9, box }],
      inference_ms: 400,
    })),
    addPhotos: vi.fn(async (name: string, files: File[]) => {
      const cat = known.find((c) => c.name === name) ?? (known.push({ name, photos: 0 }), known[known.length - 1]);
      cat.photos += files.length;
      return { name, added: files.length, duplicates: 0, rejected: [], photos: cat.photos };
    }),
    deleteCat: vi.fn(async (name: string) => {
      known.splice(known.findIndex((c) => c.name === name), 1);
    }),
  };
}

function setup(api: AgentApi, token = "") {
  const setToken = vi.fn();
  const hook = renderHook(() => useAgent({ token, setToken, api }));
  return { ...hook, setToken };
}

const assistant = (messages: Message[]) => messages.filter((m): m is AssistantMessage => m.role === "assistant");
const last = (messages: Message[]) => {
  const replies = assistant(messages);
  return replies[replies.length - 1];
};

describe("useAgent", () => {
  it("greets with the cats it knows", async () => {
    const { result } = setup(fakeApi([{ name: "luna", photos: 5 }, { name: "milo", photos: 3 }]));
    await waitFor(() => expect(last(result.current.messages).text).toBe("Hi! I know Luna and Milo. Send me a photo and I'll tell you who's in it."));
  });

  it("checks a photo step by step and offers to name an unknown cat", async () => {
    const api = fakeApi();
    const { result } = setup(api);
    await waitFor(() => expect(result.current.cats).toBeDefined());

    await act(() => result.current.send("", [photo()]));

    const reply = last(result.current.messages);
    expect(api.identify).toHaveBeenCalledOnce();
    expect(reply.text).toBe("A cat, but not one of yours");
    expect(reply.steps.map((s) => [s.label, s.status, s.detail])).toEqual([
      ["Preparing the photo", "done", "1 KB"],
      ["Looking for cats", "done", "1 cat found"],
      ["Comparing with your 1 cat", "done", "no match"],
    ]);
    expect(reply.actions?.map((a) => a.label)).toEqual(["It's Luna", "It's a new cat…", "Check another photo"]);
  });

  it("asks for the admin token when needed, then carries on by itself", async () => {
    const api = fakeApi();
    let tokenOk = false;
    const save = api.addPhotos;
    api.addPhotos = vi.fn(async (...args: Parameters<AgentApi["addPhotos"]>) => {
      if (!tokenOk) throw new ApiError(401, "Missing or wrong admin token");
      return save(...args);
    });
    const { result, setToken } = setup(api);
    await waitFor(() => expect(result.current.cats).toBeDefined());

    await act(() => result.current.send("this is Milo, remember him", [photo()]));
    let reply = last(result.current.messages);
    expect(reply.askToken).toBe(true);
    expect(reply.text).toContain("I need the admin token");

    tokenOk = true;
    await act(() => result.current.submitToken(reply.id, "s3cret"));
    expect(setToken).toHaveBeenCalledWith("s3cret");
    reply = last(result.current.messages);
    expect(reply.text).toBe("Got it! I'll recognise Milo from now on.");
    expect(api.addPhotos).toHaveBeenLastCalledWith("milo", expect.any(Array), "s3cret");
    expect(result.current.cats?.cats.map((c) => c.name)).toEqual(["luna", "milo"]);
  });

  it("saves a checked photo from a button", async () => {
    const api = fakeApi();
    const { result } = setup(api, "s3cret");
    await waitFor(() => expect(result.current.cats).toBeDefined());
    await act(() => result.current.send("", [photo()]));

    const reply = last(result.current.messages);
    await act(() => result.current.act(reply.id, reply.actions![0]));
    expect(api.addPhotos).toHaveBeenCalledWith("luna", [expect.any(File)], "s3cret");
    expect(last(result.current.messages).text).toBe("Thanks! Luna now has 6 photos.");
  });

  it("asks before forgetting a cat", async () => {
    const api = fakeApi();
    const { result } = setup(api, "s3cret");
    await waitFor(() => expect(result.current.cats).toBeDefined());

    await act(() => result.current.send("forget luna", []));
    const question = last(result.current.messages);
    expect(question.text).toBe("Forget Luna and all 5 photos?");
    expect(api.deleteCat).not.toHaveBeenCalled();

    await act(() => result.current.act(question.id, question.actions![0]));
    expect(api.deleteCat).toHaveBeenCalledWith("luna", "s3cret");
    expect(last(result.current.messages).text).toBe("Done, I've forgotten Luna.");
  });

  it("says when it doesn't understand", async () => {
    const { result } = setup(fakeApi());
    await waitFor(() => expect(result.current.cats).toBeDefined());
    await act(() => result.current.send("what's the weather?", []));
    expect(last(result.current.messages).actions?.map((a) => a.kind)).toEqual(["show-cats", "how-to-add"]);
  });
});
