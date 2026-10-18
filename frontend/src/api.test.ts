import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, CAT_NAME, MAX_PHOTOS_PER_UPLOAD, addPhotos, deleteCat, listCats, toCatName } from "./api";

function respond(status: number, body?: unknown) {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("api", () => {
  it("uploads photos in batches the server accepts and merges the results", async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const count = (init.body as FormData).getAll("images").length;
      return respond(200, { name: "luna", added: count - 1, duplicates: 1, rejected: [], photos: 99 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const files = Array.from({ length: MAX_PHOTOS_PER_UPLOAD + 5 }, (_, i) => new File(["x"], `${i}.jpg`));

    const result = await addPhotos("luna", files, "s3cret");

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe("/v1/cats/luna/photos");
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: "Bearer s3cret" });
    expect(result).toEqual({ name: "luna", added: 23, duplicates: 2, rejected: [], photos: 99 });
  });

  it("turns error responses into readable errors", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => respond(401, { detail: "Missing or wrong admin token" })));
    await expect(deleteCat("luna", "")).rejects.toEqual(new ApiError(401, "Missing or wrong admin token"));
  });

  it("reports when the server is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    await expect(listCats()).rejects.toMatchObject({ status: 0 });
  });

  it("treats 503 on /v1/cats as identification disabled", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => respond(503, { detail: "not configured" })));
    await expect(listCats()).resolves.toBeNull();
  });

  it("makes valid cat names from what people type", () => {
    expect(toCatName("Mr Whiskers")).toBe("mr-whiskers");
    expect(toCatName("Chloé!")).toBe("chloe");
    expect(CAT_NAME.test(toCatName("Mr Whiskers"))).toBe(true);
    expect(CAT_NAME.test("-luna")).toBe(false);
  });
});
