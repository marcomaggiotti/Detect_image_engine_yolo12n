import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import type { AssistantMessage } from "../agent/types";
import { Detections, toJson } from "./Detections";

beforeAll(() => {
  URL.createObjectURL = vi.fn(() => "blob:x");
  URL.revokeObjectURL = vi.fn();
});
afterEach(cleanup);

const original = new File(["x"], "IMG_2041.jpg", { type: "image/jpeg" });

// A 4000x3000 photo, uploaded as 1600x1200 (scale 2.5).
const checked: AssistantMessage = {
  id: 7,
  role: "assistant",
  steps: [],
  check: {
    photo: { file: original, url: "blob:uploaded" },
    original: { file: original, url: "blob:original" },
    scale: { x: 2.5, y: 2.5 },
    checkedAt: Date.UTC(2026, 9, 17, 10, 42),
    result: {
      image: { width: 1600, height: 1200 },
      findings: [
        { box: { x1: 12.4, y1: 30.1, x2: 320.8, y2: 488 }, title: "Luna", known: true, confidence: 0.91, similarity: 0.96, closest: "luna" },
        { box: { x1: 900, y1: 100, x2: 1600, y2: 1199.9 }, title: "Unknown cat", known: false, confidence: 0.8, similarity: 0.6, closest: "milo" },
      ],
      ms: 400,
    },
  },
};

describe("Detections", () => {
  it("lists each cat with its box in pixels of the original photo", () => {
    render(<Detections messages={[checked]} onCheck={() => {}} />);
    expect(screen.getByText(/2 cats in 1 photo/)).toBeTruthy();
    expect(screen.getByText(/4000 × 3000 px/)).toBeTruthy();

    const [luna, unknown] = screen.getAllByRole("listitem");
    const coords = (item: HTMLElement) => within(item).getAllByRole("definition").map((d) => d.textContent);
    expect(coords(luna)).toEqual(["31", "75", "802", "1220", "771 × 1145"]);
    expect(coords(unknown)).toEqual(["2250", "250", "4000", "3000", "1750 × 2750"]);
    expect(within(unknown).getByText("Unknown cat")).toBeTruthy();
  });

  it("says when nothing has been checked", () => {
    render(<Detections messages={[]} onCheck={() => {}} />);
    expect(screen.getByText(/No photos checked yet/)).toBeTruthy();
  });

  it("exports the boxes as JSON in original pixels", () => {
    const json = JSON.parse(
      toJson("IMG_2041.jpg", { width: 4000, height: 3000 }, [
        { ...checked.check!.result.findings[0], index: 1, originalBox: { x1: 31, y1: 75.3, x2: 802, y2: 1220 } },
      ]),
    );
    expect(json).toEqual({
      file: "IMG_2041.jpg",
      width: 4000,
      height: 3000,
      detections: [
        {
          index: 1,
          label: "cat",
          name: "luna",
          confidence: 0.91,
          similarity: 0.96,
          closest: "luna",
          box: { x1: 31, y1: 75.3, x2: 802, y2: 1220 },
          width: 771,
          height: 1144.7,
        },
      ],
    });
  });
});
