import { describe, expect, it } from "vitest";
import { displayName, fromDetect, fromIdentify, summarize, type Finding } from "./results";

const box = { x1: 0, y1: 0, x2: 10, y2: 10 };
const cat = (title: string, known: boolean | null): Finding => ({ box, title, known, confidence: 0.9 });

describe("summarize", () => {
  it.each([
    [[], "No cat in this photo", false],
    [[cat("Cat", null)], "Yes, there's a cat!", true],
    [[cat("Cat", null), cat("Cat", null)], "Yes, 2 cats!", true],
    [[cat("Luna", true)], "It's Luna!", true],
    [[cat("Luna", true), cat("Milo", true)], "Luna and Milo", true],
    [[cat("Luna", true), cat("Luna", true)], "It's Luna!", true],
    [[cat("Luna", true), cat("Unknown cat", false)], "Luna and a cat you haven't added", true],
    [[cat("Unknown cat", false)], "A cat, but not one of yours", true],
    [[cat("Unknown cat", false), cat("Unknown cat", false)], "2 cats, none of them yours", true],
  ] as const)("%j -> %s", (findings, headline, found) => {
    expect(summarize([...findings])).toEqual({ headline, found });
  });
});

describe("results", () => {
  it("maps identify responses to named and unknown cats", () => {
    const result = fromIdentify({
      image: { width: 100, height: 50 },
      names: ["mr-whiskers"],
      cats: [
        { name: "mr-whiskers", closest: "mr-whiskers", similarity: 0.95, confidence: 0.9, box },
        { name: null, closest: "luna", similarity: 0.6, confidence: 0.8, box },
      ],
      inference_ms: 400,
    });
    expect(result.findings.map((f) => [f.title, f.known])).toEqual([
      ["Mr whiskers", true],
      ["Unknown cat", false],
    ]);
  });

  it("maps detect responses to unnamed cats", () => {
    const result = fromDetect({
      image: { width: 100, height: 50 },
      found: true,
      counts: { cat: 1 },
      detections: [{ label: "cat", class_id: 15, confidence: 0.9, box }],
      inference_ms: 200,
    });
    expect(result.findings).toEqual([{ box, title: "Cat", known: null, confidence: 0.9 }]);
  });

  it("formats names", () => {
    expect(displayName("luna")).toBe("Luna");
    expect(displayName("big_tom-2")).toBe("Big tom 2");
  });
});
