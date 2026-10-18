import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PhotoBoxes } from "./PhotoBoxes";

afterEach(cleanup);

describe("PhotoBoxes", () => {
  it("draws a labelled box per cat, in percentages of the photo", () => {
    const { container } = render(
      <PhotoBoxes
        url="blob:photo"
        result={{
          image: { width: 200, height: 100 },
          findings: [
            { box: { x1: 20, y1: 10, x2: 120, y2: 60 }, title: "Luna", known: true, confidence: 0.9 },
            { box: { x1: 140, y1: 2, x2: 200, y2: 100 }, title: "Unknown cat", known: false, confidence: 0.8 },
          ],
          ms: 420,
        }}
      />,
    );
    const boxes = container.querySelectorAll<HTMLElement>(".box");
    expect(boxes).toHaveLength(2);
    expect([boxes[0].style.left, boxes[0].style.top, boxes[0].style.width]).toEqual(["10%", "10%", "50%"]);
    expect(boxes[0].textContent).toBe("Luna");
    expect(boxes[1].className).toContain("box--unknown");
    expect(boxes[1].className).toContain("box--label-inside"); // at the top edge
  });
});
