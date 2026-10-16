import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PhotoResult } from "./PhotoResult";

afterEach(cleanup);

describe("PhotoResult", () => {
  it("draws a labelled box per cat, in percentages of the photo", () => {
    const { container } = render(
      <PhotoResult
        previewUrl="blob:photo"
        result={{
          image: { width: 200, height: 100 },
          findings: [
            { box: { x1: 20, y1: 10, x2: 120, y2: 60 }, title: "Luna", known: true, confidence: 0.9, similarity: 0.96, closest: "luna" },
            { box: { x1: 140, y1: 30, x2: 200, y2: 100 }, title: "Unknown cat", known: false, confidence: 0.8, similarity: 0.6, closest: "milo" },
          ],
          ms: 420,
        }}
      />,
    );

    expect(screen.getByRole("heading").textContent).toContain("Luna and a cat you haven't added");
    const boxes = container.querySelectorAll<HTMLElement>(".box");
    expect(boxes).toHaveLength(2);
    expect(boxes[0].style.left).toBe("10%");
    expect(boxes[0].style.width).toBe("50%");
    expect(boxes[0].style.top).toBe("10%");
    expect(boxes[1].classList.contains("box--unknown")).toBe(true);
    expect(screen.getByText("closest: milo, 60% similar · detected with 80% confidence")).toBeTruthy();
  });

  it("says when there is no cat", () => {
    render(<PhotoResult previewUrl={null} result={{ image: { width: 10, height: 10 }, findings: [], ms: 100 }} />);
    expect(screen.getByRole("heading").textContent).toContain("No cat in this photo");
  });
});
