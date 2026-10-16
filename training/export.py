"""Export YOLO weights to ONNX for the PyTorch-free ``onnxruntime`` backend.

Examples:
    # pretrained YOLO12n, input size chosen at runtime (recommended)
    python -m training.export --weights yolo12n.pt --dynamic

    # a fine-tuned model, fixed 640x640 input, NMS inside the graph
    python -m training.export --weights runs/pets-v1/weights/best.pt --nms

The script prints the environment variables to serve the result. Requires the
export extra: pip install -e ".[export]"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", default="yolo12n.pt", help="Pretrained name or path to a .pt file")
    p.add_argument("--imgsz", type=int, default=640, help="Export input size (default 640)")
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Variable input size: DETECT_IMAGE_SIZE then applies at runtime, and non-square "
        "images are padded less (faster)",
    )
    p.add_argument("--half", action="store_true", help="FP16 weights (needs a GPU to export)")
    p.add_argument("--nms", action="store_true", help="Embed NMS in the graph")
    p.add_argument("--opset", type=int, default=None, help="ONNX opset (default: latest supported)")
    p.add_argument("--no-simplify", action="store_true", help="Skip onnxslim graph simplification")
    p.add_argument("--device", default="cpu", help="Export device: cpu, 0...")
    p.add_argument(
        "--output", default=None, help="Where to write the .onnx (default: next to weights)"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    from ultralytics import YOLO

    exported = Path(
        YOLO(args.weights).export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            half=args.half,
            nms=args.nms,
            opset=args.opset,
            simplify=not args.no_simplify,
            device=args.device,
        )
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        exported = Path(shutil.move(exported, output))

    print("\nServe it with:")
    print("  DETECT_BACKEND=onnxruntime")
    print(f"  DETECT_MODEL_WEIGHTS={exported}")
    if args.dynamic:
        print(f"  DETECT_IMAGE_SIZE={args.imgsz}  # any multiple of 32")
    return exported


if __name__ == "__main__":
    main()
