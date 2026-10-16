"""Export the image embedder used to tell your cats apart.

The embedder is an ImageNet-pretrained Ultralytics classification model with
its last layer removed: for an image of a cat it returns a vector (an
appearance "fingerprint"). The service compares the fingerprint of a detected
cat with those of the photos you uploaded for each of your cats, so adding a
cat needs no training.

Example:
    python -m training.export_embedder --weights yolo11n-cls.pt --output models/cat-embedder.onnx

Requires the export extra: pip install -e ".[export]"
"""

from __future__ import annotations

import argparse
from pathlib import Path

INPUT_SIZE = 224


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--weights",
        default="yolo11n-cls.pt",
        help="Ultralytics classification weights (yolo11n-cls.pt, yolo11s-cls.pt...)",
    )
    p.add_argument("--output", default="cat-embedder.onnx", help="Where to write the .onnx")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    import torch
    from torch import nn
    from ultralytics import YOLO

    yolo = YOLO(args.weights)
    if yolo.task != "classify":
        raise ValueError(
            f"{args.weights} is a {yolo.task!r} model; a classification model is needed"
        )
    layers = list(yolo.model.model)
    body, head = layers[:-1], layers[-1]
    if any(getattr(m, "f", -1) != -1 for m in body):
        raise ValueError(f"{args.weights} is not a plain chain of layers; unsupported")

    class Embedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(*body)
            self.conv, self.pool = head.conv, head.pool

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.pool(self.conv(self.body(images))).flatten(1)

    embedder = Embedder().float().eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        size = embedder(dummy).shape[1]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        embedder,
        (dummy,),
        str(output),
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )

    import onnx

    exported = onnx.load(str(output))
    for key, value in {"task": "embed", "imgsz": str(INPUT_SIZE), "source": args.weights}.items():
        exported.metadata_props.add(key=key, value=value)
    onnx.save(exported, str(output))

    print(f"\nExported a {size}-dimensional embedder to {output}")
    print("Serve it with:")
    print(f"  DETECT_EMBEDDER_WEIGHTS={output}")
    return output


if __name__ == "__main__":
    main()
