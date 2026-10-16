# Detect_image_engine_yolo12n

A microservice that receives an image over HTTP and tells you **whether there is a cat in it**, using the [YOLO12](https://docs.ultralytics.com/models/yolo12/) nano model (`yolo12n.pt`, COCO-pretrained).

It follows the design of [`detect_image_engine`](https://github.com/marcomaggiotti/detect_image_engine), with YOLO12n as the model:

- **Cats by default:** `POST /v1/cat` answers yes/no, `POST /v1/detect` returns the boxes.
- **Other objects:** people, dogs and the rest of COCO's 80 classes are a request parameter or a config change away. No code changes.
- **Swappable model:** the weights come from configuration. A fine-tuned YOLO12 `best.pt` (or YOLO11, RT-DETR, a custom backend) is a drop-in replacement.
- **Your own cats by name:** upload a few photos of each of your cats, then `POST /v1/identify` tells you which of them is in a picture. No training needed.
- **Small in production:** the Docker image serves ONNX exports with ONNX Runtime, without PyTorch. It peaks at ~200 MB of RAM, so it runs on Render's $7/month Starter plan.

```
                    POST /v1/cat  ·  POST /v1/detect (image)
                                   │
                            ┌──────▼──────┐   validates classes, size limits,
                            │   FastAPI   │   EXIF orientation, RGB conversion
                            └──────┬──────┘
                            ┌──────▼──────┐
                            │  Detection  │   class selection, thresholds,
                            │   Service   │   counts, timing, thread safety
                            └──────┬──────┘
                                   │  Detector interface (app/detectors/base.py)
           ┌───────────────────────┼───────────────────────┐
┌──────────▼─────────┐  ┌──────────▼─────────┐  ┌──────────▼─────────┐
│ ultralytics        │  │ onnxruntime        │  │ your backend       │
│ yolo12n.pt         │  │ yolo12n.onnx       │  │ "pkg.module:Class" │
│ local, training    │  │ Docker, Render     │  │                    │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ultralytics,dev]"
uvicorn app.main:app --reload
```

The first start downloads `yolo12n.pt` (~5 MB). Interactive docs: http://localhost:8000/docs

### Is there a cat?

```bash
curl -F image=@my_cat.jpg http://localhost:8000/v1/cat
```

```json
{
  "model": {"backend": "ultralytics", "weights": "yolo12n.pt", "version": null},
  "cat": true,
  "count": 1,
  "confidence": 0.91,
  "inference_ms": 48.3
}
```

### Where are the cats?

```bash
curl -F image=@my_cat.jpg http://localhost:8000/v1/detect
```

```json
{
  "model": {"backend": "ultralytics", "weights": "yolo12n.pt", "version": null},
  "image": {"width": 335, "height": 500},
  "classes": ["cat"],
  "found": true,
  "counts": {"cat": 1},
  "detections": [
    {"label": "cat", "class_id": 15, "confidence": 0.91,
     "box": {"x1": 12.4, "y1": 30.1, "x2": 320.8, "y2": 488.0}}
  ],
  "inference_ms": 48.3
}
```

Box coordinates are pixels of the uploaded image, after EXIF rotation is applied.

### Which of my cats is it?

Teach the service your cats once: 5–10 photos each, where only that cat is visible, with varied poses, rooms and lighting.

```bash
curl -H "Authorization: Bearer $DETECT_ADMIN_TOKEN" \
  -F images=@luna1.jpg -F images=@luna2.jpg -F images=@luna3.jpg \
  http://localhost:8000/v1/cats/luna/photos
```

```json
{"name": "luna", "added": 3, "duplicates": 0, "rejected": [], "photos": 3}
```

Then ask:

```bash
curl -F image=@sofa.jpg http://localhost:8000/v1/identify
```

```json
{
  "names": ["luna"],
  "cats": [
    {"name": "luna", "closest": "luna", "similarity": 0.94, "confidence": 0.9,
     "box": {"x1": 12.4, "y1": 30.1, "x2": 320.8, "y2": 488.0}},
    {"name": null, "closest": "milo", "similarity": 0.61, "confidence": 0.82,
     "box": {"x1": 341.5, "y1": 22.7, "x2": 639.5, "y2": 372.5}}
  ],
  "...": "..."
}
```

`name` is `null` for a cat that doesn't look enough like any of yours (a neighbour's cat). The `/docs` page has an **Authorize** button for the token, so you can also upload photos from the browser. See [How telling cats apart works](#how-telling-cats-apart-works).

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/cat` | Multipart upload, field `image`. Returns whether there is a cat, how many, and the best score. |
| `POST` | `/v1/detect` | Multipart upload, field `image`. Returns detections (boxes) for the requested classes. |
| `POST` | `/v1/identify` | Multipart upload, field `image`. Which of your cats are in it. |
| `GET` | `/v1/cats` | Your cats and how many photos each has. |
| `POST` | `/v1/cats/{name}/photos` | 🔒 Multipart upload, field `images` (up to 20). Adds photos of one of your cats. |
| `DELETE` | `/v1/cats/{name}` | 🔒 Forgets a cat and its photos. |
| `GET` | `/v1/model` | Active backend, weights, version, default classes and every label the model knows. |
| `GET` | `/health` | Liveness check. |

Query parameters:

| Param | Endpoints | Example | Meaning |
| --- | --- | --- | --- |
| `confidence` | `/v1/cat`, `/v1/detect`, `/v1/identify` | `0.5` | Minimum detection score (0–1). Defaults to `DETECT_CONFIDENCE_THRESHOLD`. |
| `classes` | `/v1/detect` | `cat` · `cat,dog,person` · `*` | Classes to report. Defaults to `DETECT_TARGET_CLASSES`. `*` returns everything the model knows. Case-insensitive. An unknown class returns **400**. |

`/v1/cat` always looks for `cat`, whatever `DETECT_TARGET_CLASSES` says.

🔒 needs `Authorization: Bearer <DETECT_ADMIN_TOKEN>` when a token is configured.

Errors: `400` for an invalid/corrupt image, unknown class or invalid cat name, `401` for a missing or wrong token, `409` for `/v1/identify` before any cat is added, `413` for an upload over the size limit, `422` for an invalid parameter, `503` when telling cats apart isn't configured.

## Configuration

Everything is set with environment variables (or a `.env` file). See [`.env.example`](.env.example).

| Variable | Default | Description |
| --- | --- | --- |
| `DETECT_BACKEND` | `ultralytics` (`onnxruntime` in Docker) | `ultralytics`, `onnxruntime`, or a custom `package.module:ClassName` |
| `DETECT_MODEL_WEIGHTS` | `yolo12n.pt` (`/models/yolo12n.onnx` in Docker) | Model name or path to (fine-tuned) weights; a `.onnx` file for `onnxruntime` |
| `DETECT_MODEL_ARCH` | – | ultralytics only: `yolo` (default) or `rtdetr` |
| `DETECT_MODEL_LABELS` | – | Label file (one name per line, line N = class N) overriding the names in the weights |
| `DETECT_MODEL_VERSION` | – | Tag echoed in responses, so you know which model answered |
| `DETECT_DEVICE` | auto | `cpu`, `cuda`, `cuda:0`, `mps` |
| `DETECT_THREADS` | all host cores | CPU threads per inference. Set it on small cloud instances (`1` on Render Starter) |
| `DETECT_ONNX_PROVIDERS` | from `DETECT_DEVICE` | onnxruntime: execution providers in priority order, e.g. `CUDAExecutionProvider,CPUExecutionProvider` |
| `DETECT_TARGET_CLASSES` | `cat` | Default classes for `/v1/detect`, comma separated (`*` = all) |
| `DETECT_CONFIDENCE_THRESHOLD` | `0.25` | Default minimum score |
| `DETECT_IOU_THRESHOLD` | `0.45` | NMS IoU |
| `DETECT_IMAGE_SIZE` | `640` | Inference size. With onnxruntime it only applies to `--dynamic` exports |
| `DETECT_MAX_DETECTIONS` | `100` | Max boxes per image |
| `DETECT_MAX_UPLOAD_BYTES` | `10485760` | Upload size limit |
| `DETECT_MAX_IMAGE_PIXELS` | `40000000` | Rejects decompression bombs before decoding |
| `DETECT_WARMUP` | `true` | Run one dummy inference at startup |
| `DETECT_EMBEDDER_WEIGHTS` | – (`/models/cat-embedder.onnx` in Docker) | Embedder from `training/export_embedder.py`. Unset disables `/v1/identify` and `/v1/cats` |
| `DETECT_CATS_DIR` | – (`/var/data/cats` in Docker) | Where photos of your cats are kept. Unset: memory only, lost on restart |
| `DETECT_IDENTIFY_THRESHOLD` | `0.8` | Minimum similarity to name one of your cats |
| `DETECT_ADMIN_TOKEN` | – | Required to add or delete cats. Unset: anyone who can reach the service can (a warning is logged) |

At startup the service checks that the model can detect every class in `DETECT_TARGET_CLASSES`. If it can't, startup fails with a clear error instead of returning empty results.

## Beyond cats

**1. Classes YOLO12n already knows: no training needed.** The default model is trained on COCO's 80 classes, including `person`, `cat`, `dog`, `bird`, `horse`, `teddy bear`, `chair` and more (see `GET /v1/model`):

```bash
curl -F image=@photo.jpg "http://localhost:8000/v1/detect?classes=cat,dog,person"
```

or change the service default: `DETECT_TARGET_CLASSES=cat,dog,person`.

**2. Better accuracy: a bigger YOLO12.** `DETECT_MODEL_WEIGHTS=yolo12s.pt` (or `yolo12m.pt`, `yolo12l.pt`, `yolo12x.pt`). Slower and heavier, but more accurate. For the Docker image: `docker build --build-arg MODEL=yolo12s.pt .`

**3. New classes or your own data: fine-tune** YOLO12n, e.g. for a specific cat, a breed, or rabbits. See next section.

## How telling cats apart works

There is no training. A second, small model (the *embedder*: an ImageNet-pretrained `yolo11n-cls` with its last layer removed, 6 MB as ONNX) turns a cropped cat into a "fingerprint" of 1,280 numbers. When you upload photos of Luna, the detector crops the cat in each one and its fingerprint is stored. For `/v1/identify`, each detected cat's fingerprint is compared with the stored ones (cosine similarity, average of the 3 best matches per cat), and the best match wins if it reaches `DETECT_IDENTIFY_THRESHOLD`.

- **Photos:** each must show exactly one cat. Photos with no cat or several cats are rejected and listed in the response. Exact duplicates are skipped. A cat can have up to 200 photos.
- **Storage:** crops are stored as JPEG files in `DETECT_CATS_DIR/<name>/`, with their fingerprints cached next to them, so restarts are fast. If you change the embedder, fingerprints are recomputed from the photos at the next start.
- **How well it works:** in a simulated test with 5 "own" cats and 300 other cats (the different photos of each own cat were varied crops, flips and lighting of one photo), the closest cat was always the right one. With a threshold of 0.8, 98–100% of the own cats were named and 1 stranger in 300 was mistaken for one of them. Real photos of the same cat differ more (pose, angle), so expect lower similarities: if your cats come back as `null`, add more varied photos or lower the threshold a little, and check the `similarity` values to choose it.
- **Limits:** cats that look very much alike (two black cats) may be confused, because the embedder was trained to recognise breeds and objects, not individuals. The model can be swapped: `--weights yolo11s-cls.pt` in `training/export_embedder.py` (or `--build-arg EMBEDDER=yolo11s-cls.pt`) is bigger and may separate lookalikes better (not measured).

## Fine-tuning

1. Label images in YOLO format. CVAT, Label Studio and Roboflow all export it. See [`training/dataset.example.yaml`](training/dataset.example.yaml) for the layout.
2. Train, starting from the pretrained YOLO12n (transfer learning):

   ```bash
   python -m training.finetune --data path/to/dataset.yaml \
       --base yolo12n.pt --epochs 50 --name pets-v1
   # small dataset? freeze the backbone:  --freeze 10
   # continue from a previous run:        --base runs/pets-v1/weights/best.pt
   # export for faster CPU/GPU serving:   --export onnx
   ```

   The script trains, validates (prints mAP), and prints the exact variables to serve the result.
3. Serve it:

   ```bash
   DETECT_MODEL_WEIGHTS=runs/pets-v1/weights/best.pt \
   DETECT_MODEL_VERSION=pets-v1 \
   DETECT_TARGET_CLASSES=cat,dog,person \
   uvicorn app.main:app
   ```

Fine-tuned Ultralytics weights store their own class names, so no label file is needed. To serve the result from the Docker image, export it to ONNX (`--export onnx`, or see below).

> ⚠️ Fine-tuning **replaces** the model's class list with your dataset's classes. Keep `cat` in the dataset (with labelled examples) or `/v1/cat` will answer 400.

## Serving with ONNX Runtime (no PyTorch)

The `onnxruntime` backend runs a YOLO model exported to ONNX with only ONNX Runtime, NumPy and Pillow. It gives the same detections as the ultralytics backend (checked on the test photos: same scores, boxes within a pixel) and needs a fraction of the memory: about 150 MB instead of about 1.2 GB.

```bash
pip install -e ".[export]"                      # PyTorch + Ultralytics, only to export
python -m training.export --weights yolo12n.pt --dynamic --output models/yolo12n.onnx

pip install -e ".[onnx]"                        # what production needs
DETECT_BACKEND=onnxruntime DETECT_MODEL_WEIGHTS=models/yolo12n.onnx uvicorn app.main:app
```

`training/export.py` options:

| Option | Effect |
| --- | --- |
| `--dynamic` | Recommended. Input size is chosen at runtime with `DETECT_IMAGE_SIZE`, and non-square photos are padded less, so they run faster. Without it the model always runs at the export size and ignores `DETECT_IMAGE_SIZE` (a warning is logged). |
| `--imgsz 640` | Export (and default) input size. |
| `--nms` | Puts duplicate-box removal inside the graph. The backend reads both output layouts. |
| `--half` | FP16 weights, for GPU serving. Exporting it needs a GPU. |
| `--opset N`, `--no-simplify` | ONNX opset version; skip graph simplification. |
| `--weights best.pt` | Any Ultralytics YOLO detection model (YOLOv8 to YOLO12, fine-tuned or not). RT-DETR exports aren't supported by this backend. |

Smaller `DETECT_IMAGE_SIZE` values are faster, but don't go too low for this service: at 320, YOLO12n called a dog photo a cat with a 0.85 score, which it got right at 640.

For a GPU, install `onnxruntime-gpu` instead of `onnxruntime` and set `DETECT_DEVICE=cuda` (or `DETECT_ONNX_PROVIDERS`).

## Your own backend

Subclass `Detector` and point the config at it. No changes to the service are needed:

```python
# mypkg/detector.py
from app.detectors import Detection, Detector

class MyDetector(Detector):
    backend_name = "my-model"

    def __init__(self, config):
        super().__init__(config)
        self.model = load_my_model(config.weights)

    @property
    def labels(self) -> dict[int, str]:
        return {0: "cat", 1: "dog"}

    def detect(self, image, *, confidence, class_ids=None):
        ...  # return list[Detection] in pixel xyxy
```

```bash
DETECT_BACKEND=mypkg.detector:MyDetector uvicorn app.main:app
```

## Docker

```bash
docker build -t detect-image-engine-yolo12n .
docker run -p 8000:8000 -e DETECT_THREADS=1 -e DETECT_ADMIN_TOKEN=change-me \
  -v cat-photos:/var/data detect-image-engine-yolo12n

# serve a fine-tuned model (exported with: python -m training.export --weights best.pt --dynamic)
docker run -p 8000:8000 -v "$PWD/runs/pets-v1/weights:/models/custom:ro" \
  -e DETECT_MODEL_WEIGHTS=/models/custom/best.onnx -e DETECT_MODEL_VERSION=pets-v1 \
  detect-image-engine-yolo12n
```

The build has two stages. The first installs CPU PyTorch and Ultralytics only to export YOLO12n and the embedder to ONNX (`--build-arg MODEL=yolo12s.pt` or `EMBEDDER=yolo11s-cls.pt` pick others). The final image contains ONNX Runtime and the two `.onnx` files, without PyTorch (498 MB). The `cat-photos` volume keeps your cats' photos across restarts.

Limited to 512 MB of RAM and half a CPU, the container peaked at 200 MB. `/v1/cat` took about 0.2 s per image and `/v1/identify` about 0.4 s (it runs both models).

Each process loads one model and runs one inference at a time. Scale with more replicas (or `uvicorn --workers N`; each worker loads its own copy of the model).

## Deploy on Render

The repo includes a Render Blueprint, [`render.yaml`](render.yaml). In the Render dashboard choose **New → Blueprint** and select this repository. Render builds the Dockerfile and checks `/health`. The container listens on the `$PORT` that Render provides.

- **Plan:** `starter` (512 MB, 0.5 CPU). The image peaks at about 200 MB, and `DETECT_THREADS=1` matches the half CPU. Expect about 0.2 s for `/v1/cat` and 0.4 s for `/v1/identify`. For more, move to `standard` (1 CPU) with `DETECT_THREADS=2`.
- **Disk:** a 1 GB persistent disk at `/var/data` keeps your cats' photos across deploys. It is billed separately (about $0.25/GB per month, per a third-party price list; check Render's pricing). Render may limit a service with a disk to a single instance and to deploys with a short downtime; check their docs if that matters. Without the disk, delete the `disk` block: the photos are then lost on every deploy.
- **Admin token:** Render generates `DETECT_ADMIN_TOKEN`. Copy it from the service's **Environment** tab to add or delete cats.
- **Free plan:** with a tenth of a CPU each image takes over a second, and the service sleeps when idle, so the next request waits for it to start.
- **Region:** `oregon`. Change `region` in `render.yaml` to deploy elsewhere.
- **Fine-tuned model:** Render's filesystem is reset on every deploy. Export it to ONNX, then either copy the `.onnx` into the image, or upload it to the disk and set `DETECT_MODEL_WEIGHTS=/var/data/models/best.onnx`.
- **If `/v1/cats` reports `"persistent": false` on Render:** the disk isn't writable by the container's user. The logs then say "Cannot write to /var/data/cats". The photos still work until the next restart.

## Development

```bash
pip install -e ".[ultralytics,onnx,export,dev]"
pytest
ruff check . && ruff format --check .
```

The API tests use a fake detector and a fake embedder, so they need no ML framework. The ONNX Runtime tests build tiny ONNX models with known outputs (they need `onnx` and `onnxruntime`). The ultralytics smoke test builds a randomly initialised YOLO12n offline. Tests whose framework isn't installed are skipped.

## Licensing note

This repository is MIT-licensed. The **Ultralytics** package and its YOLO12 weights are **AGPL-3.0**. Running them in a network service may require releasing your service's source code under the AGPL, unless you hold an Ultralytics Enterprise license.
