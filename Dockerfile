# Two stages: PyTorch + Ultralytics are only needed to export the model to
# ONNX. The runtime image serves it with ONNX Runtime, without PyTorch, so it
# fits in 512 MB of RAM (e.g. Render's Starter plan).

# --- Stage 1: export YOLO12n and the cat embedder to ONNX ---------------
FROM python:3.12-slim AS export

ENV PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# OpenCV (pulled in by ultralytics) needs these shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
RUN pip install "ultralytics>=8.3.78" "onnx>=1.12" "onnxslim>=0.1.82"

WORKDIR /export
COPY training ./training
# Any pretrained Ultralytics detection model, e.g. yolo12s.pt. Dynamic input
# size, so DETECT_IMAGE_SIZE can be changed without rebuilding.
ARG MODEL=yolo12n.pt
RUN python -m training.export --weights ${MODEL} --dynamic --output /export/model.onnx
# Image embedder used to tell your cats apart (ImageNet-pretrained classifier
# without its last layer).
ARG EMBEDDER=yolo11n-cls.pt
RUN python -m training.export_embedder --weights ${EMBEDDER} --output /export/cat-embedder.onnx

# --- Stage 2: runtime -----------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install ".[onnx]"

# To serve a fine-tuned model, export it (python -m training.export --dynamic),
# mount it (e.g. -v ./models:/models/custom) and set DETECT_MODEL_WEIGHTS.
COPY --from=export /export/model.onnx /models/yolo12n.onnx
COPY --from=export /export/cat-embedder.onnx /models/cat-embedder.onnx
ENV DETECT_BACKEND=onnxruntime \
    DETECT_MODEL_WEIGHTS=/models/yolo12n.onnx \
    DETECT_EMBEDDER_WEIGHTS=/models/cat-embedder.onnx \
    DETECT_CATS_DIR=/var/data/cats

# Photos of your cats live in /var/data/cats: mount a persistent disk or
# volume on /var/data to keep them across restarts and deploys.
RUN useradd --system --no-create-home app \
    && mkdir -p /var/data/cats && chown -R app /var/data
USER app

# Platforms such as Render inject the port to listen on through $PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\""]
