"""Local FaceOps dual-model FastAPI inference service.

Only cropped face images should be sent. Camera frames are not persisted.
"""
from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from src.serving.registry_loader import EXPECTED_CLASSES, load_approved_models

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 12_000_000
CONFIDENCE_WARNING_THRESHOLD = float(os.getenv("FACEOPS_LOW_CONFIDENCE", "0.60"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup rather than silently serving a different/unapproved model.
    app.state.models = load_approved_models()
    yield
    app.state.models.clear()


app = FastAPI(
    title="FaceOps TwinTrack Inference API",
    description="Two MLflow-approved experimental face classifiers; not for security or identification decisions.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)


def _decode(data: bytes) -> torch.Tensor:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be between 1 byte and 8 MiB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Only JPEG, PNG, or WEBP supported")
            if image.width * image.height > 12_000_000:
                raise ValueError("Image dimensions too large")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            arr = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(arr).permute(2, 0, 1).float().div(255)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _predict(slot: str, image_tensor: torch.Tensor) -> dict:
    loaded = app.state.models[slot]
    if loaded.architecture == "custom_cnn":
        mean, std = (.5, .5, .5), (.5, .5, .5)
    else:
        mean, std = (.485, .456, .406), (.229, .224, .225)
    mu = image_tensor.new_tensor(mean).view(3, 1, 1)
    sigma = image_tensor.new_tensor(std).view(3, 1, 1)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = loaded.model(((image_tensor - mu) / sigma).unsqueeze(0))
        probabilities = logits.softmax(1)[0].tolist()
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    return {
        "slot": slot,
        "architecture": loaded.architecture,
        "registered_model": loaded.registry_name,
        "model_version": loaded.registry_version,
        "predicted_class": EXPECTED_CLASSES[index],
        "confidence": confidence,
        "low_confidence": confidence < CONFIDENCE_WARNING_THRESHOLD,
        "probabilities": {label: float(probabilities[i]) for i, label in enumerate(EXPECTED_CLASSES)},
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }


async def _read_image(image: UploadFile) -> torch.Tensor:
    try:
        data = await image.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await image.close()
    return _decode(data)


@app.get("/health")
def health():
    loaded = getattr(app.state, "models", {})
    return {"status": "ok" if len(loaded) == 2 else "not_ready", "loaded_models": len(loaded)}


@app.get("/models")
def models():
    return {
        slot: {"architecture": m.architecture, "registered_model": m.registry_name, "version": m.registry_version, "alias": "production"}
        for slot, m in app.state.models.items()
    }


@app.post("/predict/model-a")
async def predict_model_a(image: UploadFile = File(...)):
    return _predict("model-a", await _read_image(image))


@app.post("/predict/model-b")
async def predict_model_b(image: UploadFile = File(...)):
    return _predict("model-b", await _read_image(image))


@app.post("/predict/compare")
async def compare(image: UploadFile = File(...)):
    tensor = await _read_image(image)
    a = _predict("model-a", tensor)
    b = _predict("model-b", tensor)
    return {"model_a": a, "model_b": b, "agreement": a["predicted_class"] == b["predicted_class"]}
