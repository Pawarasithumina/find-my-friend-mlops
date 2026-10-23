"""Simple typed-enough FaceOps inference API client. No frame persistence."""
from __future__ import annotations

import os
import requests

API_URL = os.getenv("FACEOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")


def health(url: str = API_URL) -> dict:
    response = requests.get(f"{url.rstrip('/')}/health", timeout=(1.5, 3.0))
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "ok" or result.get("loaded_models") != 2:
        raise RuntimeError(f"Production API is not ready: {result}")
    return result


def compare(jpeg_bytes: bytes, url: str = API_URL) -> dict:
    if not jpeg_bytes:
        raise ValueError("No face image supplied")
    response = requests.post(
        f"{url.rstrip('/')}/predict/compare",
        files={"image": ("face.jpg", jpeg_bytes, "image/jpeg")},
        timeout=(2.0, 6.0),
    )
    response.raise_for_status()
    result = response.json()
    for slot in ("model_a", "model_b"):
        entry = result.get(slot)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing API output: {slot}")
        if "predicted_class" not in entry or "confidence" not in entry:
            raise ValueError(f"Incomplete API output: {slot}")
    return result
