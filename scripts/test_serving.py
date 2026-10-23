"""Non-biometric synthetic-image integration smoke test."""
import io
import os

import requests
from PIL import Image

URL = os.getenv("FACEOPS_API_URL", "http://127.0.0.1:8000")
response = requests.get(URL + "/health", timeout=15)
response.raise_for_status()
assert response.json()["loaded_models"] == 2, response.text
metadata = requests.get(URL + "/models", timeout=15).json()
assert set(metadata) == {"model-a", "model-b"}

image = Image.new("RGB", (224, 224), (120, 105, 90))
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
response = requests.post(
    URL + "/predict/compare",
    files={"image": ("synthetic.jpg", buffer.getvalue(), "image/jpeg")},
    timeout=30,
)
response.raise_for_status()
predictions = response.json()
for slot in ("model_a", "model_b"):
    output = predictions[slot]
    assert output["predicted_class"] in ("pawara", "uchitha", "dhamith", "unknown")
    assert abs(sum(output["probabilities"].values()) - 1) < 0.0001
    print(slot, output["architecture"], "v" + output["model_version"], "OK")
print("PASS: both registered production models returned well-formed predictions")
print("Synthetic test image: predicted class is not a measure of accuracy.")
