
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageOps
import re
import json

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/manifest.csv"
OUTPUT_DIR = ROOT / "data/tensors"
PROCESSED_DIR = ROOT / "data/processed"

IMAGE_SIZE = 224
SEED = 42

CLASSES = {
    "pawara": 0,
    "uchitha": 1,
    "dhamith": 2,
    "unknown": 3
}

CLASS_NAMES = list(CLASSES)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n===== FACEOPS IMPROVED TENSOR PIPELINE =====")

df = pd.read_csv(MANIFEST)

assert df["path"].is_unique

for _, row in df.iterrows():
    assert int(row["label"]) == CLASSES[row["class_name"]]


def capture_information(path):
    """Extract session ID and chronological burst frame."""
    stem = Path(path).stem

    match = re.match(
        r"^(.*?)[_-]?TIMEBURST(\d+)$",
        stem,
        re.IGNORECASE
    )

    if match:
        return match.group(1), int(match.group(2))

    return f"unverified:{stem}", 0


splits = {
    "train": [],
    "val": [],
    "test": []
}

# ==============================================
# CHRONOLOGICAL AND GROUP-AWARE SPLITTING
# ==============================================

for class_name in CLASS_NAMES:

    class_rows = df[
        df["class_name"] == class_name
    ].to_dict("records")

    # Sort by session and burst frame number.
    class_rows.sort(
        key=lambda row: (
            *capture_information(row["path"]),
            row["path"]
        )
    )

    if class_name != "unknown":

        n = len(class_rows)
        n_train = round(n * 0.70)
        n_val = round(n * 0.15)

        splits["train"].extend(
            class_rows[:n_train]
        )

        splits["val"].extend(
            class_rows[n_train:n_train + n_val]
        )

        splits["test"].extend(
            class_rows[n_train + n_val:]
        )

    else:

        # Keep Unknown capture sessions intact.
        groups = defaultdict(list)

        for row in class_rows:
            session, _ = capture_information(row["path"])
            groups[session].append(row)

        sessions = sorted(groups)

        if len(sessions) < 3:
            raise ValueError(
                "Need at least 3 Unknown capture sessions "
                "for group-aware splitting."
            )

        # Reserve the last two capture sessions
        # for validation and testing.
        for session in sessions[:-2]:
            splits["train"].extend(groups[session])

        splits["val"].extend(
            groups[sessions[-2]]
        )

        splits["test"].extend(
            groups[sessions[-1]]
        )

        print("Unknown capture groups:", len(sessions))


# ==============================================
# VALIDATE SPLIT INTEGRITY
# ==============================================

all_rows = sum(splits.values(), [])

all_paths = [row["path"] for row in all_rows]

assert len(all_rows) == len(df)
assert len(all_paths) == len(set(all_paths))
assert set(all_paths) == set(df["path"])

print("\nSplit distribution:")

for split_name, rows in splits.items():
    print(
        f"{split_name}: {len(rows)} images",
        dict(Counter(row["class_name"] for row in rows))
    )


# ==============================================
# IMAGE PREPROCESSING
# ==============================================

def image_to_tensor(path):

    with Image.open(ROOT / path) as image:

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image = ImageOps.pad(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0)
        )

        array = np.array(
            image,
            dtype=np.uint8,
            copy=True
        )

    return torch.from_numpy(array).permute(
        2, 0, 1
    ).contiguous()


# ==============================================
# GENERATE PYTORCH DATASETS
# ==============================================

for split_name, rows in splits.items():

    images = torch.stack([
        image_to_tensor(row["path"])
        for row in rows
    ])

    labels = torch.tensor(
        [int(row["label"]) for row in rows],
        dtype=torch.long
    )

    dataset = {
        "images": images,
        "labels": labels,
        "paths": [row["path"] for row in rows],
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE
    }

    output = OUTPUT_DIR / f"{split_name}.pt"
    torch.save(dataset, output)

    print(
        f"\n{split_name.upper()}:",
        images.shape,
        labels.shape
    )


# ==============================================
# SAVE METADATA
# ==============================================

parts = []

for split_name, rows in splits.items():
    part = pd.DataFrame(rows)
    part["split"] = split_name
    parts.append(part)

split_manifest = pd.concat(
    parts,
    ignore_index=True
)

split_manifest.to_csv(
    PROCESSED_DIR / "split_manifest.csv",
    index=False
)

metadata = {
    "total_images": len(df),
    "image_size": IMAGE_SIZE,
    "channels": 3,
    "dtype": "uint8",
    "seed": SEED,
    "classes": CLASSES,
    "strategy": "chronological_known_grouped_unknown",
    "limitation": (
        "Known classes have only one capture session. "
        "Evaluation is not session-independent."
    ),
    "splits": {
        name: len(rows)
        for name, rows in splits.items()
    }
}

with (
    PROCESSED_DIR / "tensor_metadata.json"
).open("w") as file:
    json.dump(metadata, file, indent=4)

print("\n===== TENSOR PIPELINE COMPLETED =====")

