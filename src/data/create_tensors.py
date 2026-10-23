
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
import json

# ======================================
# CONFIGURATION
# ======================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data/processed/manifest.csv"
TENSOR_DIR = ROOT / "data/tensors"
PROCESSED_DIR = ROOT / "data/processed"

IMAGE_SIZE = 224
SEED = 42

CLASS_NAMES = [
    "pawara",
    "uchitha",
    "dhamith",
    "unknown"
]

CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

TENSOR_DIR.mkdir(parents=True, exist_ok=True)

# ======================================
# LOAD MANIFEST
# ======================================

print("\n===== FACEOPS TENSOR PIPELINE =====")

df = pd.read_csv(MANIFEST)

assert len(df) > 0, "Manifest is empty"
assert df["path"].is_unique, "Duplicate paths found"

for _, row in df.iterrows():
    expected = CLASS_TO_IDX[row["class_name"]]
    assert int(row["label"]) == expected, (
        f"Incorrect label: {row['path']}"
    )

print(f"Total images: {len(df)}")

# ======================================
# STRATIFIED DATASET SPLIT
# ======================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

# Ensure that no image path appears
# in multiple datasets.
all_paths = []

for subset in splits.values():
    all_paths.extend(subset["path"].tolist())

assert len(all_paths) == len(set(all_paths))
assert len(all_paths) == len(df)

print("\nDataset split:")
for name, subset in splits.items():
    print(f"{name}: {len(subset)} images")

# ======================================
# IMAGE TO TENSOR
# ======================================

def image_to_tensor(image_path):
    with Image.open(image_path) as img:
        # Fix image orientation.
        img = ImageOps.exif_transpose(img)

        # Convert to RGB.
        img = img.convert("RGB")

        # Preserve aspect ratio using padding.
        img = ImageOps.pad(
            img,
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0)
        )

        array = np.array(
            img, dtype=np.uint8, copy=True
        )

    # HWC -> CHW
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).contiguous()

    return tensor

# ======================================
# CREATE AND SAVE DATASETS
# ======================================

for split_name, subset in splits.items():

    images = []
    labels = []
    paths = []

    for _, row in subset.iterrows():
        image_path = ROOT / row["path"]

        tensor = image_to_tensor(image_path)

        images.append(tensor)
        labels.append(int(row["label"]))
        paths.append(row["path"])

    images = torch.stack(images)

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    dataset = {
        "images": images,
        "labels": labels,
        "paths": paths,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE
    }

    output_path = TENSOR_DIR / f"{split_name}.pt"

    torch.save(dataset, output_path)

    print(f"\n{split_name.upper()} DATASET")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Data type:", images.dtype)
    print("Class distribution:", dict(Counter(labels.tolist())))
    print("Saved:", output_path)

# ======================================
# SAVE SPLIT METADATA
# ======================================

split_manifest = []

for split_name, subset in splits.items():
    part = subset.copy()
    part["split"] = split_name
    split_manifest.append(part)

split_manifest = pd.concat(
    split_manifest,
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
    "pixel_range": [0, 255],
    "seed": SEED,
    "classes": CLASS_TO_IDX,
    "splits": {
        name: len(subset)
        for name, subset in splits.items()
    }
}

with open(
    PROCESSED_DIR / "tensor_metadata.json",
    "w"
) as f:
    json.dump(metadata, f, indent=4)

print("\n===== TENSOR GENERATION COMPLETED =====")
