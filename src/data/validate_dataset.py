
from pathlib import Path
from PIL import Image
from collections import defaultdict
import csv
import hashlib
import json

# Project configuration
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

CLASSES = {
    "pawara": 0,
    "uchitha": 1,
    "dhamith": 2,
    "unknown": 3,
}

EXTENSIONS = {".jpg", ".jpeg", ".png"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []
errors = []
hashes = defaultdict(list)
class_counts = {}

print("\n===== FACEOPS DATASET VALIDATION =====\n")

for class_name, label in CLASSES.items():
    class_dir = DATA_DIR / class_name

    if not class_dir.is_dir():
        errors.append(f"Missing class directory: {class_dir}")
        class_counts[class_name] = 0
        continue

    files = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    valid = 0

    for image_path in files:
        try:
            # Verify integrity
            with Image.open(image_path) as img:
                img.verify()

            # Fully decode image
            with Image.open(image_path) as img:
                img.load()
                width, height = img.size

            # Calculate file hash
            sha256 = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()

            relative_path = image_path.relative_to(ROOT)

            records.append({
                "path": str(relative_path),
                "class_name": class_name,
                "label": label,
                "width": width,
                "height": height,
                "sha256": sha256,
            })

            hashes[sha256].append(str(relative_path))
            valid += 1

        except Exception as exc:
            errors.append(f"{image_path}: {exc}")

    class_counts[class_name] = valid
    print(f"{class_name:10s}: {valid} valid images")

# Detect exact duplicate files
duplicates = [
    paths for paths in hashes.values()
    if len(paths) > 1
]

# Save metadata manifest
manifest_path = OUTPUT_DIR / "manifest.csv"

with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "path", "class_name", "label",
            "width", "height", "sha256"
        ],
    )
    writer.writeheader()
    writer.writerows(records)

report = {
    "total_valid_images": len(records),
    "class_counts": class_counts,
    "invalid_images": errors,
    "exact_duplicate_groups": duplicates,
}

report_path = OUTPUT_DIR / "validation_report.json"

with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print("\n===== VALIDATION SUMMARY =====")
print("Valid images:", len(records))
print("Invalid images:", len(errors))
print("Exact duplicate groups:", len(duplicates))
print("Manifest:", manifest_path)
print("Report:", report_path)

if errors:
    print("\nInvalid image details:")
    for error in errors:
        print(" -", error)

if duplicates:
    print("\nDuplicate file groups:")
    for group in duplicates:
        print(" -", group)

print("\nValidation completed.")
