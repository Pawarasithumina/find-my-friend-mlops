
from pathlib import Path
from itertools import combinations
from collections import Counter
import csv

import imagehash
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data/processed/split_manifest.csv"
REPORT_DIR = ROOT / "reports/private_audit"

HASH_THRESHOLD = 6

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_perceptual_hash(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return imagehash.phash(image, hash_size=8)


def main():
    print("\n===== FACEOPS DATASET LEAKAGE AUDIT =====")

    with MANIFEST.open(newline="") as file:
        rows = list(csv.DictReader(file))

    records = []

    for row in rows:
        image_path = ROOT / row["path"]

        records.append({
            "path": row["path"],
            "class_name": row["class_name"],
            "split": row["split"],
            "hash": get_perceptual_hash(image_path),
        })

    print(f"Images analyzed: {len(records)}")
    print("Split distribution:",
          dict(Counter(r["split"] for r in records)))

    suspicious_pairs = []
    compared = 0

    for first, second in combinations(records, 2):

        # Only compare different dataset splits.
        if first["split"] == second["split"]:
            continue

        compared += 1

        distance = first["hash"] - second["hash"]

        if distance <= HASH_THRESHOLD:
            suspicious_pairs.append({
                "image_1": first["path"],
                "image_2": second["path"],
                "split_1": first["split"],
                "split_2": second["split"],
                "class_1": first["class_name"],
                "class_2": second["class_name"],
                "phash_distance": distance,
                "same_class": (
                    first["class_name"] == second["class_name"]
                ),
            })

    suspicious_pairs.sort(
        key=lambda item: item["phash_distance"]
    )

    report_path = (
        REPORT_DIR / "cross_split_candidates.csv"
    )

    with report_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_1",
                "image_2",
                "split_1",
                "split_2",
                "class_1",
                "class_2",
                "phash_distance",
                "same_class",
            ]
        )
        writer.writeheader()
        writer.writerows(suspicious_pairs)

    print("\n===== AUDIT RESULTS =====")
    print("Cross-split pairs compared:", compared)
    print("Potential similar-image pairs:",
          len(suspicious_pairs))

    print(
        "Very similar pHash pairs (distance <= 2):",
        sum(
            item["phash_distance"] <= 2
            for item in suspicious_pairs
        )
    )

    print(
        "Pairs with different class labels:",
        sum(
            not item["same_class"]
            for item in suspicious_pairs
        )
    )

    if suspicious_pairs:
        print("\nTop candidates for manual review:")

        for item in suspicious_pairs[:15]:
            print(
                f"\nDistance: {item['phash_distance']}"
                f"\n {item['split_1']}: {item['image_1']}"
                f"\n {item['split_2']}: {item['image_2']}"
            )

    print("\nPrivate audit report:", report_path)
    print("\nAudit completed. No images were modified.")


if __name__ == "__main__":
    main()
