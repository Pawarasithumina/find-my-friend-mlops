
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/split_manifest.csv"

CLASSES = ["pawara", "uchitha", "dhamith", "unknown"]

groups = defaultdict(list)

with MANIFEST.open(newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    filename = Path(row["path"]).stem

    # Identify images from the same TIMEBURST session
    match = re.match(
        r"^(.*?)[_-]?TIMEBURST\d+$",
        filename,
        re.IGNORECASE
    )

    if match:
        session = match.group(1)
    else:
        # Treat unmatched filenames as separate groups
        # until their capture origin is verified.
        session = f"unverified:{filename}"

    key = (row["class_name"], session)
    groups[key].append(row["split"])


print("\n===== FACEOPS CAPTURE SESSION AUDIT =====")

total_cross_split = 0

for class_name in CLASSES:
    class_groups = {
        key: splits
        for key, splits in groups.items()
        if key[0] == class_name
    }

    cross_split = {
        key: splits
        for key, splits in class_groups.items()
        if len(set(splits)) > 1
    }

    total_cross_split += len(cross_split)

    print(f"\nCLASS: {class_name.upper()}")
    print("Capture groups:", len(class_groups))
    print(
        "Largest group:",
        max(map(len, class_groups.values()), default=0)
    )
    print("Groups crossing splits:", len(cross_split))

    largest = sorted(
        class_groups.items(),
        key=lambda item: len(item[1]),
        reverse=True
    )[:5]

    for (_, session), splits in largest:
        print(
            f"  {session}: "
            f"{len(splits)} images, "
            f"splits={dict(Counter(splits))}"
        )

print("\n===== FINAL SUMMARY =====")
print("Total images:", len(rows))
print("Total capture groups:", len(groups))
print("Cross-split groups:", total_cross_split)
print("\nAudit completed. No files were modified.")
