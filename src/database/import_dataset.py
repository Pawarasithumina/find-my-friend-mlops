
from pathlib import Path
from collections import Counter
import csv
import hashlib
import os

import psycopg
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")

MANIFEST = ROOT / "data/processed/split_manifest.csv"
DVC_FILE = ROOT / "data/raw.dvc"

CLASSES = {
    0: "pawara",
    1: "uchitha",
    2: "dhamith",
    3: "unknown",
}


def main():
    print("\n===== FACEOPS DATABASE IMPORT =====")

    # Read dataset metadata
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # Read tracked dataset version
    with DVC_FILE.open() as f:
        dvc_info = yaml.safe_load(f)

    raw_version = dvc_info["outs"][0]["md5"]
    expected_count = dvc_info["outs"][0]["nfiles"]

    split_version = hashlib.sha256(
        MANIFEST.read_bytes()
    ).hexdigest()

    if len(rows) != expected_count:
        raise ValueError(
            "Manifest count does not match DVC dataset"
        )

    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image paths")

    for row in rows:
        label = int(row["label"])

        if CLASSES.get(label) != row["class_name"]:
            raise ValueError(
                f"Invalid class mapping: {row['path']}"
            )

        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid split: {row['path']}"
            )

        if not (ROOT / row["path"]).is_file():
            raise FileNotFoundError(row["path"])

    # Connect to PostgreSQL
    with psycopg.connect(
        host="127.0.0.1",
        port=5433,
        dbname="faceops_db",
        user="faceops_admin",
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=10,
    ) as conn:

        with conn.cursor() as cur:

            # Register dataset version
            cur.execute(
                """
                INSERT INTO dataset_versions (
                    raw_dvc_md5,
                    split_manifest_sha256,
                    total_images
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (
                    raw_dvc_md5,
                    split_manifest_sha256
                )
                DO UPDATE SET
                    total_images = EXCLUDED.total_images
                RETURNING id
                """,
                (
                    raw_version,
                    split_version,
                    len(rows),
                ),
            )

            version_id = cur.fetchone()[0]

            # Insert image metadata
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO images (
                        dataset_version_id,
                        class_label,
                        relative_path,
                        split,
                        width,
                        height,
                        sha256
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (
                        dataset_version_id,
                        relative_path
                    )
                    DO UPDATE SET
                        class_label = EXCLUDED.class_label,
                        split = EXCLUDED.split,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        sha256 = EXCLUDED.sha256
                    """,
                    (
                        version_id,
                        int(row["label"]),
                        row["path"],
                        row["split"],
                        int(row["width"]),
                        int(row["height"]),
                        row["sha256"],
                    ),
                )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM images
                WHERE dataset_version_id = %s
                """,
                (version_id,),
            )

            stored_count = cur.fetchone()[0]

            cur.execute(
                """
                SELECT split, COUNT(*)
                FROM images
                WHERE dataset_version_id = %s
                GROUP BY split
                ORDER BY split
                """,
                (version_id,),
            )

            split_counts = dict(cur.fetchall())

    print(f"Dataset version ID: {version_id}")
    print(f"DVC hash: {raw_version}")
    print(f"Images stored: {stored_count}")
    print(f"Splits: {split_counts}")
    print(f"Classes: {dict(Counter(r['class_name'] for r in rows))}")

    print("\nDatabase import completed.")


if __name__ == "__main__":
    main()
