"""Retrieve FaceOps trial checkpoints directly from private local RustFS.

Bypasses the MLflow proxy for large artifact downloads; MLflow remains the
system of record for run IDs, metrics, and model registration. Never logs keys.
"""
import argparse
import os
import re
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "faceops-mlflow"
RUN_IDS = {
    "mobilenet_v3": "419640ebb60644a9bf96c97c15254dce",
    "resnet18": "499828f2c005446ca3c7eb6adae41b85",
    "custom_cnn": "4af1ae38b417489b80a14c0086b12035",
}


def s3_client():
    load_dotenv(ROOT / ".env")
    access_key = os.getenv("RUSTFS_ACCESS_KEY")
    secret_key = os.getenv("RUSTFS_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Missing RustFS credentials in project .env")
    return boto3.client(
        "s3",
        endpoint_url="http://127.0.0.1:9000",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=90,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def locate_checkpoint(client, run_id):
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("Expected an exact 32-character MLflow run ID")
    matches = []
    # The MLflow artifact root is s3://faceops-mlflow/artifacts,
    # but keys are discovered, not guessed from an MLflow HTTP URI.
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="artifacts/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if f"/{run_id}/" in f"/{key}/" and key.endswith(
                "/checkpoint/best_state.pt"
            ):
                matches.append((key, int(obj["Size"])))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one checkpoint object for MLflow run {run_id}; "
            f"found {len(matches)}. Check RustFS {BUCKET} and the MLflow run."
        )
    key, size = matches[0]
    if size < 100 or size > 250 * 1024 * 1024:
        raise RuntimeError(f"Unexpected checkpoint size for run {run_id}: {size} bytes")
    return key, size


def download_trial_checkpoint(run_id, destination_directory):
    client = s3_client()
    key, expected_size = locate_checkpoint(client, run_id)
    directory = Path(destination_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "best_state.pt"
    temporary = directory / "best_state.pt.partial"
    print(f"[RustFS direct] Retrieving run {run_id[:8]}... ({expected_size / 1048576:.1f} MiB)", flush=True)
    try:
        response = client.get_object(Bucket=BUCKET, Key=key)
        actual_size = 0
        try:
            with temporary.open("wb") as out:
                for chunk in response["Body"].iter_chunks(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
                        actual_size += len(chunk)
        finally:
            response["Body"].close()
        if actual_size != expected_size:
            raise IOError(
                f"Incomplete RustFS checkpoint: expected {expected_size}, got {actual_size} bytes"
            )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[RustFS direct] Download verified: {target.name}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="List checkpoint existence and size without downloading")
    args = parser.parse_args()
    if not args.check:
        parser.error("Use --check; the evaluation and registration scripts perform downloads")
    client = s3_client()
    for name, run_id in RUN_IDS.items():
        _, size = locate_checkpoint(client, run_id)
        print(f"{name:14s} checkpoint: FOUND ({size / 1048576:.1f} MiB)")
    print("All 3 source checkpoints found in private RustFS.")


if __name__ == "__main__":
    main()
