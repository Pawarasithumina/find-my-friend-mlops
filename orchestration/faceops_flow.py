"""FaceOps TwinTrack: DVC preflight, two validation tracks, Optuna tuning."""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import boto3
import psycopg
import torch
from botocore.config import Config
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
EXPECTED_SPLITS = {"train": 144, "val": 32, "test": 33}
MODELS = ("custom_cnn", "mobilenet_v3", "resnet18")


@task(name="DVC Reproduction Gate")
def reproduce_dataset():
    result = subprocess.run(["dvc", "repro"], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("DVC repro failed:\n" + result.stdout + result.stderr)
    get_run_logger().info("DVC pipeline up to date / reproduced")
    return True


@task(name="Track A - Dataset Quality Gate")
def verify_dataset():
    logger = get_run_logger()
    seen = set()
    counts = {}
    for split, expected in EXPECTED_SPLITS.items():
        data = torch.load(ROOT / "data/tensors" / f"{split}.pt",
                          map_location="cpu", weights_only=True)
        images, labels, paths = data["images"], data["labels"], data["paths"]
        if images.shape != (expected, 3, 224, 224) or len(labels) != expected:
            raise ValueError(f"Unexpected {split} tensor shape")
        if len(paths) != expected or not torch.all((labels >= 0) & (labels < 4)):
            raise ValueError(f"Invalid labels/paths for {split}")
        if set(paths) & seen:
            raise ValueError(f"Same image path appears in multiple splits: {split}")
        seen.update(paths)
        counts[split] = expected
    if len(seen) != 209:
        raise ValueError("Expected 209 distinct paths")

    groups = {}
    with (ROOT / "data/processed/split_manifest.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["class_name"] != "unknown":
                continue
            stem = Path(row["path"]).stem
            match = re.match(r"^(.*?)[_-]?TIMEBURST\d+$", stem, re.IGNORECASE)
            session = match.group(1) if match else stem
            groups.setdefault(session, set()).add(row["split"])
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Unknown capture session crosses splits")

    logger.info("Dataset quality PASSED: %s", counts)
    logger.warning("Known classes have one capture session each; "
                   "accuracy is not independent-session generalization")
    return counts


@task(name="Track B - Infrastructure Gate", retries=2, retry_delay_seconds=5)
def verify_infrastructure():
    with urlopen("http://127.0.0.1:5050/health", timeout=10) as result:
        if result.status != 200:
            raise RuntimeError("MLflow not healthy")
    with psycopg.connect(host="127.0.0.1", port=5433, dbname="faceops_db",
                         user="faceops_admin", password=os.environ["POSTGRES_PASSWORD"],
                         connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM images WHERE dataset_version_id=2")
            if cur.fetchone()[0] != 209:
                raise ValueError("PostgreSQL dataset v2 missing 209 images")
    s3 = boto3.client("s3", endpoint_url="http://127.0.0.1:9000",
        aws_access_key_id=os.environ["RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUSTFS_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}))
    buckets = {b["Name"] for b in s3.list_buckets()["Buckets"]}
    if not {"faceops-datasets", "faceops-mlflow"}.issubset(buckets):
        raise RuntimeError("RustFS bucket missing")
    get_run_logger().info("MLflow, PostgreSQL, RustFS healthy")
    return True


@task(name="Optuna Model Branch", task_run_name="train-{model_name}")
def run_model(model_name: str, mode: str, trials: int, epochs: int):
    args = [sys.executable, "-m", "src.training.tune", "--model", model_name]
    if mode == "smoke":
        args.append("--smoke")
    else:
        args += ["--trials", str(trials), "--epochs", str(epochs)]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{model_name} Optuna failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
        )
    get_run_logger().info("%s finished:\n%s", model_name, result.stdout[-1300:])
    return model_name


@flow(name="FaceOps TwinTrack", log_prints=True)
def faceops_flow(mode: str = "preflight", trials: int = 6, epochs: int = 12):
    if mode not in ("preflight", "smoke", "tune"):
        raise ValueError(f"Unsupported mode: {mode}")
    logger = get_run_logger()
    logger.info("Starting FaceOps TwinTrack mode=%s", mode)
    reproduce_dataset()
    a = verify_dataset.submit()
    b = verify_infrastructure.submit()
    # Both independent tracks must finish successfully.
    a.result()
    b.result()
    logger.info("TwinTrack gates PASSED")

    if mode != "preflight":
        # Deliberately sequential for 16 GB RAM and CPU-only environments.
        for model_name in MODELS:
            run_model(model_name, mode, trials, epochs)

    logger.info("FaceOps TwinTrack completed: %s", mode)
    return {"mode": mode, "dataset_version": 2,
            "quality_gate": "passed", "infrastructure_gate": "passed",
            "training": "not_requested" if mode == "preflight" else "completed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "tune"),
                        default="preflight")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    print(faceops_flow(mode=args.mode, trials=args.trials, epochs=args.epochs))
