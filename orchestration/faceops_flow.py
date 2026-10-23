
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

EXPECTED_SPLITS = {
    "train": 144,
    "val": 32,
    "test": 33,
}

MODELS = [
    "custom_cnn",
    "mobilenet_v3",
    "resnet18",
]


# ==========================================
# TASK 1: DVC REPRODUCIBILITY
# ==========================================

@task(name="DVC Reproduction Gate")
def reproduce_dataset():

    logger = get_run_logger()

    result = subprocess.run(
        ["dvc", "repro"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "DVC reproduction failed:\n"
            + result.stdout
            + result.stderr
        )

    logger.info(
        "DVC pipeline reproduced successfully."
    )

    return True


# ==========================================
# TRACK A: DATA QUALITY
# ==========================================

@task(name="Track A - Dataset Quality Gate")
def verify_dataset():

    logger = get_run_logger()

    all_paths = []
    counts = {}

    for split, expected in EXPECTED_SPLITS.items():

        path = ROOT / "data/tensors" / f"{split}.pt"

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        images = data["images"]
        labels = data["labels"]
        paths = data["paths"]

        if len(images) != expected:
            raise ValueError(
                f"{split}: expected {expected} images"
            )

        if images.shape[1:] != (3, 224, 224):
            raise ValueError(
                f"{split}: invalid tensor shape"
            )

        if len(labels) != len(paths):
            raise ValueError(
                f"{split}: labels/paths mismatch"
            )

        if not torch.all(
            (labels >= 0) & (labels < 4)
        ):
            raise ValueError(
                f"{split}: invalid labels"
            )

        all_paths.extend(paths)
        counts[split] = len(images)

    if len(all_paths) != len(set(all_paths)):
        raise ValueError(
            "Dataset leakage: duplicate paths"
        )

    # Check capture-session grouping for Unknown.
    manifest = (
        ROOT / "data/processed/split_manifest.csv"
    )

    unknown_sessions = {}

    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):

            if row["class_name"] != "unknown":
                continue

            stem = Path(row["path"]).stem

            match = re.match(
                r"^(.*?)[_-]?TIMEBURST\d+$",
                stem,
                re.IGNORECASE
            )

            session = (
                match.group(1) if match else stem
            )

            unknown_sessions.setdefault(
                session, set()
            ).add(row["split"])

    for session, splits in unknown_sessions.items():
        if len(splits) > 1:
            raise ValueError(
                f"Unknown session crosses splits: {session}"
            )

    logger.info("Dataset quality gate PASSED.")
    logger.info("Split counts: %s", counts)

    logger.warning(
        "Known classes each have one capture session. "
        "Session-independent generalization cannot "
        "be established from the current dataset."
    )

    return counts


# ==========================================
# TRACK B: INFRASTRUCTURE
# ==========================================

@task(
    name="Track B - Infrastructure Gate",
    retries=2,
    retry_delay_seconds=5,
)
def verify_infrastructure():

    logger = get_run_logger()

    # MLflow
    with urlopen(
        "http://127.0.0.1:5050/health",
        timeout=10
    ) as response:

        if response.status != 200:
            raise RuntimeError(
                "MLflow health check failed"
            )

    # PostgreSQL
    with psycopg.connect(
        host="127.0.0.1",
        port=5433,
        dbname="faceops_db",
        user="faceops_admin",
        password=os.environ["POSTGRES_PASSWORD"],
    ) as conn:

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM images
                WHERE dataset_version_id = %s
                """,
                (2,)
            )

            count = cur.fetchone()[0]

            if count != 209:
                raise ValueError(
                    "PostgreSQL dataset version mismatch"
                )

    # RustFS
    s3 = boto3.client(
        "s3",
        endpoint_url="http://127.0.0.1:9000",
        aws_access_key_id=os.environ[
            "RUSTFS_ACCESS_KEY"
        ],
        aws_secret_access_key=os.environ[
            "RUSTFS_SECRET_KEY"
        ],
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"}
        )
    )

    buckets = {
        bucket["Name"]
        for bucket in s3.list_buckets()["Buckets"]
    }

    required = {
        "faceops-datasets",
        "faceops-mlflow",
    }

    if not required.issubset(buckets):
        raise ValueError(
            "Required RustFS buckets are missing"
        )

    logger.info(
        "MLflow, PostgreSQL and RustFS verified."
    )

    return True


# ==========================================
# MODEL BRANCH: OPTUNA INTEGRATION
# ==========================================

@task(
    name="Optuna Model Branch",
    task_run_name="optuna-{model_name}",
)
def run_model_smoke(model_name: str):

    logger = get_run_logger()

    command = [
        sys.executable,
        "-m",
        "src.training.tune",
        "--model",
        model_name,
        "--smoke",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{model_name} failed:\n"
            + result.stdout[-3000:]
            + result.stderr[-3000:]
        )

    logger.info(
        "%s Optuna smoke test completed.",
        model_name
    )

    logger.info(
        "Training summary:\n%s",
        result.stdout[-1500:]
    )

    return model_name


# ==========================================
# FACEOPS TWINTRACK ORCHESTRATOR
# ==========================================

@flow(
    name="FaceOps TwinTrack",
    log_prints=True,
)
def faceops_flow(mode: str = "preflight"):

    logger = get_run_logger()

    if mode not in {"preflight", "smoke"}:
        raise ValueError("Invalid execution mode")

    logger.info(
        "Starting FaceOps TwinTrack: %s", mode
    )

    # Stage 1: Reproducible data preparation.
    reproduce_dataset()

    # Stage 2: Independent validation tracks.
    data_future = verify_dataset.submit()
    infra_future = verify_infrastructure.submit()

    # Both gates must pass.
    data_result = data_future.result()
    infra_result = infra_future.result()

    if not data_result or not infra_result:
        raise RuntimeError("Preflight gate failed")

    logger.info("Both TwinTrack gates PASSED.")

    if mode == "smoke":
        # Sequential model execution limits RAM use.
        for model_name in MODELS:
            run_model_smoke(model_name)

    logger.info(
        "FaceOps TwinTrack completed: %s", mode
    )

    return {
        "mode": mode,
        "dataset_version": 2,
        "quality_gate": "passed",
        "infrastructure_gate": "passed",
        "training": (
            "smoke_complete"
            if mode == "smoke"
            else "not_requested"
        )
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["preflight", "smoke"],
        default="preflight"
    )

    args = parser.parse_args()

    print(faceops_flow(mode=args.mode))
