"""Load exactly the two MLflow-registry approved FaceOps models.

Fetches the authoritative registered version aliases from MLflow, resolves their
source trial run, and retrieves the corresponding versioned state_dict directly
from private RustFS to avoid timeout-prone MLflow proxy checkpoint downloads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import torch
from botocore.config import Config
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.models.architectures import create_model

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
EXPECTED_CLASSES = ["pawara", "uchitha", "dhamith", "unknown"]
APPROVED_MODELS = {
    "model-a": ("mobilenet_v3", "faceops_mobilenet_v3"),
    "model-b": ("resnet18", "faceops_resnet18"),
}


@dataclass
class LoadedModel:
    architecture: str
    registry_name: str
    registry_version: str
    source_trial_id: str
    model: torch.nn.Module


def _checkpoint_from_rustfs(run_id: str, destination: Path) -> Path:
    # We deliberately do not infer or hardcode an S3 object key from run IDs.
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("FACEOPS_S3_ENDPOINT", "http://127.0.0.1:9000"),
        aws_access_key_id=os.environ["RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUSTFS_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=90,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )
    hits = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="faceops-mlflow", Prefix="artifacts/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if f"/{run_id}/" in f"/{key}/" and key.endswith("/checkpoint/best_state.pt"):
                hits.append((key, int(obj["Size"])))
    if len(hits) != 1:
        raise RuntimeError(f"Found {len(hits)} checkpoints for approved trial run {run_id}")
    key, expected_size = hits[0]
    if not 100 < expected_size < 250 * 1024 * 1024:
        raise RuntimeError("Unexpected model checkpoint size")
    target = destination / "best_state.pt"
    response = s3.get_object(Bucket="faceops-mlflow", Key=key)
    try:
        with target.open("wb") as f:
            for chunk in response["Body"].iter_chunks(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    finally:
        response["Body"].close()
    if target.stat().st_size != expected_size:
        target.unlink(missing_ok=True)
        raise RuntimeError("Incomplete checkpoint download")
    return target


def load_approved_models() -> dict[str, LoadedModel]:
    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    result = {}
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))

    for slot, (architecture, registered_name) in APPROVED_MODELS.items():
        version = client.get_model_version_by_alias(registered_name, "production")
        if not version.run_id:
            raise RuntimeError(f"Registered model {registered_name} has no registration run ID")
        registration_run = client.get_run(version.run_id)
        params = registration_run.data.params
        if params.get("architecture") != architecture or params.get("dataset_version") != "2":
            raise RuntimeError(f"Registry metadata mismatch for {registered_name}")
        trial_id = params.get("source_run_id")
        if not trial_id or len(trial_id) != 32:
            raise RuntimeError(f"Missing trial provenance for {registered_name}")
        trial = client.get_run(trial_id)
        if trial.data.params.get("model_name") != architecture or trial.data.params.get("dataset_version") != "2":
            raise RuntimeError("Source training run does not match registered model")

        with TemporaryDirectory() as temporary:
            checkpoint_file = _checkpoint_from_rustfs(trial_id, Path(temporary))
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != architecture or checkpoint.get("dataset_version") != 2:
            raise RuntimeError("Checkpoint identity mismatch")
        if list(checkpoint.get("classes", [])) != EXPECTED_CLASSES:
            raise RuntimeError("Checkpoint class label ordering mismatch")
        model = create_model(
            architecture,
            num_classes=4,
            dropout=float(checkpoint["hyperparameters"]["dropout"]),
            pretrained=False,
            freeze_backbone=bool(checkpoint["freeze_backbone"]),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        result[slot] = LoadedModel(
            architecture=architecture,
            registry_name=registered_name,
            registry_version=str(version.version),
            source_trial_id=trial_id,
            model=model,
        )
        print(f"[FaceOps] Loaded {registered_name} v{version.version} from approved MLflow alias")
    return result
