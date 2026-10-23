
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Runs inside the MLflow Docker container
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "FaceOps_Infrastructure_Test"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("\n===== FACEOPS MLFLOW CONNECTION TEST =====")

# Create a test experiment run
with mlflow.start_run(
    run_name="postgres_rustfs_connection_test"
) as run:

    mlflow.log_param("dataset_images", 209)
    mlflow.log_param("classes", 4)
    mlflow.log_param("test_type", "infrastructure")

    mlflow.log_metric("connection_success", 1.0)

    mlflow.log_text(
        "FaceOps MLflow artifact storage test successful.",
        "connection_check.txt"
    )

    run_id = run.info.run_id

print("Run ID:", run_id)

# Verify PostgreSQL tracking
client = MlflowClient()
saved_run = client.get_run(run_id)

assert saved_run.data.params["dataset_images"] == "209"
assert saved_run.data.metrics["connection_success"] == 1.0

print("PostgreSQL tracking: SUCCESS")

# Verify artifact storage
artifacts = client.list_artifacts(run_id)

assert any(
    artifact.path == "connection_check.txt"
    for artifact in artifacts
)

downloaded = client.download_artifacts(
    run_id,
    "connection_check.txt",
    dst_path="/tmp/faceops-mlflow-test"
)

assert Path(downloaded).read_text() == (
    "FaceOps MLflow artifact storage test successful."
)

print("RustFS artifact upload/download: SUCCESS")
print("\n===== ALL CHECKS PASSED =====")
