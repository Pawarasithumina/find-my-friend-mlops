#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

required=(
  compose.yaml compose.prefect.yaml compose.serving.yaml
  docker/Dockerfile.mlflow docker/Dockerfile.serving
  docker/postgres/schema.sql
  dvc.yaml dvc.lock data/raw.dvc
  orchestration/faceops_flow.py src/training/tune.py
  src/serving/api.py desktop/main.py
  .github/workflows/ci.yml .github/workflows/release.yml
  scripts/start_serving.sh scripts/start_desktop.sh
)
for file in "${required[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "MISSING: $file" >&2
    exit 1
  fi
done

if git ls-files | grep -E '^(data/raw/|data/tensors/|data/processed/|data/optuna/|reports/private_|\.env$|\.dvc/config\.local$)'; then
  echo 'ERROR: private data or secrets tracked in Git.' >&2
  exit 1
fi

printf 'PASS: Required project and submission files are present.\n'
printf 'NOTE: PDF report must be created later from final experiment results.\n'
