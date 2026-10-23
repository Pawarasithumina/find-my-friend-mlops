#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

MODE="${1:-preflight}"
TRIALS="${2:-6}"
EPOCHS="${3:-12}"
export PREFECT_API_URL="${PREFECT_API_URL:-http://127.0.0.1:4200/api}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5050}"

case "$MODE" in
  preflight|smoke|tune) ;;
  *) echo "Usage: $0 {preflight|smoke|tune} [additional_trials] [epochs]"; exit 2 ;;
esac

python -m orchestration.faceops_flow \
  --mode "$MODE" --trials "$TRIALS" --epochs "$EPOCHS"
