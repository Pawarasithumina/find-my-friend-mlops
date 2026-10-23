#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

python -m compileall -q src orchestration desktop scripts tests
find scripts -maxdepth 1 -name '*.sh' -type f -print0 | xargs -0 -r -n 1 bash -n
# Compose needs credentials for interpolation only; never print them.
set -a
source .env
set +a
docker compose -f compose.yaml -f compose.prefect.yaml -f compose.serving.yaml config --quiet

PYTHONPATH=. python tests/test_model_shapes.py
python -m unittest tests.test_desktop_client -v

if git ls-files | grep -E '^(data/raw/|data/tensors/|data/processed/|data/optuna/|reports/private_|\.env$|\.dvc/config\.local$)'; then
  echo 'ERROR: Private file tracked by Git.' >&2
  exit 1
fi

echo 'PASS: FaceOps local CI checks succeeded.'
