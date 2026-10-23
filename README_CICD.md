# FaceOps TwinTrack — CI/CD and Submission Handoff

This is the final MLOps delivery layer for a consent-based four-class *Find My Friend* lab.

## What CI validates

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests, or manual dispatch:

1. Python compilation and Bash shell syntax
2. Docker Compose configuration parsing (with dummy environment values, **without starting services**)
3. PyTorch model output shape/parameter smoke test using **synthetic tensors**
4. Desktop HTTP client unit tests using mocked requests
5. Git tracked-file audit to prevent committing raw images, tensors, evaluation reports, or credentials

**Important:** Hosted GitHub runners do not have the private face dataset, RustFS, PostgreSQL, MLflow, or Prefect. The CI job intentionally does **not** execute full model training, test evaluations, DVC reproduction, registry operations, or webcam tests.

## Controlled CD

`.github/workflows/release.yml` is manually triggered using the Actions tab. It publishes a source-only FastAPI serving container to GitHub Container Registry (GHCR), tagged with the Git commit SHA and `latest`. `docker/Dockerfile.serving` and `.dockerignore` exclude private data, and no production credentials are passed to the image build. GHCR image visibility/access depend on repository and package settings; verify them before publishing.

This implements *continuous delivery with manual approval*. It does **not** automatically deploy to a private Ubuntu laptop.

For local Docker deployment, **stop the native FastAPI process on port 8000 first**, then run:

```bash
cd ~/Desktop/find-my-friend-mlops
bash scripts/deploy_local.sh
bash scripts/start_desktop.sh
```

Deployment loads exactly two approved models from MLflow aliases and private RustFS. On startup failure the API is not considered ready. The script preserves Docker volumes.

## Verification and source-code commits

```bash
bash scripts/check_ci.sh
bash scripts/check_submission.sh
git status --short
```

Commit only public-safe files. `.env` and face images must never be uploaded. Final report is intentionally not generated in this phase.

## Suggested report CI/CD explanation

Git push / PR -> GitHub Actions CI (syntax, contracts, Compose validation, privacy gate) -> manually approved serving image release to GHCR -> local operator deploys the serving service -> startup health check / `/models` verification -> PySide6 desktop. Prefect TwinTrack independently orchestrates DVC data checks, infrastructure checks, and Optuna training, while MLflow manages artifacts and model approval. These are distinct pipeline layers.

## Suggested screenshots for the PDF report

- GitHub Actions CI successful run
- GitHub Actions published image (when configured)
- DVC DAG and `dvc status`
- Prefect flow / task execution
- MLflow experiment runs and model registry aliases
- FastAPI OpenAPI docs, `/health`, `/models`
- Desktop webcam UI with both model outputs (get participant consent)
