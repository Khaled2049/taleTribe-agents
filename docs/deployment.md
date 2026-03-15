# Deployment

`novelsync-agents` deploys to Google Cloud Run through GitHub Actions, Artifact Registry, Secret Manager, and Terraform.

## Runtime shape

- container image built from `Dockerfile`
- Cloud Run service name: `novelsync-agents`
- region: `us-central1`
- Artifact Registry repo: `novelsync-agents`
- Terraform directory: `terraform/`

## GitHub workflows

### `.github/workflows/pr-check.yml`

Runs on pull requests to `main` and validates:

- `black --check .`
- `isort --check-only .`
- `ruff check .`
- `pytest tests/ -v`
- `terraform fmt -check -recursive`
- `terraform validate`

### `.github/workflows/deploy.yml`

Runs on pushes to `main` and manual dispatch. The workflow:

1. authenticates to GCP through Workload Identity Federation
2. creates Artifact Registry and Firestore if needed
3. builds and pushes Docker images tagged with `${github.sha}` and `latest`
4. creates or updates the `google-ai-studio-api-key` secret in Secret Manager
5. initializes Terraform and applies the Cloud Run infrastructure
6. reads the deployed service URL and performs a `/health` check

Markdown-only changes are ignored by the deploy workflow because it has `paths-ignore` for `*.md`.

## Required secrets

GitHub Actions expects:

- `WIF_PROVIDER`
- `WIF_SERVICE_ACCOUNT`
- `GOOGLE_AI_STUDIO_API_KEY`

## Local Docker build

```bash
docker build --platform=linux/amd64 -t novelsync-agents:local .
```

The image uses `requirements-prod.txt`, which excludes the local image-generation dependencies.

## Terraform notes

`terraform/` manages the Cloud Run service and related IAM wiring. The deploy workflow can also import existing resources into an empty state before running `terraform plan`.

For local Terraform validation:

```bash
cd terraform
terraform init -backend=false
terraform validate
```

## Operational notes

- keep `.env` local and uncommitted
- keep `terraform.tfvars` free of secrets
- rotate the Google AI Studio key in GitHub Secrets and redeploy when needed
- if you need local image generation, run the app locally with `requirements.txt`; production intentionally uses the slimmer dependency set
