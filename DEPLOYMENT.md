# Deployment Guide: Terraform + GitHub Actions

This guide covers deploying NovelSync Agents to Google Cloud Run using Terraform for infrastructure and GitHub Actions for CI/CD.

**Quick Links:**

- [Architecture Overview](#architecture-overview)
- [Security](#security)
- [Free tier](#free-tier)
- [Prerequisites Setup](#prerequisites-setup)
- [GitHub Configuration](#github-repository-secrets)
- [First Deployment](#first-deployment)
- [Ongoing Operations](#ongoing-operations)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

```
GitHub Repository
    ↓
    ├─→ GitHub Actions CI/CD Workflows
    │   ├─→ ci.yml (lint & test on every push)
    │   ├─→ deploy.yml (build & deploy on main)
    │   └─→ pr-check.yml (validate PRs)
    ↓
Google Cloud Platform
    ├─→ Workload Identity Federation (keyless auth)
    ├─→ Artifact Registry (Docker images)
    ├─→ Cloud Run (FastAPI server)
    ├─→ Firestore (data storage)
    └─→ Secret Manager (API keys)
```

## Security

This setup follows security best practices:

| Practice | How we do it |
|----------|--------------|
| **No long-lived keys** | GitHub Actions uses **Workload Identity Federation** (OIDC). No service account JSON keys are stored in GitHub or on disk. |
| **Least privilege** | The Cloud Run service account (`novelsync-agents-run`) has only `roles/secretmanager.secretAccessor` (one secret) and `roles/datastore.user`. It cannot modify IAM or other project resources. |
| **Secrets in Secret Manager** | The Google AI Studio API key is stored in Secret Manager. Cloud Run receives it via environment injection; the value never appears in code or in Terraform. |
| **GitHub Secrets** | `GOOGLE_AI_STUDIO_API_KEY`, `WIF_PROVIDER`, and `WIF_SERVICE_ACCOUNT` are stored as GitHub Actions secrets. Never commit `.env` or `terraform.tfvars` with real keys (both are in `.gitignore`). |
| **Public access is optional** | By default the service allows unauthenticated invocations (`enable_public_access = true`). For authenticated-only access, set `enable_public_access = false` in Terraform and grant `roles/run.invoker` to specific identities. |
| **Key rotation** | Rotate the API key in Google AI Studio, update the GitHub secret `GOOGLE_AI_STUDIO_API_KEY`, then redeploy; the workflow pushes the new key to Secret Manager (GitHub masks secret values in logs). |
| **Deploy pipeline** | The workflow never persists the API key to the repo or to Terraform; it is passed only from GitHub Secrets to Secret Manager. |

**Do not:** commit `.env` or `terraform.tfvars` with secrets, grant the GitHub Actions SA more roles than listed in Prerequisites, or use a service account key instead of WIF.

## Free tier

The stack is tuned to stay within **GCP free tier** where possible:

| Service | Free tier (approx.) | How we stay within it |
|---------|---------------------|------------------------|
| **Cloud Run** | 2M requests/month, 360K GB-seconds memory, 180K vCPU-seconds | `min_instances = 0` (scale to zero), `cpu_idle = true` (no CPU charge when idle), 1 vCPU, 1 Gi memory, `max_instances = 5` to cap cost spikes. |
| **Firestore** | 1 GB storage, 50K reads/day, 20K writes/day | Use the default database; typical app usage stays under these limits. |
| **Artifact Registry** | 0.5 GB storage per region (then paid) | Keep only a few image tags (e.g. `latest` + recent SHAs); delete old images if you approach the limit. |
| **Secret Manager** | 6 active secret versions free | One secret with 1–2 versions is well within free tier. |
| **Cloud Build / GitHub Actions** | N/A | Builds run on GitHub-hosted runners; no GCP build minutes used. |

**Recommendation:** Enable [billing alerts](https://console.cloud.google.com/billing/budgets) (e.g. alert at $1 and $10) so you are notified if usage grows.

## Prerequisites Setup

Run these commands **once** to prepare GCP for deployment. This sets up:

- Required APIs
- Cloud Run and GitHub Actions service accounts
- Workload Identity Federation (keyless authentication; no keys stored)
- Secret Manager secret for the API key

**Note:** The Artifact Registry repository and Firestore database can be created by the deploy workflow on first run, or you can create them in these steps.

### 1. Configure GCP Project

```bash
# Set default project
gcloud config set project story-6f89f

# Enable required APIs (takes 1-2 minutes)
gcloud services enable \
  run.googleapis.com \
  firestore.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  iam.googleapis.com

# Verify billing account is linked
gcloud beta billing projects describe story-6f89f
```

### 2. Create Firestore database (optional before first deploy)

The deploy workflow can create the default Firestore database if it does not exist. To create it manually:

```bash
gcloud firestore databases create \
  --location=us-central1 \
  --type=firestore-native \
  --project=story-6f89f

gcloud firestore databases list --project=story-6f89f
```

### 3. Create Cloud Run service account (required, one-time)

Terraform references this SA; it does not create it (so CI does not need `iam.serviceAccounts.create`). Create it once:

```bash
gcloud iam service-accounts create novelsync-agents-run \
  --display-name="NovelSync Agents Cloud Run Service Account" \
  --project=story-6f89f
```

### 4. Create Service Account for GitHub Actions

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Deployment" \
  --project=story-6f89f

# Grant necessary IAM roles
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/secretmanager.admin"
```

### 5. Set Up Workload Identity Federation (Keyless Auth)

Workload Identity Federation allows GitHub Actions to authenticate to GCP **without storing service account keys**.

```bash
# Get project number (needed below)
PROJECT_NUMBER=$(gcloud projects describe story-6f89f --format="value(projectNumber)")
echo "Project Number: $PROJECT_NUMBER"

# Create Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --project="story-6f89f" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create Workload Identity Provider (OIDC for GitHub)
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="story-6f89f" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Actions Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'Khaled2049'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Allow GitHub Actions to impersonate service account
# Replace PROJECT_NUMBER with the value from above
# ⚠️ IMPORTANT: GitHub usernames are case-sensitive (use "Khaled2049" not "khaled2049")
gcloud iam service-accounts add-iam-policy-binding \
  github-actions@story-6f89f.iam.gserviceaccount.com \
  --project=story-6f89f \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/Khaled2049/novelsync-agents"

# Get Workload Identity Provider resource name (needed for GitHub Secrets)
WIF_PROVIDER=$(gcloud iam workload-identity-pools providers describe "github-provider" \
  --project="story-6f89f" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)")

echo "WIF_PROVIDER: $WIF_PROVIDER"
# Save this for GitHub Secrets - format:
# projects/{PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

### 6. Add IAM roles for GitHub Actions

The GitHub Actions service account needs these roles to deploy (create/update Cloud Run, push images, manage secrets, etc.):

```bash
# Enable and list APIs
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/serviceusage.serviceUsageAdmin"

# Required for Terraform (IAM bindings, etc.)
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/iam.securityAdmin"

# Firestore (workflow may create DB; Terraform does not manage it)
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/datastore.owner"
```

### 7. Create Secret in Secret Manager

```bash
# Create secret with your API key
echo "YOUR_API_KEY_HERE" | \
  gcloud secrets create google-ai-studio-api-key \
  --data-file=- \
  --replication-policy=automatic
```

## GitHub Repository Secrets

Configure these secrets in your GitHub repository:
**Settings → Secrets and variables → Actions**

### Required Secrets

1. **WIF_PROVIDER** (from step 4 above)
   - Value: `projects/{PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider`

2. **WIF_SERVICE_ACCOUNT**
   - Value: `github-actions@story-6f89f.iam.gserviceaccount.com`

3. **GOOGLE_AI_STUDIO_API_KEY**
   - Value: Your API key from https://aistudio.google.com/app/apikey
   - **Security:** Never commit this to the repo. Keep it only in GitHub Secrets and in Secret Manager. Rotate it if it was ever exposed (e.g. in `.env` that was committed).

### How to Add Secrets in GitHub

1. Go to: https://github.com/khaled2049/novelsync-agents/settings/secrets/actions
2. Click "New repository secret"
3. Add each secret:
   - Name: `WIF_PROVIDER`
   - Value: (paste the full provider resource name)
4. Repeat for other secrets

## First Deployment

### Step 0: Docker image size (recommended)

The full `requirements.txt` includes large ML libraries not needed for Cloud Run (image generation is disabled in production). The repo uses `requirements-prod.txt` in the Dockerfile to keep the image small (~500MB instead of ~8GB).

### Step 1: Prerequisites and secrets

1. Complete [Prerequisites Setup](#prerequisites-setup) (APIs, Cloud Run SA, GitHub Actions SA, WIF, secret).
2. Add [GitHub Repository Secrets](#github-repository-secrets): `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`, `GOOGLE_AI_STUDIO_API_KEY`.

The deploy workflow creates the Artifact Registry repository and Firestore database if they do not exist. You can create them manually in Prerequisites if you prefer.

### Step 2: Deploy

Push to `main` or run the workflow manually (Actions → Deploy to Cloud Run → Run workflow):

```bash
git add .
git commit -m "feat: deploy to Cloud Run"
git push origin main
```

The workflow will: build the image, push to Artifact Registry, create the secret (if missing) or add a new version so the live key matches GitHub Secrets, create Firestore if missing, run Terraform (APIs, IAM, Cloud Run service), and run a health check.

### Step 3: Verify deployment

After a deploy, the workflow summary shows the service URL. Or get it locally (if you have Terraform state) or from GCP:

```bash
# If you ran Terraform locally:
SERVICE_URL=$(cd terraform && terraform output -raw service_url)

# Or from gcloud:
SERVICE_URL=$(gcloud run services describe novelsync-agents --region=us-central1 --format='value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health
# Expected response: {"status":"healthy","project_id":"story-6f89f",...}

# Test agent endpoint
curl -X POST $SERVICE_URL/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action": "generateNextLines",
    "parameters": {
      "storyId": "test-story",
      "content": "Once upon a time",
      "cursorPosition": 17
    }
  }'
# Expected response: {"success":true,"data":{...}}
```

### Step 4: Keep secrets out of the repo

- **Never commit** `.env` or `terraform.tfvars` with real API keys (both are in `.gitignore`).
- If a key was ever committed: remove it from history, rotate the key in Google AI Studio, update the GitHub secret and Secret Manager, then redeploy.

## Ongoing Operations

### Deploying Updates

**Automatic (recommended):**

```bash
# Make changes
git add .
git commit -m "feat: Add new feature"
git push origin main

# GitHub Actions automatically:
# 1. Runs tests and linting
# 2. Builds Docker image
# 3. Pushes to Artifact Registry
# 4. Applies Terraform (updates Cloud Run)
# 5. Tests the deployment

# Monitor at: https://github.com/khaled2049/novelsync-agents/actions
```

**Manual (if needed):**

```bash
# Build and push image
docker build -t us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:v1.1 .
docker push us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:v1.1

# Update Cloud Run with Terraform
cd terraform
terraform apply -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:v1.1"
```

### Viewing Logs

```bash
# Real-time logs
gcloud run services logs tail novelsync-agents --region=us-central1

# Last 50 lines
gcloud run services logs read novelsync-agents \
  --region=us-central1 \
  --limit=50

# Errors only
gcloud run services logs read novelsync-agents \
  --region=us-central1 \
  --filter="severity>=ERROR" \
  --limit=20

# View in Cloud Console
# https://console.cloud.google.com/run/detail/us-central1/novelsync-agents/logs
```

### Updating secrets

To rotate the Google AI Studio API key:

1. Create a new key at https://aistudio.google.com/app/apikey
2. Update the GitHub secret `GOOGLE_AI_STUDIO_API_KEY` (Settings → Secrets and variables → Actions)
3. Push to `main` or re-run the deploy workflow — it will add a new Secret Manager version so Cloud Run uses the new key
4. Delete the old key in Google AI Studio

Alternatively, add a new version manually:  
`echo "NEW_API_KEY" | gcloud secrets versions add google-ai-studio-api-key --data-file=-`  
Cloud Run uses the latest version automatically.

### Monitoring costs

See [Free tier](#free-tier) for how the stack stays within free limits. Recommended:

```bash
# Confirm billing is linked
gcloud billing projects describe story-6f89f

# Set up billing alerts (e.g. $1 and $10) in Cloud Console
# https://console.cloud.google.com/billing/budgets
```

### Tear-down (delete all resources)

To remove everything Terraform manages (Cloud Run, IAM bindings; Artifact Registry, Firestore, and the secret are created outside Terraform):

```bash
cd terraform
terraform init
terraform destroy -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:latest"
# Type yes when prompted
```

APIs remain enabled. To remove Firestore or the secret, delete them in GCP (e.g. `gcloud firestore databases delete --database="(default)"`, `gcloud secrets delete google-ai-studio-api-key`). See [Free tier](#free-tier) for cost impact.

## Rollback Procedures

If a deployment introduces issues:

### Option 1: Revert Git Commit (Recommended)

```bash
# Revert the problematic commit
git revert HEAD
git push origin main

# GitHub Actions automatically redeploys with the previous version
# Monitor at: https://github.com/khaled2049/novelsync-agents/actions
```

### Option 2: Deploy Previous Docker Image

```bash
# List recent images
gcloud artifacts docker images list us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app

# Deploy previous image
cd terraform
terraform apply -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:PREVIOUS_COMMIT_SHA"
```

### Option 3: Use Cloud Run Traffic Splitting

```bash
# View recent revisions
gcloud run services describe novelsync-agents --region=us-central1

# Route all traffic to a previous revision
gcloud run services update-traffic novelsync-agents \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=us-central1
```

## Real-World Issues & Solutions

### Docker Image Too Large (~8GB)

**Problem:** Docker image is too large, takes forever to push to registry.

**Root Cause:** `requirements.txt` includes PyTorch, Diffusers, and other large ML libraries that aren't needed in Cloud Run.

**Solution:**
1. Create `requirements-prod.txt` without ML dependencies
2. Update `Dockerfile` to use `requirements-prod.txt`
3. Update `.github/workflows/ci.yml` to use `requirements-prod.txt`
4. Rebuild and push: `docker build --platform=linux/amd64 -t ... .`

Result: Image size drops from ~8GB to ~500MB ✅

### GitHub Actions Authentication Fails

**Problem:** `failed to generate Google Cloud federated token` or `The given credential is rejected by the attribute condition`

**Root Cause:** GitHub username is case-sensitive in Workload Identity Federation. If you use `khaled2049` but your username is `Khaled2049`, authentication fails.

**Solution:**
```bash
# Update Workload Identity Provider with CORRECT capitalization
gcloud iam workload-identity-pools providers update-oidc "github-provider" \
  --project="story-6f89f" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --attribute-condition="assertion.repository_owner == 'Khaled2049'"  # Use YOUR exact capitalization

# Update IAM binding with CORRECT capitalization
gcloud iam service-accounts add-iam-policy-binding \
  github-actions@story-6f89f.iam.gserviceaccount.com \
  --project=story-6f89f \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/Khaled2049/novelsync-agents"
```

### Terraform: Service Account Can't Create Resources

**Problem:** `Permission 'iam.serviceAccounts.create' denied` or `Failed to list services`

**Root Cause:** GitHub Actions service account is missing required IAM roles.

**Solution:**
```bash
# Add all required roles to the service account
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/serviceusage.serviceUsageAdmin"

gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/iam.securityAdmin"

gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/datastore.owner"
```

### Terraform: Resource already exists (409)

**Problem:** `Error 409: Resource 'novelsync-agents' already exists` (Cloud Run) or similar for another resource.

**Root Cause:** The resource exists in GCP but Terraform state (e.g. in CI) does not track it.

**Solution:**
- **Cloud Run:** Delete the service so Terraform can create it:  
  `gcloud run services delete novelsync-agents --region=us-central1 --project=story-6f89f`  
  Then re-run the deploy workflow or `terraform apply`.
- **Artifact Registry, Secret, Firestore:** The current setup uses data sources or workflow-created resources; Terraform does not create these. If you see 409 on them, ensure you are not still defining them as `resource` blocks (they should be `data` or created by the workflow only).

### Docker Build: Architecture Mismatch

**Problem:** Cloud Run: `Application failed to start: failed to load /usr/local/bin/python: exec format error`

**Root Cause:** Docker image built on M1/M2 Mac (ARM64) but Cloud Run runs on x86_64.

**Solution:**
```bash
# Always use --platform=linux/amd64 when building
docker build --platform=linux/amd64 \
  -t us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:initial \
  .
```

## Troubleshooting

### GitHub Actions Workflows Fail

**Problem:** Workflow shows "Authentication failed"

```bash
# Solution: Verify Workload Identity is configured correctly
gcloud iam service-accounts get-iam-policy \
  github-actions@story-6f89f.iam.gserviceaccount.com

# Check GitHub Secrets are set
# https://github.com/khaled2049/novelsync-agents/settings/secrets/actions
```

**Problem:** "Resource already exists" (409)

- For **Cloud Run:** delete the service with `gcloud run services delete novelsync-agents --region=us-central1 --project=story-6f89f`, then redeploy.
- **Firestore** is created by the workflow if missing; Terraform does not manage it. No import needed.

### Cloud Run Deployment Fails

**Problem:** "Permission denied" error

```bash
# Solution: Verify Cloud Run service account has necessary permissions
gcloud projects get-iam-policy story-6f89f \
  --flatten="bindings[].members" \
  --filter="bindings.members:novelsync-agents-run@"
```

**Problem:** "Secret not found" error

```bash
# Solution: Create the secret
echo "YOUR_API_KEY" | gcloud secrets create google-ai-studio-api-key --data-file=-

# Or add a new version
echo "YOUR_API_KEY" | gcloud secrets versions add google-ai-studio-api-key --data-file=-
```

**Problem:** Container image not found

```bash
# Solution: Verify image was pushed to Artifact Registry
gcloud artifacts docker images list us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app

# If missing, push manually
docker build -t us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:TAG .
docker push us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:TAG
```

### Terraform Issues

**Problem:** "API not enabled" error

```bash
# Solution: Enable the required APIs
gcloud services enable run.googleapis.com firestore.googleapis.com secretmanager.googleapis.com
```

**Problem:** State file not found or corrupted

```bash
# Solution: Reinitialize Terraform
cd terraform
rm -rf .terraform
rm .terraform.lock.hcl
terraform init
```

**Problem:** Firestore database already exists

Terraform no longer creates the Firestore database; the deploy workflow creates it if missing. No import or resource change needed.

### Testing Deployment

```bash
# Test health endpoint
curl https://novelsync-agents-XXX-uc.a.run.app/health

# Test with an agent endpoint
curl -X POST https://novelsync-agents-XXX-uc.a.run.app/agent/execute \
  -H "Content-Type: application/json" \
  -d '{"action":"generateNextLines","parameters":{"storyId":"test","content":"Once","cursorPosition":4}}'

# Check logs if it fails
gcloud run services logs read novelsync-agents --region=us-central1 --limit=50
```

## Documentation Reference

- **Terraform:** See [terraform/README.md](./terraform/README.md)
- **GitHub Actions:** See [.github/workflows/](./github/workflows/)
- **Local Development:** See [LOCAL_DEVELOPMENT.md](./LOCAL_DEVELOPMENT.md)
- **GCP Documentation:** https://cloud.google.com/run/docs

## FAQ

**Q: Why use Workload Identity Federation?**
A: It allows keyless authentication - GitHub Actions don't need service account keys, improving security.

**Q: How much will this cost?**
A: With default settings and light usage, the stack stays within [GCP free tier](#free-tier) (Cloud Run scale-to-zero, Firestore, Secret Manager, Artifact Registry). Set billing alerts to be notified if usage grows.

**Q: Can I use a different GCP project?**
A: Yes, change `project_id` in `terraform/variables.tf`.

**Q: Can I deploy to a different region?**
A: Yes, change `region` in `terraform/variables.tf`. Note: Firestore has limited region availability.

**Q: How do I add environment-specific configuration?**
A: Create separate `terraform.tfvars` files and use `-var-file=` flag in Terraform commands.

**Q: What if the API key expires?**
A: Rotate it:

1. Create new key in Google AI Studio
2. Add new version to Secret Manager
3. Delete old key
4. Redeploy to test the new key

## Support

For issues:

1. Check the troubleshooting section above
2. View logs: `gcloud run services logs tail novelsync-agents`
3. Check GitHub Actions: https://github.com/khaled2049/novelsync-agents/actions
4. Review plan: `terraform plan` to see what will change
