# Deployment Guide: Terraform + GitHub Actions

This guide covers deploying NovelSync Agents to Google Cloud Run using Terraform for infrastructure and GitHub Actions for CI/CD.

**Quick Links:**

- [Architecture Overview](#architecture-overview)
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

## Prerequisites Setup

Run these commands **once** to prepare GCP for deployment. This sets up:

- Required APIs
- Firestore database
- Service accounts and IAM
- Workload Identity Federation (keyless authentication)
- Secrets

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

### 2. Create Firestore Database

```bash
# Create Firestore in Native mode (required for app to work)
gcloud firestore databases create \
  --location=us-central1 \
  --type=firestore-native

# Verify database created
gcloud firestore databases list
```

### 3. Create Service Account for GitHub Actions

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

### 4. Set Up Workload Identity Federation (Keyless Auth)

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

### 5. Add Missing IAM Roles to GitHub Actions Service Account

GitHub Actions needs additional permissions to manage resources:

```bash
# Service usage management (to enable/list APIs)
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/serviceusage.serviceUsageAdmin"

# IAM management (to create service accounts)
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/iam.securityAdmin"

# Firestore management (to manage database)
gcloud projects add-iam-policy-binding story-6f89f \
  --member="serviceAccount:github-actions@story-6f89f.iam.gserviceaccount.com" \
  --role="roles/datastore.owner"
```

### 6. Create Secret in Secret Manager

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
   - **IMPORTANT:** After first successful deployment, delete from `.env` and rotate the key

### How to Add Secrets in GitHub

1. Go to: https://github.com/khaled2049/novelsync-agents/settings/secrets/actions
2. Click "New repository secret"
3. Add each secret:
   - Name: `WIF_PROVIDER`
   - Value: (paste the full provider resource name)
4. Repeat for other secrets

## First Deployment

### Step 0: Optimize Docker Image (IMPORTANT)

The full `requirements.txt` includes large ML libraries (PyTorch, Diffusers) that are not needed for Cloud Run since image generation is disabled. Create a production-only requirements file:

**Create `requirements-prod.txt`:**
```bash
# This file is already created for you
# It excludes: torch, diffusers, Pillow, transformers, accelerate
cat requirements-prod.txt
```

**Update Dockerfile and CI/CD to use it:**
- ✅ `Dockerfile` - uses `requirements-prod.txt`
- ✅ `.github/workflows/ci.yml` - uses `requirements-prod.txt`
- ✅ Keep `requirements.txt` for local development with full features

This reduces Docker image size from ~8GB to ~500MB and speeds up CI/CD!

### Step 1: Create Artifact Registry Repository

```bash
# Create the Docker repository (Terraform will use this)
gcloud artifacts repositories create novelsync-agents \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repository for NovelSync Agents"
```

### Step 2: Initialize Terraform

```bash
# Navigate to project
cd /Users/kh1011/Documents/Developer/2026/novelsync-agents

# Initialize Terraform
cd terraform
terraform init

# Validate configuration
terraform validate

# Format check (for code quality)
terraform fmt -check -recursive
```

### Step 3: Build and Push Initial Docker Image

**Important: Use correct architecture for Cloud Run (x86_64)**

```bash
# Back to project root
cd ..

# Build Docker image for x86_64 (required for Cloud Run)
# Use --platform if on M1/M2 Mac (ARM64)
docker build --platform=linux/amd64 \
  -t us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:initial \
  .

# Authenticate Docker to Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push image
docker push us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:initial
```

### Step 4: Import Existing Resources into Terraform

If you created resources manually (Firestore database, Secret, Artifact Registry), tell Terraform about them:

```bash
cd terraform

# Import Artifact Registry repository
terraform import google_artifact_registry_repository.docker_repo \
  projects/story-6f89f/locations/us-central1/repositories/novelsync-agents

# Import Firestore database
terraform import google_firestore_database.database \
  'projects/story-6f89f/databases/(default)'

# Import Secret Manager secret (if it exists)
terraform import google_secret_manager_secret.google_ai_studio_api_key \
  google-ai-studio-api-key
```

### Step 5: Deploy with Terraform

```bash
# Plan deployment
terraform plan \
  -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:initial"

# Apply (creates/updates infrastructure)
terraform apply \
  -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:initial"

# Get service URL
terraform output service_url
# Output: https://novelsync-agents-XXXXX-uc.a.run.app
```

### Step 4: Verify Deployment

```bash
# Get the service URL
SERVICE_URL=$(terraform output -raw service_url)

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

### Step 5: Commit and Enable Automated Deployments

```bash
# Remove the exposed API key from .env
# CRITICAL SECURITY STEP
rm .env

# Add deployment files to git
git add .
git commit -m "feat: Add Terraform and GitHub Actions deployment

- Terraform configuration for Cloud Run, Firestore, Secret Manager
- GitHub Actions CI/CD workflows (lint, test, build, deploy)
- Test suite with pytest
- Deployment documentation
- Secure secrets management with Workload Identity Federation"

# Push to main - GitHub Actions will automatically deploy on subsequent changes
git push origin main

# Monitor deployment at:
# https://github.com/khaled2049/novelsync-agents/actions
```

### Step 6: Secure the API Key

**CRITICAL:** The API key is currently exposed in the `.env` file. Complete these steps immediately:

```bash
# 1. The .env file with the exposed key should already be deleted above
#    Verify it's gone:
ls -la .env  # Should NOT exist

# 2. Rotate the API key in Google AI Studio:
#    - Go to: https://aistudio.google.com/app/apikey
#    - Delete the old key (the one that was in .env)
#    - Create a new API key
#    - Copy the new key

# 3. Update GitHub Secrets with the new API key:
#    - Go to: https://github.com/khaled2049/novelsync-agents/settings/secrets/actions
#    - Update GOOGLE_AI_STUDIO_API_KEY with the new key
#    - GitHub Actions will use the new key for future deployments

# 4. Verify the new key is in Secret Manager:
gcloud secrets versions list google-ai-studio-api-key --limit=3
# Should show the new version is "latest"
```

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

### Updating Secrets

```bash
# Rotate Google AI Studio API key
# 1. Create new API key in Google AI Studio console
# 2. Add new secret version to Secret Manager
echo "NEW_API_KEY" | gcloud secrets versions add google-ai-studio-api-key --data-file=-

# 3. Update GitHub Secret with same key
#    Go to: https://github.com/khaled2049/novelsync-agents/settings/secrets/actions

# 4. Cloud Run automatically uses latest version
# 5. Delete old API key from Google AI Studio console

# Verify the change
gcloud secrets versions list google-ai-studio-api-key --limit=3
```

### Monitoring Costs

The app is configured to stay within GCP's free tier.

```bash
# Check billing account
gcloud billing projects describe story-6f89f

# View resource usage (Cloud Console)
# https://console.cloud.google.com/billing

# Free tier limits we're using:
# - Cloud Run: 2M requests/month (we use <100K)
# - Cloud Run: 360K GB-seconds (we use <20K)
# - Firestore: 1GB storage (we use <100MB)
# - Firestore: 50K reads/day (we use <5K)
```

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

### Terraform: Resources Already Exist

**Problem:** `Error 409: the repository already exists` or `Error 409: Secret already exists`

**Root Cause:** Resources were created manually before Terraform knew about them.

**Solution:** Import existing resources into Terraform state:
```bash
cd terraform

# Import Artifact Registry
terraform import google_artifact_registry_repository.docker_repo \
  projects/story-6f89f/locations/us-central1/repositories/novelsync-agents

# Import Secret
terraform import google_secret_manager_secret.google_ai_studio_api_key \
  google-ai-studio-api-key

# Import Firestore database
terraform import google_firestore_database.database \
  'projects/story-6f89f/databases/(default)'
```

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

**Problem:** "Resource already exists" error

```bash
# Solution: Some resources are already created
# This is safe - Terraform will adopt them on next run
terraform import google_firestore_database.database projects/story-6f89f/databases/(default)
```

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

```bash
# Solution: Terraform can't create what already exists
# Just skip creating the database resource or import it
terraform import google_firestore_database.database projects/story-6f89f/databases/(default)
```

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
A: $0/month with light usage (within GCP free tier). See [terraform/README.md](./terraform/README.md#free-tier-limits).

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
