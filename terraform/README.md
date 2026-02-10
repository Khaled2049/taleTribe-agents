# Terraform Configuration for NovelSync Agents

This directory contains the Terraform infrastructure-as-code for deploying NovelSync Agents to Google Cloud Run.

## Overview

The configuration provisions:

- **Cloud Run Service** - FastAPI backend server with auto-scaling (0-5 instances)
- **Firestore Database** - Document database for story data
- **Secret Manager** - Secure storage for API keys
- **Artifact Registry** - Docker image repository
- **IAM Service Accounts** - Minimal privilege access control

All resources are configured to stay within GCP's free tier limits.

## Prerequisites

Before running Terraform, you must:

1. **Enable Required GCP APIs** (one-time setup):

   ```bash
   gcloud config set project story-6f89f

   gcloud services enable \
     run.googleapis.com \
     firestore.googleapis.com \
     secretmanager.googleapis.com \
     artifactregistry.googleapis.com \
     iam.googleapis.com
   ```

2. **Create Firestore Database** (if not exists):

   ```bash
   gcloud firestore databases create \
     --location=us-central1 \
     --type=firestore-native
   ```

3. **Create Secret** for Google AI Studio API key:
   ```bash
   # Get your API key from https://aistudio.google.com/app/apikey
   echo "YOUR_API_KEY_HERE" | \
     gcloud secrets create google-ai-studio-api-key \
     --data-file=- \
     --replication-policy=automatic
   ```

## Configuration

### Quick Start

1. **Initialize Terraform**:

   ```bash
   terraform init
   ```

2. **Set variables**:

   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your values, especially the Docker image URI
   ```

3. **Plan deployment**:

   ```bash
   terraform plan
   ```

4. **Apply configuration**:
   ```bash
   terraform apply
   ```

### Terraform Variables

See `variables.tf` for all configuration options. Key variables:

| Variable                 | Default            | Description                                 |
| ------------------------ | ------------------ | ------------------------------------------- |
| `project_id`             | `story-6f89f`      | GCP Project ID                              |
| `region`                 | `us-central1`      | GCP Region                                  |
| `service_name`           | `novelsync-agents` | Cloud Run service name                      |
| `image`                  | _(required)_       | Docker image URI from Artifact Registry     |
| `min_instances`          | `0`                | Min Cloud Run instances (0 = scale to zero) |
| `max_instances`          | `5`                | Max Cloud Run instances                     |
| `memory`                 | `1Gi`              | Memory allocation (free tier: 1Gi)          |
| `cpu`                    | `1`                | CPU allocation (free tier: 1)               |
| `timeout_seconds`        | `300`              | Request timeout (5 minutes)                 |
| `concurrency`            | `80`               | Max concurrent requests per instance        |
| `google_ai_studio_model` | `gemini-2.5-flash` | AI model to use                             |
| `enable_public_access`   | `true`             | Allow unauthenticated access                |

### Passing Variables

#### Method 1: terraform.tfvars file (recommended)

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform apply
```

#### Method 2: Command-line arguments

```bash
terraform apply \
  -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:sha123"
```

#### Method 3: Environment variables

```bash
export TF_VAR_image="us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:latest"
terraform apply
```

## Outputs

After applying Terraform, get the deployed service URL:

```bash
# Get service URL
terraform output service_url

# Get artifact registry repository URL
terraform output artifact_registry_repository

# Get service account email
terraform output service_account_email
```

## State Management

### Local State (Default)

By default, Terraform stores state locally in `terraform.tfstate`. This file:

- Contains sensitive data (secrets, resource IDs)
- Should be in `.gitignore` (it is)
- Should be backed up securely

### GCS Backend (Optional)

For team collaboration, use Google Cloud Storage as the state backend:

1. **Create GCS bucket**:

   ```bash
   gsutil mb gs://story-6f89f-terraform-state
   gsutil versioning set on gs://story-6f89f-terraform-state
   ```

2. **Uncomment backend block in `backend.tf`**

3. **Reinitialize Terraform**:
   ```bash
   terraform init
   # Choose "yes" to migrate state to GCS
   ```

## Deployment Workflow

### Manual Deployment

```bash
# 1. Build and push Docker image
docker build -t us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:v1.0 .
docker push us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:v1.0

# 2. Update Terraform
terraform plan -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:v1.0"
terraform apply -var="image=us-central1-docker.pkg.dev/story-6f89f/novelsync-agents/app:v1.0"

# 3. Get service URL
terraform output service_url

# 4. Test health endpoint
curl $(terraform output -raw service_url)/health
```

### Automated Deployment (via GitHub Actions)

GitHub Actions automatically:

1. Builds and pushes Docker image
2. Runs `terraform apply` with the new image
3. Tests the deployment
4. Reports the service URL

## Cost Monitoring

Free tier limits (stay within these):

| Service        | Free Tier Limit       | Our Config                   |
| -------------- | --------------------- | ---------------------------- |
| Cloud Run      | 2M requests/month     | ~100K/month (est.)           |
| Cloud Run      | 360K GB-seconds/month | ~20K GB-seconds/month (est.) |
| Firestore      | 1GB storage           | <100MB                       |
| Firestore      | 50K reads/day         | <5K reads/day (est.)         |
| Secret Manager | 6 active versions     | 1 secret                     |

**Expected monthly cost: $0** (within free tier)

Monitor billing:

```bash
gcloud billing projects describe story-6f89f
```

## Updating Secrets

To rotate the Google AI Studio API key:

```bash
# Create new API key in Google AI Studio console
# https://aistudio.google.com/app/apikey

# Add new secret version to Secret Manager
echo "NEW_API_KEY" | gcloud secrets versions add google-ai-studio-api-key --data-file=-

# Cloud Run automatically uses latest version
# Verify by checking the deployment
gcloud run services describe novelsync-agents --region=us-central1
```

## Troubleshooting

### Terraform init fails

```bash
# Ensure APIs are enabled
gcloud services enable run.googleapis.com firestore.googleapis.com secretmanager.googleapis.com
```

### Secret not found error

```bash
# Create secret if missing
echo "YOUR_API_KEY" | gcloud secrets create google-ai-studio-api-key --data-file=-
```

### Cloud Run deployment fails

```bash
# Check service logs
gcloud run services logs read novelsync-agents --region=us-central1

# Common causes:
# - Image not found (check Artifact Registry)
# - Secret not accessible (check IAM permissions)
# - Firestore database not created
```

### State file issues

```bash
# Don't edit state files manually!
# For state problems, see Terraform docs:
# https://www.terraform.io/docs/commands/state/index.html

# To see current state
terraform state list
terraform state show google_cloud_run_v2_service.app
```

## Cleanup

To destroy all resources (WARNING: deletes production!):

```bash
# See what will be deleted
terraform plan -destroy

# Delete all resources
terraform destroy

# Verify deletion
gcloud run services list --region=us-central1
```

Note: This does NOT delete:

- Firestore database (must be deleted manually)
- Secret Manager secrets (must be deleted manually)
- Artifact Registry images (must be deleted manually)

Delete these manually if needed:

```bash
# Delete Firestore database
gcloud firestore databases delete --database='(default)'

# Delete secret
gcloud secrets delete google-ai-studio-api-key

# Delete Artifact Registry repository
gcloud artifacts repositories delete novelsync-agents --location=us-central1
```

## References

- [Terraform Google Provider Documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Firestore Documentation](https://cloud.google.com/firestore/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
