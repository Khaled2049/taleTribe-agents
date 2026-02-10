# Terraform State Backend Configuration
#
# Currently using local state (default).
#
# To migrate to GCS backend for team collaboration:
# 1. Create GCS bucket: gsutil mb gs://story-6f89f-terraform-state
# 2. Uncomment the backend block below
# 3. Run: terraform init
#
# For production, enable versioning and encryption on the bucket:
#   gsutil versioning set on gs://story-6f89f-terraform-state
#   gsutil encryption set gs://story-6f89f-terraform-state

# Uncomment the block below to use GCS backend:
#
# terraform {
#   backend "gcs" {
#     bucket  = "story-6f89f-terraform-state"
#     prefix  = "terraform/state"
#   }
# }
#
# With GCS backend, you also need to initialize with credentials:
# terraform init -backend-config=bucket=story-6f89f-terraform-state
