# Enable required Google Cloud APIs
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "firestore" {
  service            = "firestore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}


data "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "novelsync-agents"
  project       = var.project_id
}

# Firestore database is created by the deploy workflow if missing (no data source in provider).
# Terraform does not manage it to avoid 409 in CI when DB already exists and state is not shared.

# Use existing Secret Manager secret (created outside Terraform or by GitHub Actions)
# Secret versions are added by the deploy workflow; Terraform only references it.
data "google_secret_manager_secret" "google_ai_studio_api_key" {
  secret_id = "google-ai-studio-api-key"
  project   = var.project_id
}

data "google_service_account" "cloud_run_sa" {
  account_id = "novelsync-agents-run"
  project    = var.project_id
}

# IAM: Allow Cloud Run service account to read secrets from Secret Manager
resource "google_secret_manager_secret_iam_member" "secret_access" {
  secret_id = data.google_secret_manager_secret.google_ai_studio_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_service_account.cloud_run_sa.email}"
}

# IAM: Allow Cloud Run service account to use Firestore
resource "google_project_iam_member" "firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${data.google_service_account.cloud_run_sa.email}"
}

# Cloud Run Service
# Free tier: 2M requests/month, 360K GB-seconds, 180K vCPU-seconds
# Configured to scale to zero when idle (min_instances=0)
resource "google_cloud_run_v2_service" "app" {
  name     = var.service_name
  location = var.region

  template {
    service_account = data.google_service_account.cloud_run_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      # Resource limits (within free tier)
      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        cpu_idle          = true  # Only charge for CPU during request execution
        startup_cpu_boost = false # Free tier compatible
      }

      # Environment variables (non-secret configuration)
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_AI_STUDIO_MODEL"
        value = var.google_ai_studio_model
      }

      env {
        name  = "ENVIRONMENT"
        value = "production"
      }

      env {
        name  = "ENABLE_LOCAL_IMAGE_GENERATION"
        value = "false"
      }

      # Secret from Secret Manager (accessed via service account)
      env {
        name = "GOOGLE_AI_STUDIO_API_KEY"
        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.google_ai_studio_api_key.secret_id
            version = "latest"
          }
        }
      }

      # Startup probe - waits for service to be ready
      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        timeout_seconds       = 3
        period_seconds        = 10
        failure_threshold     = 3
      }

      # Liveness probe - restarts container if unhealthy
      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds       = 3
        period_seconds        = 30
        failure_threshold     = 3
      }
    }

    # Request timeout
    timeout = "${var.timeout_seconds}s"

    # Max concurrent requests per instance
    max_instance_request_concurrency = var.concurrency
  }

  # Route all traffic to the latest revision
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.run,
    google_secret_manager_secret_iam_member.secret_access,
    google_project_iam_member.firestore_user,
  ]
}

# IAM: Allow public access (unauthenticated invocations)
# Comment out if you want authenticated-only access
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count    = var.enable_public_access ? 1 : 0
  name     = google_cloud_run_v2_service.app.name
  location = google_cloud_run_v2_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
