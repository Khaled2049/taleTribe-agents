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

data "google_secret_manager_secret" "story_data_service_token" {
  secret_id = "story-data-service-token"
}

# IAM: Allow Cloud Run service account to read secrets from Secret Manager
resource "google_secret_manager_secret_iam_member" "secret_access" {
  secret_id = data.google_secret_manager_secret.google_ai_studio_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_service_account.cloud_run_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "story_data_service_token_access" {
  secret_id = data.google_secret_manager_secret.story_data_service_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_service_account.cloud_run_sa.email}"
}

# IAM: Allow Cloud Run service account to use Firestore
resource "google_project_iam_member" "firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${data.google_service_account.cloud_run_sa.email}"
}

locals {
  # ENVIRONMENT is what arms /agent/execute's OIDC caller check (see
  # config.py). Public invoker access is only safe while this is "production",
  # so it is a named local rather than an inline string: the precondition on the
  # service below asserts the coupling, and both read the same value.
  agent_environment = "production"
}

# Cloud Run Service
# Free tier: 2M requests/month, 360K GB-seconds, 180K vCPU-seconds
# Configured to scale to zero when idle (min_instances=0)
resource "google_cloud_run_v2_service" "app" {
  name     = var.service_name
  location = var.region

  lifecycle {
    # The MCP server needs public invoker access, which puts /agent/execute,
    # /credits/balance and /credits/purchase on the open internet behind nothing
    # but their own OIDC + service-account check — and config.py only arms that
    # check when ENVIRONMENT=production. Loosening the environment while public
    # access is on would silently unguard all three, with no failure anywhere.
    # Fail the plan instead of relying on the warning in variables.tf.
    precondition {
      condition     = !var.enable_public_access || local.agent_environment == "production"
      error_message = <<-EOT
        enable_public_access=true requires local.agent_environment="production".
        Public invoker access exposes /agent/execute, /credits/balance and
        /credits/purchase to the internet; their only guard is the OIDC caller
        allowlist, which config.py enforces solely when ENVIRONMENT=production.
        Either keep the environment at "production" or set
        enable_public_access=false (which disables the MCP server's OAuth flow,
        since MCP clients must reach it unauthenticated).
      EOT
    }
  }

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
        cpu_idle          = true # Only charge for CPU during request execution
        startup_cpu_boost = true
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
        value = local.agent_environment
      }

      env {
        name  = "ENABLE_LOCAL_IMAGE_GENERATION"
        value = "false"
      }

      # Browser origins: the MCP consent page (frontend) calls /oauth/txn and
      # /oauth/complete cross-origin. /agent/execute remains server-to-server.
      env {
        name  = "CORS_ORIGINS"
        value = jsonencode(var.cors_origins)
      }

      env {
        name  = "ENABLE_MCP"
        value = tostring(var.enable_mcp)
      }

      env {
        name  = "MCP_CONSENT_URL"
        value = var.mcp_consent_url
      }

      # story-data's HTTP API. The MCP read tools serve story content from it,
      # so ENABLE_MCP without this fails startup rather than reading nothing.
      env {
        name  = "STORY_DATA_URL"
        value = var.story_data_url
      }

      # OAuth issuer == this service's public URL (also the MCP resource base).
      env {
        name  = "MCP_ISSUER_URL"
        value = var.agent_service_url
      }

      env {
        name  = "MCP_MAX_REQUESTS_PER_MINUTE_PER_USER"
        value = tostring(var.mcp_max_requests_per_minute_per_user)
      }

      env {
        name  = "ENABLE_MCP_WRITES"
        value = tostring(var.enable_mcp_writes)
      }

      env {
        name  = "MCP_MAX_WRITES_PER_MINUTE_PER_USER"
        value = tostring(var.mcp_max_writes_per_minute_per_user)
      }

      env {
        name  = "ENABLE_MCP_ACCESS_ALLOWLIST"
        value = tostring(var.enable_mcp_access_allowlist)
      }

      env {
        name  = "MCP_ACCESS_CACHE_TTL_SECONDS"
        value = tostring(var.mcp_access_cache_ttl_seconds)
      }

      env {
        name  = "FIREBASE_FUNCTIONS_SERVICE_ACCOUNT"
        value = var.firebase_functions_service_account
      }

      # Must match the audience Firebase Functions uses when minting identity tokens.
      env {
        name  = "AGENT_SERVICE_URL"
        value = var.agent_service_url
      }

      env {
        name  = "CREDIT_PROXY_URL"
        value = var.credit_proxy_url
      }

      env {
        name  = "MAX_REQUESTS_PER_MINUTE_PER_USER"
        value = tostring(var.max_requests_per_minute_per_user)
      }

      # Lets MCP assert X-User-ID to story-data. Must match SERVICE_TOKEN there.
      env {
        name = "STORY_DATA_SERVICE_TOKEN"
        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.story_data_service_token.secret_id
            version = "latest"
          }
        }
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
        timeout_seconds       = 5
        period_seconds        = 15
        failure_threshold     = 8
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

# Firestore TTL garbage collection for expired MCP artifacts.
# TTL deletion can lag 24-72h; the application re-checks expiresAt on every
# read, so TTL here is cleanup, not enforcement.
# mcpOauthClients is included because /register is unauthenticated: a client
# record starts with a short expiry that only slides forward once the client is
# actually used, so abandoned registrations get collected.
# mcpWrites holds 2-minute write-idempotency reservations. It is keyed on
# enable_mcp rather than enable_mcp_writes so that flipping writes on does not
# also require a TTL policy change — the collection is simply unused until then.
resource "google_firestore_field" "mcp_oauth_ttl" {
  for_each = var.enable_mcp ? toset(["mcpOauthTxns", "mcpOauthCodes", "mcpOauthTokens", "mcpOauthClients", "mcpWrites"]) : toset([])

  project    = var.project_id
  database   = "(default)"
  collection = each.key
  field      = "expiresAt"

  ttl_config {}

  depends_on = [google_project_service.firestore]
}

# IAM: Allow public access (unauthenticated invocations). Required for MCP:
# end-user MCP clients authenticate with OAuth bearer tokens at the app layer.
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count    = var.enable_public_access ? 1 : 0
  name     = google_cloud_run_v2_service.app.name
  location = google_cloud_run_v2_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# IAM: Allow Firebase Functions service account to invoke the agent
resource "google_cloud_run_v2_service_iam_member" "firebase_functions_invoker" {
  count    = var.firebase_functions_service_account != "" ? 1 : 0
  name     = google_cloud_run_v2_service.app.name
  location = google_cloud_run_v2_service.app.location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.firebase_functions_service_account}"
}
