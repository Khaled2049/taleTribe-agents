variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "story-6f89f"
}

variable "region" {
  description = "GCP region for Cloud Run and Firestore"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "novelsync-agents"
}

variable "agent_service_url" {
  description = "Public URL of THIS Cloud Run service, used as the OIDC token audience. Set from the service's status.url (gcloud run services describe <service> --region <region> --format='value(status.url)'). Stable for the life of the service. Must match the AGENT_SERVICE_URL Firebase Functions uses to mint identity tokens."
  type        = string
  default     = "https://novelsync-agents-ukvrbnaddq-uc.a.run.app"
}

variable "image" {
  description = "Docker image URI for Cloud Run deployment"
  type        = string
  validation {
    condition     = length(var.image) > 0
    error_message = "Image URI cannot be empty"
  }
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances (0 for free tier - scale to zero)"
  type        = number
  default     = 0
  validation {
    condition     = var.min_instances >= 0
    error_message = "Min instances must be >= 0"
  }
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances (prevent runaway costs)"
  type        = number
  default     = 5
  validation {
    condition     = var.max_instances > 0
    error_message = "Max instances must be > 0"
  }
}

variable "memory" {
  description = "Memory allocation for Cloud Run (within free tier: 1Gi)"
  type        = string
  default     = "1Gi"
  validation {
    condition     = contains(["256Mi", "512Mi", "1Gi", "2Gi", "4Gi", "8Gi", "16Gi"], var.memory)
    error_message = "Memory must be valid Cloud Run value"
  }
}

variable "cpu" {
  description = "CPU allocation for Cloud Run (within free tier: 1)"
  type        = string
  default     = "1"
  validation {
    condition     = contains(["1", "2", "4", "6", "8"], var.cpu)
    error_message = "CPU must be valid Cloud Run value"
  }
}

variable "timeout_seconds" {
  description = "Request timeout in seconds (max 3600)"
  type        = number
  default     = 300
  validation {
    condition     = var.timeout_seconds > 0 && var.timeout_seconds <= 3600
    error_message = "Timeout must be between 1 and 3600 seconds"
  }
}

variable "concurrency" {
  description = "Maximum concurrent requests per instance"
  type        = number
  default     = 80
  validation {
    condition     = var.concurrency > 0 && var.concurrency <= 1000
    error_message = "Concurrency must be between 1 and 1000"
  }
}

variable "google_ai_studio_model" {
  description = "Google AI Studio model name (Free tier stable)"
  type        = string
  default     = "gemini-2.5-flash"
}

variable "enable_public_access" {
  description = "Enable public (unauthenticated) access to the service. Required for the MCP server: end-user MCP clients (Claude, etc.) reach /mcp and the OAuth endpoints directly, authenticated at the application layer by OAuth bearer tokens. /agent/execute stays protected by its own OIDC + service-account allowlist check, which is armed because this module pins ENVIRONMENT=production — never deploy with public access AND a non-production ENVIRONMENT, or /agent/execute would be unguarded."
  type        = bool
  default     = true
}

variable "firebase_functions_service_account" {
  description = "Service account email used by Firebase Functions to invoke the agent (e.g. story-6f89f@appspot.gserviceaccount.com). Granted roles/run.invoker on the Cloud Run service."
  type        = string
  default     = ""
}

variable "credit_proxy_url" {
  description = "Internal URL of the creditProxy gateway Cloud Run service (INGRESS_INTERNAL_ONLY). All LLM calls route through here."
  type        = string
  default     = "https://credit-proxy-gateway-ukvrbnaddq-uc.a.run.app"
}

variable "enable_mcp" {
  description = "Serve the MCP server (OAuth 2.1 AS + read-only story tools) from this service."
  type        = bool
  default     = true
}

variable "mcp_consent_url" {
  description = "Frontend consent page the MCP OAuth /authorize flow redirects users to. Required when enable_mcp is true (config.py fails fast in production without it)."
  type        = string
  default     = "https://thetaletribe.web.app/mcp-connect"
}

variable "cors_origins" {
  description = "Browser origins allowed by the agents service. Needed by the MCP consent page (GET /oauth/txn, POST /oauth/complete). JSON-encoded into CORS_ORIGINS."
  type        = list(string)
  default     = ["https://thetaletribe.web.app", "https://thetaletribe.com", "https://www.thetaletribe.com"]
}

variable "mcp_max_requests_per_minute_per_user" {
  description = "Per-user rate limit on MCP tool calls. Same per-instance caveat as max_requests_per_minute_per_user."
  type        = number
  default     = 60
}

variable "max_requests_per_minute_per_user" {
  description = "Per-user rate limit on POST /agent/execute. Note: in-memory per-instance bucket, so effective global ceiling is max_instances * this value. 0 disables the limiter."
  type        = number
  default     = 20
  validation {
    condition     = var.max_requests_per_minute_per_user >= 0
    error_message = "Rate limit must be >= 0"
  }
}
