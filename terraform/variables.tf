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
  description = "Enable public (unauthenticated) access to the service"
  type        = bool
  default     = true
}
