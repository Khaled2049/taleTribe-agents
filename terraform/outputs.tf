output "service_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.app.uri
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.app.name
}

output "service_region" {
  description = "Region of the Cloud Run service"
  value       = google_cloud_run_v2_service.app.location
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository URL for Docker images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${data.google_artifact_registry_repository.docker_repo.repository_id}"
}

output "service_account_email" {
  description = "Email of the Cloud Run service account"
  value       = data.google_service_account.cloud_run_sa.email
}

output "firestore_database_id" {
  description = "Firestore database ID"
  value       = google_firestore_database.database.name
}

output "secret_manager_secret_id" {
  description = "Secret Manager secret ID for Google AI Studio API key"
  value       = data.google_secret_manager_secret.google_ai_studio_api_key.secret_id
}

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}
