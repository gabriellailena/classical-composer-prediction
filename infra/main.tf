provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_service" "services" {
  for_each = {
    for svc in var.services : svc.name => svc
  }

  name     = each.key
  location = var.region

  template {
    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/docker-repo/${each.key}:${each.value.image_version}"
        ports {
          container_port = each.value.container_port
        }
        resources {
          limits = {
            memory = each.value.memory_limit
            cpu    = each.value.cpu_limit
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "public" {
  for_each = google_cloud_run_service.services

  service  = each.value.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
