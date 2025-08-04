variable "project_id" {}
variable "region" {}
variable "image_version" {
  default = "latest"
}
variable "services" {
  type = list(object({
    name          = string
    image_version = string
    container_port = number
    memory_limit   = optional(string, "2Gi")
    cpu_limit      = optional(string, "1000m")
  }))
}

