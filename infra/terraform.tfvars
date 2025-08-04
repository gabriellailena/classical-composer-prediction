services = [
  {
    name           = "api"
    image_version  = "v1.0.0"
    container_port = 8000
    memory_limit   = "2Gi"
    cpu_limit      = "1000m"
  },
  {
    name           = "mlflow"
    image_version  = "v1.0.0"
    container_port = 5000
    memory_limit   = "2Gi"
    cpu_limit      = "2000m"
  },
  {
    name           = "mage"
    image_version  = "v1.0.0"
    container_port = 6789
    memory_limit   = "4Gi"
    cpu_limit      = "2000m"
  }
]
