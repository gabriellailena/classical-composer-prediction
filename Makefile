.PHONY: setup-venv remove-venv setup download-data run stop run-servers stop-servers lint

VENV=.venv
ifneq (,$(wildcard .env))
	include .env
	export
endif
MLFLOW_BACKEND_URI ?= sqlite:///mlflow.db
MAGE_PROJECT_NAME ?= train-classical-composer-prediction
USERNAME ?= user
SERVICE ?= api
VERSION ?= v0.0.1
GCP_PROJECT_ID ?= gcp-project-id
GCP_REGION ?= europe-central2
TERRAFORM_DIR ?= infra

setup-venv:
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

remove-venv:
	rm -rf $(VENV)
	@echo "Virtual environment removed."

# Fresh setup which removes the existing virtual environment and sets it up again
setup: remove-venv setup-venv
	@echo "Setting up fresh environment..."

# Downloads the required MusicNet dataset
download-data:
	@echo "Downloading MusicNet data..."
	./scripts/download_data.sh

# Start Docker containers for all services
# This includes Mage AI, MLFlow, and the API service
up:
	@echo "Starting Docker containers for all services..."
	docker compose up

# Build Docker containers for all services
build:
	@echo "Building Docker containers for all services..."
	docker compose build

# Stops Docker containers for all services
down:
	@echo "Stopping Docker containers for all services..."
	docker compose down

# Deploys the Flask App with gunicorn
deploy:
	@echo "Deploying Flask App..."
	. $(VENV)/bin/activate && gunicorn --workers=2 --bind=0.0.0.0:8000 --reload app:app

# Runs Ruff linter to check code style and formatting
lint:
	@echo "Running Ruff linter..."
	@. $(VENV)/bin/activate && ruff check --fix .
	@. $(VENV)/bin/activate && ruff format .

# Setup Google Cloud authentication for Docker
# NOTE: Run this once before using push-gcp commands
setup-gcp-auth:
	@echo "Setting up Google Cloud authentication..."
	gcloud auth login --no-launch-browser
	gcloud config set project $(GCP_PROJECT_ID)
	gcloud auth configure-docker
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev

	# Authenticate application default credentials for Terraform
	@echo "Setting up application default credentials..."
	gcloud auth application-default login --no-launch-browser

# Build and push all services to the image destination defined in docker-compose.yml
# NOTE: Need to run setup-gcp-auth first
push-all:
	@echo "Building and pushing all services..."
	docker compose build
	docker compose push
	@echo "All services pushed successfully!"

# Push docker image to Google Cloud
# Usage: make push-gcp SERVICE=api USERNAME=myusername VERSION=v0.0.1 GCP_PROJECT_ID=my-project
# NOTE: Need to run setup-gcp-auth first and ensure image exists in GitHub Container Registry
push-gcp:
	@echo "Pushing to Google Cloud..."
	@echo "Service: $(SERVICE), Username: $(USERNAME), Version: $(VERSION), Project: $(GCP_PROJECT_ID)"
	
	# Tag for Google Cloud Artifact Registry
	docker tag ghcr.io/$(USERNAME)/$(SERVICE):latest $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/docker-repo/$(SERVICE):latest
	docker tag ghcr.io/$(USERNAME)/$(SERVICE):$(VERSION) $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/docker-repo/$(SERVICE):$(VERSION)
	
	# Push to Google Cloud
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/docker-repo/$(SERVICE):latest
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/docker-repo/$(SERVICE):$(VERSION)
	
	@echo "Successfully pushed $(SERVICE) to Google Cloud!"

# Runs tests in the specified environment
test:
	@echo "Running tests..."
	. $(VENV)/bin/activate && pytest -v

# Run Terraform initialization
terraform-init:
	cd $(TERRAFORM_DIR) && terraform init

terraform-plan:
	cd $(TERRAFORM_DIR) && terraform plan -var="project_id=$(GCP_PROJECT_ID)" -var="region=$(GCP_REGION)"

terraform-apply:
	cd $(TERRAFORM_DIR) && terraform apply -var="project_id=$(GCP_PROJECT_ID)" -var="region=$(GCP_REGION)" -auto-approve

# Force replace Cloud Run service to create new revision
terraform-replace:
	cd $(TERRAFORM_DIR) && terraform apply -replace="google_cloud_run_service.services" -var="project_id=$(GCP_PROJECT_ID)" -var="region=$(GCP_REGION)" -auto-approve

# Import existing Cloud Run service into Terraform state
# Usage: make terraform-import SERVICE_NAME=your-service-name
terraform-import:
	@if [ -z "$(SERVICE_NAME)" ]; then \
		echo "Error: SERVICE_NAME is required. Usage: make terraform-import SERVICE_NAME=your-service-name"; \
		exit 1; \
	fi
	cd $(TERRAFORM_DIR) && terraform import google_cloud_run_service.services projects/$(GCP_PROJECT_ID)/locations/$(GCP_REGION)/services/$(SERVICE_NAME)

# Refresh Terraform state to match current infrastructure
terraform-refresh:
	cd $(TERRAFORM_DIR) && terraform refresh -var="project_id=$(GCP_PROJECT_ID)" -var="region=$(GCP_REGION)"

# Show current Terraform state
terraform-show:
	cd $(TERRAFORM_DIR) && terraform show

# List all resources in Terraform state
terraform-state-list:
	cd $(TERRAFORM_DIR) && terraform state list

# Check if service exists and suggest import if needed
terraform-check-existing:
	@echo "Checking for existing Cloud Run services..."
	@gcloud run services list --region=$(GCP_REGION) --project=$(GCP_PROJECT_ID) --format="table(metadata.name,status.url)" || echo "No services found or authentication needed"

# Deploy with import check
deploy: test push-all terraform-init terraform-check-existing terraform-refresh terraform-plan terraform-apply

# Deploy with force replace (creates new revision)
deploy-replace: test push-all terraform-init terraform-plan terraform-replace
