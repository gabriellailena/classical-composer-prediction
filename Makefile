.PHONY: setup-venv remove-venv setup download-data run stop run-servers stop-servers lint

VENV=.venv
ifneq (,$(wildcard .env))
	include .env
	export
endif
MLFLOW_BACKEND_URI ?= sqlite:///mlflow.db
MAGE_PROJECT_NAME ?= train-classical-composer-prediction

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

# Runs all servers
run-servers:
	@echo "Starting all servers..."
	. $(VENV)/bin/activate && \
	mlflow ui --backend-store-uri $(MLFLOW_BACKEND_URI) --default-artifact-root $(PWD)/artifacts & \
	mage start $(MAGE_PROJECT_NAME) & \
	python app.py

# Stops all servers
stop-servers:
	@echo "Stopping all servers..."
	@kill $(shell lsof -t -i:6789) 2>/dev/null || true  # Mage AI default port
	@kill $(shell lsof -t -i:5000) 2>/dev/null || true  # MLFlow default port
	@kill $(shell lsof -t -i:8000) 2>/dev/null || true  # API default port

# Deploys the Flask App with gunicorn
deploy:
	@echo "Deploying Flask App..."
	. $(VENV)/bin/activate && gunicorn --workers=2 --bind=0.0.0.0:8000 --reload app:app

# Runs Ruff linter to check code style and formatting
lint:
	@echo "Running Ruff linter..."
	@. $(VENV)/bin/activate && ruff check --fix .
	@. $(VENV)/bin/activate && ruff format .

