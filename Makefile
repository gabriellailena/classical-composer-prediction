.PHONY: setup-venv remove-venv
VENV=.venv

setup-venv:
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

remove-venv:
	rm -rf $(VENV)
	@echo "Virtual environment removed."

# Downloads the required MusicNet dataset
download-data:
	@echo "Downloading MusicNet data..."
	./scripts/download_data.sh