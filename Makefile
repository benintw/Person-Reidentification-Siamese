# Person Re-Identification with Siamese Network
# Makefile for automating dataset download, training, validation, and testing

# Python environment management
PYTHON = python3
VENV_NAME = .venv
UV = uv

# Default config file paths
TRAIN_CONFIG = configs/training.yaml
VAL_CONFIG = configs/validation.yaml
PRED_CONFIG = configs/inference.yaml

# Dataset paths and URLs
DATA_DIR = data
MARKET1501_DIR = $(DATA_DIR)/market1501
MARKET1501_URL = https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view
MARKET1501_ZIP = $(DATA_DIR)/market1501.zip

# Output directories
CHECKPOINTS_DIR = checkpoints
RESULTS_DIR = results
LOGS_DIR = logs

# Create necessary directories
dirs:
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(CHECKPOINTS_DIR)
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(LOGS_DIR)

# Setup Python virtual environment
setup-env:
	$(UV) venv create -p $(PYTHON) $(VENV_NAME)
	$(UV) pip install -r requirements.txt
	@echo "Environment setup complete. Activate with 'source $(VENV_NAME)/bin/activate'"

# Download Market-1501 dataset
download-market1501: dirs
	@echo "Downloading Market-1501 dataset..."
	@echo "NOTE: Due to Google Drive restrictions, automatic download may not work."
	@echo "Please manually download from: $(MARKET1501_URL)"
	@echo "and place the zip file in $(DATA_DIR)/market1501.zip"
	@echo "Then run 'make extract-market1501'"

# Extract Market-1501 dataset
extract-market1501: dirs
	@if [ -f $(MARKET1501_ZIP) ]; then \
		echo "Extracting Market-1501 dataset..."; \
		unzip -q $(MARKET1501_ZIP) -d $(DATA_DIR); \
		echo "Dataset extracted to $(MARKET1501_DIR)"; \
	else \
		echo "Market-1501 zip file not found at $(MARKET1501_ZIP)"; \
		echo "Please download it first using 'make download-market1501'"; \
	fi

# Prepare dataset (create train/val/test splits if needed)
prepare-data: extract-market1501
	$(UV) run $(PYTHON) scripts/prepare_data.py --data_dir $(MARKET1501_DIR)

# Training command
train:
	@./scripts/train.sh	

# Validation command
validate:
	@./scripts/validate.sh

# Prediction/testing command
predict:
	@./scripts/predict.sh

# Analyze predictions
analyze: dirs
	@$(UV) run $(PYTHON) scripts/analyze_predictions.py

# Explore predictions (without requiring image files)
explore: dirs
	@$(UV) run $(PYTHON) scripts/explore_predictions.py

# Run all steps in sequence
all: train validate predict analyze

# Clean generated files
clean:
	@rm -rf $(RESULTS_DIR)/*
	@rm -rf __pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete

# Deep clean (remove all generated files including datasets)
deep-clean: clean
	@rm -rf $(CHECKPOINTS_DIR)/*
	@rm -rf $(LOGS_DIR)/*

# Help command
help:
	@echo "Person Re-Identification with Siamese Network"
	@echo "Available commands:"
	@echo "  make setup-env          - Set up Python virtual environment"
	@echo "  make download-market1501 - Download Market-1501 dataset"
	@echo "  make extract-market1501  - Extract Market-1501 dataset"
	@echo "  make prepare-data       - Prepare dataset for training"
	@echo "  make train              - Train the model"
	@echo "  make validate           - Validate the model"
	@echo "  make predict            - Test the model and generate predictions"
	@echo "  make analyze            - Analyze prediction results"
	@echo "  make explore            - Explore predictions without image files"
	@echo "  make all                - Run train, validate, predict, and analyze"
	@echo "  make clean              - Remove generated files"
	@echo "  make deep-clean         - Remove all generated files including checkpoints"

.PHONY: dirs setup-env download-market1501 extract-market1501 prepare-data train validate predict analyze explore all clean deep-clean help
