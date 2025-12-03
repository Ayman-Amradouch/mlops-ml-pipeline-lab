
# ======================================================================
# Makefile for MLOps Training Project 
# ======================================================================

# -------------------- VARIABLES --------------------
PYTHON          ?= python

INPUT_DATA_PATH ?= datastores/raw_data/housing.csv
OUTPUT_FILENAME ?= clean_housing.csv

INPUT_TRAIN_DATA ?= datastores/splits_data/train_data.csv
INPUT_TEST_DATA  ?= datastores/splits_data/test_data.csv
MODEL_FILENAME   ?= LinearRegression.joblib   # juste le nom du fichier

CONDA_ENV ?= ml_env

PREPROCESSING_SCRIPT = ml_houseprice_prediction/src/ml_houseprice_prediction/data_preprocessing/preprocessing.py
SPLITS_SCRIPT        = ml_houseprice_prediction/src/ml_houseprice_prediction/data_splits/splits.py
TRAIN_SCRIPT         = ml_houseprice_prediction/src/ml_houseprice_prediction/train_model/train.py


# --------------------------- DEFAULT TARGETS ------------------------------
.PHONY: env_update install_dependencies update_dependencies clean split train pipeline

# ======================================================================
# ENVIRONMENT MANAGEMENT
# ======================================================================

# Update conda environment
env_update:
	@echo "=> Updating conda environment from conda.yaml (env: $(CONDA_ENV))"
	conda env update -f conda.yaml 
	@echo "=> Conda environment '$(CONDA_ENV)' updated successfully."

# ======================================================================
# DEPENDENCY MANAGEMENT
# ======================================================================

# Install project dependencies (Poetry)
install_dependencies:
	@echo "=> Installing project dependencies..."
	poetry install
	@echo "=> Dependencies installed successfully."

# Update all dependencies (Poetry)
update_dependencies:
	@echo "=> Updating project dependencies..."
	poetry update
	@echo "=> Dependencies updated successfully."

# ======================================================================
# DATA PREPROCESSING, SPLITING & TRAINING
# ======================================================================

# Run data preprocessing script
clean:
	@echo "=> Running data preprocessing..."
	$(PYTHON) $(PREPROCESSING_SCRIPT) \
		--input_data_path $(INPUT_DATA_PATH) \
		--output_data_filename $(OUTPUT_FILENAME)
	@echo "=> Data preprocessing completed. Clean data saved to $(OUTPUT_FILENAME)."

# Run data splitting script
split:
	@echo "=> Running splits data..."
	$(PYTHON) $(SPLITS_SCRIPT) \
		--input_data_path datastores/clean_data/$(OUTPUT_FILENAME)
	@echo "=> Splits data completed. Train/test saved in datastores/splits_data/."

# Run training script
train:
	@echo "=> Running train model..."
	$(PYTHON) $(TRAIN_SCRIPT) \
		--input_train_data $(INPUT_TRAIN_DATA) \
		--input_test_data $(INPUT_TEST_DATA) \
		--model_filename $(MODEL_FILENAME)
	@echo "=> Train model completed successfully. Model saved in modelstores/$(MODEL_FILENAME)."

# ALL-IN-ONE WORKFLOW : local ci pipeline
pipeline: clean split train
	@echo "=========================================="
	@echo "     FULL PIPELINE EXECUTED SUCCESSFULLY"
	@echo "=========================================="


