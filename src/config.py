import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Model paths
BREAST_CANCER_MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_model.pkl")
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")
HEART_DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_model.pkl")
PARKINSONS_MODEL_PATH = os.path.join(MODEL_DIR, "parkinsons_model.pkl")

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2 