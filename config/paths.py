from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
METADATA_DIR    = DATA_DIR / "metadata"
IMAGES_DIR      = DATA_DIR / "images"
ARRAYS_DIR      = DATA_DIR / "arrays"
FEATURES_DIR    = PROJECT_ROOT / "features"
MODELS_DIR      = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOG_DIR         = PROJECT_ROOT / "logs"

# Hugging Face repository IDs
HF_DATASET_REPO = "tkbarb10/ADS504-Image-Arrays"
HF_MODEL_REPO   = "tkbarb10/ADS504-Image-Model"
