from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
METADATA_DIR    = DATA_DIR / "metadata"
IMAGES_DIR      = DATA_DIR / "images"
ARRAYS_DIR      = DATA_DIR / "arrays"
FEATURES_DIR    = PROJECT_ROOT / "features"
MODELS_DIR      = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
