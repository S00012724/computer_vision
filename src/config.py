"""Project configuration constants"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
TEST_RATIO = 0.2

CNN_MODEL_PATH = PROJECT_ROOT / "models" / "model.keras"
CNN_IMAGE_SIZE = (96, 96)
