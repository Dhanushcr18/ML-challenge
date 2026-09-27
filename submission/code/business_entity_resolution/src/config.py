from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_default_dataset = ROOT / "dataset"
DATASET_DIR = _default_dataset if (_default_dataset / "train").exists() else ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
MAX_KEY_POSTINGS = 250
MAX_CANDIDATES_PER_ENTITY = 300
MAX_TRAIN_NEGATIVES_PER_ENTITY = 20
VALIDATION_FRACTION = 0.2
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95)
