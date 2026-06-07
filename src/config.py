from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"

SENSOR_CHANNELS = list(range(8))
BOARD_IDS = ["B1", "B2", "B3", "B4", "B5"]
TARGET_GAS = "methane"

TRAIN_BOARDS = ["B1", "B2", "B3"]
VALID_BOARD = "B4"
TEST_BOARD = "B5"

SAMPLING_HZ = 100
EXPERIMENT_DURATION_S = 600
