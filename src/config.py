import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Config options
DATASET_NAME = "ml-1m"  # Support 'ml-1m' or 'ml-25m'
RANDOM_SEED = 42

# CF Parameters
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
SVD_LR_ALL = 0.005
SVD_REG_ALL = 0.02

# Content-Based Parameters
TFIDF_MAX_FEATURES = 5000

# Hybrid Parameters
SIGMOID_TAU = 10
SIGMOID_BETA = 0.5

# Evaluation Parameters
COLD_START_THRESHOLD = 5
RATING_THRESHOLD = 4.0
K_LIST = [5, 10, 20]
NUM_NEGATIVE_CANDIDATES = 500

# MMR Parameters
MMR_LAMBDA = 0.5
