# config.py
# this config file is for the Mock Interview using Load data
# updated 23.08.2026

import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Detect if running on Streamlit Cloud or locally
IS_LOCAL = os.path.exists(r'Q:\scripts\projects\Mock_Interview-July2026')
PROJECT_PATH = r'Q:\scripts\projects\Mock_Interview-July2026' if IS_LOCAL else os.path.dirname(os.path.abspath(__file__))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

# Paths
SRC_PATH     = os.path.join(PROJECT_PATH, 'src')
DATA_PATH    = os.path.join(PROJECT_PATH, 'data')
UTILS_PATH   = os.path.join(PROJECT_PATH, 'utils')
OUTPUTS_PATH = os.path.join(PROJECT_PATH, 'outputs')
FIGURES_PATH = os.path.join(OUTPUTS_PATH, 'figures')
REPORTS_PATH = os.path.join(OUTPUTS_PATH, 'reports')
EXPORTS_PATH = os.path.join(PROJECT_PATH, 'exports')
MODELS_PATH = os.path.join(OUTPUTS_PATH, 'models')

# Compatibility aliases used by the modeling pipeline.
FIGURES_DIR = FIGURES_PATH
REPORTS_DIR = REPORTS_PATH

# Dataset
FILE_NAME     = "loans_modified.csv"
FILE_PATH = os.path.join(DATA_PATH, FILE_NAME)
TRAIN_FILE = FILE_PATH

# DATE_COLUMN   = "date"
TARGET_COLUMN = "loan_status"

# MLflow
MLFLOW_URI    = os.getenv('MLFLOW_URI', 'http://localhost:5000')
EXPERIMENT    = "loan-approval-classification"

# Reproducibility
RANDOM_SEED  = 53
