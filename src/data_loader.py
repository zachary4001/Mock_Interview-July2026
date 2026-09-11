# data_loader.py
# v1.1 23.08.26

from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime
from src.config import EXPORTS_PATH


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV dataset."""

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)

def save_progress(name, value):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = Path(EXPORTS_PATH) / f'loan_{name}_{timestamp}.pkl'
    joblib.dump(value, path)
    print(f'Progress saved: {path}')
    return str(path)

def latest_progress(name):
    paths = sorted(Path(EXPORTS_PATH).glob(f'loan_{name}_*.pkl'), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f'No progress checkpoint found for {name!r}')
    return paths[-1]

def load_progress(name):
    path = latest_progress(name)
    value = joblib.load(path)
    print(f'Progress loaded: {path}')
    return value

def checkpoint_exists(name):
    return any(
        Path(EXPORTS_PATH).glob(f"loan_{name}_*.pkl")
    )

def use_checkpoint(name):
    return (
        RESUME_FROM is None
        and checkpoint_exists(name)
    )