"""Load the trained model and score protein pairs."""
import json
from functools import lru_cache
from pathlib import Path

import joblib

from .features import pair_features, protein_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_DIR / "ppi_model.joblib"):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train it first: python -m ppi_ai.train")
    return joblib.load(path)


def load_metrics(path: Path = MODEL_DIR / "metrics.json") -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def predict_interaction(seq_a: str, seq_b: str) -> float:
    """Probability (0-1) that the two proteins physically interact."""
    x = pair_features(protein_features(seq_a), protein_features(seq_b)).reshape(1, -1)
    return float(load_model().predict_proba(x)[0, 1])
