from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src import config
from src.data_config import (
    BASELINE_MODEL_SPECS,
    CLASSIFICATION_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_SPECS,
)
from src.model_pipeline import (
    build_encoder,
    get_model_spec,
    split_classification_and_checkpoint,
)


EXPECTED_FINAL_CLASSIFICATION_FEATURES = [
    "credit_history",
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "married",
    "self_employed",
]


def make_model_frame(rows=20):
    data = {
        "loan_status": [0, 1] * (rows // 2),
        "credit_history": [0, 1] * (rows // 2),
        "applicant_income": list(range(100, 100 + rows)),
        "coapplicant_income": [0] * rows,
        "loan_amount": [100] * rows,
        "loan_amount_term": [360] * rows,
        "married": ["Yes", "No"] * (rows // 2),
        "self_employed": ["No", "Yes"] * (rows // 2),
        "gender": ["Male"] * rows,
        "education": ["Graduate"] * rows,
        "property_area": ["Urban"] * rows,
        "has_dependents": [0] * rows,
        "total_income": list(range(100, 100 + rows)),
        "debt_income_ratio": [1.0] * rows,
    }
    return pd.DataFrame(data)


def test_baseline_and_final_registries_are_distinct():
    assert len(BASELINE_MODEL_SPECS["classification"]["feature_columns"]) == 13
    assert CLASSIFICATION_FEATURE_COLUMNS == EXPECTED_FINAL_CLASSIFICATION_FEATURES
    assert MODEL_SPECS["classification"]["feature_columns"] == EXPECTED_FINAL_CLASSIFICATION_FEATURES
    assert MODEL_SPECS["regression"]["feature_columns"] == FEATURE_COLUMNS


def test_get_model_spec_accepts_explicit_registry():
    baseline = get_model_spec("classification", specs=BASELINE_MODEL_SPECS)
    final = get_model_spec("classification", specs=MODEL_SPECS)

    assert len(baseline["feature_columns"]) == 13
    assert final["feature_columns"] == EXPECTED_FINAL_CLASSIFICATION_FEATURES


def test_encoder_uses_selected_registry():
    frame = make_model_frame()

    baseline_encoder = build_encoder(
        task="classification",
        scaled=True,
        specs=BASELINE_MODEL_SPECS,
    )
    final_encoder = build_encoder(
        task="classification",
        scaled=True,
        specs=MODEL_SPECS,
    )

    baseline_encoder.fit(frame[BASELINE_MODEL_SPECS["classification"]["feature_columns"]])
    final_encoder.fit(frame[EXPECTED_FINAL_CLASSIFICATION_FEATURES])

    assert baseline_encoder.transform(
        frame[BASELINE_MODEL_SPECS["classification"]["feature_columns"]]
    ).shape[0] == len(frame)
    assert final_encoder.transform(
        frame[EXPECTED_FINAL_CLASSIFICATION_FEATURES]
    ).shape[0] == len(frame)


def test_split_uses_explicit_registry_and_preserves_default_path(monkeypatch):
    frame = make_model_frame()

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        monkeypatch.setattr(config, "EXPORTS_PATH", temp_dir)

        baseline_split = split_classification_and_checkpoint(
            frame,
            random_state=7,
            specs=BASELINE_MODEL_SPECS,
        )
        final_split = split_classification_and_checkpoint(
            frame,
            random_state=7,
            specs=MODEL_SPECS,
        )

    assert baseline_split["X_train"].shape[1] == 13
    assert final_split["X_train"].columns.tolist() == EXPECTED_FINAL_CLASSIFICATION_FEATURES
    assert final_split["X_test"].shape[1] == 6
