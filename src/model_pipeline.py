# src/model_pipeline.py
# Modeling pipeline: split, encode, train, evaluate
# Updated: 23.08.2026

import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from src import config
from src.data_config import (
    MODEL_SPECS,
    TARGET_COLUMN,
)

def train_dummy_classifier(
    X_train,
    y_train,
    random_state=None,
):
    model = DummyClassifier(
        strategy="most_frequent",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def get_model_spec(task: str, specs: dict | None = None) -> dict:
    """Return a task specification from the selected registry."""
    specs = MODEL_SPECS if specs is None else specs
    if task not in specs:
        raise ValueError(
            f"Unsupported task {task!r}. "
            f"Available tasks: {list(specs)}"
        )
    return specs[task]


def split_and_checkpoint(
    df,
    task="classification",
    test_size=0.2,
    random_state=None,
    specs=None,
):
    """
    Split df into train/test sets using the selected model specification.
    Save split dict to EXPORTS_PATH as a timestamped .pkl checkpoint.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame (output of run_preprocessing_pipeline()).
    task : str
        Model purpose from data_config.MODEL_SPECS.
    test_size : float
        Fraction of data held out for testing. Default 0.2 (80/20 split).
    random_state : int or None
        Seed for reproducibility. Defaults to config.RANDOM_SEED if None.
    specs : dict or None
        Model specification registry. Defaults to final MODEL_SPECS.

    Returns
    -------
    dict
        Keys: X_train, X_test, y_train, y_test, train_idx, test_idx, checkpoint_path
    """
    if random_state is None:
        random_state = config.RANDOM_SEED

    model_spec = get_model_spec(task, specs=specs)
    X = df[model_spec["feature_columns"]]
    y = df[TARGET_COLUMN]
    if y.isna().any():
        raise ValueError(
            f"{TARGET_COLUMN} contains missing values before splitting"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task == "classification" else None,
    )

    split_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_idx": X_train.index,
        "test_idx": X_test.index,
        "random_state": random_state,
        "test_size": test_size,
    }

    # Ensure EXPORTS_PATH exists
    os.makedirs(config.EXPORTS_PATH, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task}_train_test_split_{timestamp}.pkl"
    checkpoint_path = os.path.join(config.EXPORTS_PATH, filename)

    joblib.dump(split_dict, checkpoint_path)
    split_dict["checkpoint_path"] = checkpoint_path

    print(f"Split checkpoint saved: {checkpoint_path}")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    return split_dict


def split_classification_and_checkpoint(
    df,
    test_size=0.2,
    random_state=None,
    specs=None,
):
    return split_and_checkpoint(
        df,
        task="classification",
        test_size=test_size,
        random_state=random_state,
        specs=specs,
    )


def load_split_checkpoint(checkpoint_path):
    """Load a previously saved split checkpoint."""
    return joblib.load(checkpoint_path)


def _columns_by_type(encoding_map, transformer_type):
    """Return list of column names matching a given ENCODING_MAP transformer type."""
    return [col for col, t in encoding_map.items() if t == transformer_type]


def build_encoder(
    task="classification",
    scaled=False,
    specs=None,
):
    """
    Build a ColumnTransformer from the selected model specification.

    Parameters
    ----------
    task : str
        Model purpose from data_config.MODEL_SPECS.
    scaled : bool
        If True, numeric columns are passed through StandardScaler.
    specs : dict or None
        Model specification registry. Defaults to final MODEL_SPECS.

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    model_spec = get_model_spec(task, specs=specs)
    encoding_map = model_spec["encoding_map"]
    ordinal_categories = model_spec["ordinal_categories"]

    categorical_cols = _columns_by_type(encoding_map, "categorical")
    ordinal_cols = _columns_by_type(encoding_map, "ordinal")
    numeric_cols = _columns_by_type(encoding_map, "numeric")
    passthrough_cols = _columns_by_type(encoding_map, "pass_through")

    ordinal_categories_ordered = [ordinal_categories[col] for col in ordinal_cols]

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(categories=ordinal_categories_ordered),
        ),
    ])
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)
    passthrough_transformer = SimpleImputer(strategy="median")

    transformers = [
        ("onehot", categorical_transformer, categorical_cols),
        ("ordinal", ordinal_transformer, ordinal_cols),
        ("numeric", numeric_transformer, numeric_cols),
        ("pass", passthrough_transformer, passthrough_cols),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_and_checkpoint_encoders(X_train, task="classification", specs=None):
    """
    Fit both scaled and raw encoders for the selected model purpose.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (from split_and_checkpoint output).
    specs : dict or None
        Model specification registry. Defaults to final MODEL_SPECS.

    Returns
    -------
    dict
        Keys: encoder_scaled, encoder_raw, scaled_path, raw_path
    """
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    encoder_scaled = build_encoder(task=task, scaled=True, specs=specs)
    encoder_scaled.fit(X_train)
    scaled_path = os.path.join(
        config.MODELS_PATH, f"{task}_encoder_fitted_scaled_{timestamp}.pkl"
    )
    joblib.dump(encoder_scaled, scaled_path)

    encoder_raw = build_encoder(task=task, scaled=False, specs=specs)
    encoder_raw.fit(X_train)
    raw_path = os.path.join(
        config.MODELS_PATH, f"{task}_encoder_fitted_raw_{timestamp}.pkl"
    )
    joblib.dump(encoder_raw, raw_path)

    print(f"Scaled encoder saved: {scaled_path}")
    print(f"Raw encoder saved: {raw_path}")

    return {
        "encoder_scaled": encoder_scaled,
        "encoder_raw": encoder_raw,
        "scaled_path": scaled_path,
        "raw_path": raw_path,
    }


def load_encoder_checkpoint(checkpoint_path):
    """Load a previously fitted encoder checkpoint."""
    return joblib.load(checkpoint_path)


def evaluate_regressor(model, X_test, y_test, model_name, train_time_sec=None):
    """
    Evaluate a fitted model on test data; return metrics dict.

    Parameters
    ----------
    model : estimator
        Fitted regression model.
    X_test : pd.DataFrame or array-like
        Test features (transformed/encoded).
    y_test : pd.Series or array-like
        Test target values.
    model_name : str
        Human-readable model name (e.g., "Linear Regression", "Decision Tree").
    train_time_sec : float, optional
        Training time in seconds. If None, will be recorded as None in the dict.

    Returns
    -------
    dict
        Keys: model_name, MAE, RMSE, R2, train_time_sec
        All numeric values are floats; times as floats (seconds).
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "model_name": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "train_time_sec": train_time_sec,
    }


def train_linear_regression(X_train, y_train, random_state=None):
    """
    Train a Linear Regression baseline.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Encoded training features (output of fitted encoder.transform()).
    y_train : array-like, shape (n_samples,)
        Training target values.
    random_state : int, optional
        Not used by LinearRegression but kept for API consistency.

    Returns
    -------
    LinearRegression (fitted)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def demo_linear_regression(df, encoder_checkpoint_path):
    """
    Demo: Load split, encoder, train Linear Regression, evaluate.

    Used for M-02 verification only. In production, final_analysis.ipynb
    orchestrates this step-by-step.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame (output of run_preprocessing_pipeline()).
        Not used directly here — the demo loads the already-split checkpoint
        from EXPORTS_PATH rather than re-splitting df. Kept in the signature
        to mirror the intended notebook orchestration call shape.
    encoder_checkpoint_path : str
        Path to encoder_fitted_scaled_*.pkl from M-08.

    Returns
    -------
    dict
        Keys: model, metrics, split_checkpoint_path, encoder_checkpoint_path
    """
    # Step 1: Load split checkpoint (M-01)
    latest_split_pkl = sorted(
        [
            f for f in os.listdir(config.EXPORTS_PATH)
            if f.startswith("regression_train_test_split_")
        ],
        reverse=True
    )[0]
    split_path = os.path.join(config.EXPORTS_PATH, latest_split_pkl)
    split = load_split_checkpoint(split_path)

    # Step 2: Load encoder checkpoint (M-08)
    encoder = load_encoder_checkpoint(encoder_checkpoint_path)

    # Step 3: Encode training data
    X_train_encoded = encoder.transform(split["X_train"])
    X_test_encoded = encoder.transform(split["X_test"])

    # Step 4: Train Linear Regression
    start_time = time.time()
    model = train_linear_regression(X_train_encoded, split["y_train"])
    train_time_sec = time.time() - start_time

    # Step 5: Evaluate
    metrics = evaluate_regressor(
        model,
        X_test_encoded,
        split["y_test"],
        model_name="Linear Regression",
        train_time_sec=train_time_sec
    )

    print(f"\n=== Linear Regression Baseline ===")
    print(f"  MAE:  ${metrics['MAE']:.2f}")
    print(f"  RMSE: ${metrics['RMSE']:.2f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  Time: {train_time_sec:.3f}s")

    return {
        "model": model,
        "metrics": metrics,
        "split_checkpoint_path": split_path,
        "encoder_checkpoint_path": encoder_checkpoint_path,
    }


def train_logistic_regression(X_train, y_train, random_state=None):
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree_classifier(X_train, y_train, random_state=None):
    if random_state is None:
        random_state = config.RANDOM_SEED

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_classifier(
    X_train,
    y_train,
    random_state=None,
):
    if random_state is None:
        random_state = config.RANDOM_SEED

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model,
    X_test,
    y_test,
    model_name,
    train_time_sec=None,
):
    y_pred = model.predict(X_test)
    y_probability = None
    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time_sec": train_time_sec,
    }
    metrics["roc_auc"] = (
        roc_auc_score(y_test, y_probability)
        if y_probability is not None
        else None
    )
    return metrics


def update_classification_leaderboard_csv(
    metrics_dict,
    leaderboard_path=None,
):
    if leaderboard_path is None:
        leaderboard_path = os.path.join(
            config.REPORTS_DIR,
            "classification_model_leaderboard.csv",
        )

    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
    row = {
        "model_name": metrics_dict["model_name"],
        "accuracy": metrics_dict["accuracy"],
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "f1": metrics_dict["f1"],
        "roc_auc": metrics_dict["roc_auc"],
        "train_time_sec": metrics_dict["train_time_sec"],
    }

    if os.path.exists(leaderboard_path):
        leaderboard = pd.read_csv(leaderboard_path)
        leaderboard = leaderboard[
            leaderboard["model_name"] != row["model_name"]
        ]
        leaderboard = pd.concat(
            [leaderboard, pd.DataFrame([row])],
            ignore_index=True,
        )
    else:
        leaderboard = pd.DataFrame([row])

    leaderboard = leaderboard.sort_values(
        ["roc_auc", "f1"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    leaderboard.to_csv(leaderboard_path, index=False)
    return leaderboard_path


def log_classification_model_to_mlflow(
    model,
    metrics_dict,
    encoder_checkpoint_path,
    split_checkpoint_path,
):
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT)

    with mlflow.start_run():
        for metric_name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "train_time_sec",
        ):
            metric_value = metrics_dict.get(metric_name)
            if metric_value is not None:
                mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_param("task", "classification")
        mlflow.log_param("model_name", metrics_dict["model_name"])
        mlflow.log_param(
            "encoder_checkpoint",
            os.path.basename(encoder_checkpoint_path),
        )
        mlflow.log_param(
            "split_checkpoint",
            os.path.basename(split_checkpoint_path),
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=None,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: {run_id}")
        return run_id


def update_regression_leaderboard_csv(metrics_dict, leaderboard_path=None):
    """
    Update or append a model's metrics to the leaderboard CSV.

    Strategy: Per-model snapshot (upsert). If model_name already exists,
    overwrite its row. Otherwise, append a new row.

    Parameters
    ----------
    metrics_dict : dict
        Output of evaluate_regressor(): {model_name, MAE, RMSE, R2, train_time_sec}
    leaderboard_path : str, optional
        Path to model_leaderboard.csv. Defaults to REPORTS_DIR/model_leaderboard.csv

    Returns
    -------
    str
        Path to updated CSV file
    """
    if leaderboard_path is None:
        leaderboard_path = os.path.join(config.REPORTS_DIR, "model_leaderboard.csv")

    # Ensure the leaderboard's parent directory exists
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)

    # Add timestamp to metrics dict for tracking
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    row = {
        "model_name": metrics_dict["model_name"],
        "MAE": metrics_dict["MAE"],
        "RMSE": metrics_dict["RMSE"],
        "R2": metrics_dict["R2"],
        "train_time_sec": metrics_dict["train_time_sec"],
        "last_updated": timestamp,
    }

    # Load existing leaderboard or create new
    if os.path.exists(leaderboard_path):
        df = pd.read_csv(leaderboard_path)
        # Upsert: drop any existing row for this model, then append the new one
        df = df[df["model_name"] != row["model_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    # Sort by RMSE (ascending) for easy comparison
    df = df.sort_values("RMSE").reset_index(drop=True)
    df.to_csv(leaderboard_path, index=False)

    print(f"Leaderboard updated: {leaderboard_path}")
    print(df.to_string(index=False))

    return leaderboard_path


def log_regression_model_to_mlflow(
    model,
    metrics_dict,
    encoder_checkpoint_path,
    split_checkpoint_path,
):
    """
    Log a trained model run to MLflow: metrics, params, artifacts.

    Parameters
    ----------
    model : estimator
        Fitted sklearn model.
    metrics_dict : dict
        Output of evaluate_regressor(): {model_name, MAE, RMSE, R2, train_time_sec}
    encoder_checkpoint_path : str
        Path to encoder_fitted_[scaled|raw]_*.pkl used for this model.
    split_checkpoint_path : str
        Path to train_test_split_*.pkl used for this model.

    Returns
    -------
    str
        MLflow run ID
    """
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT)

    with mlflow.start_run():
        # Log metrics
        mlflow.log_metric("MAE", metrics_dict["MAE"])
        mlflow.log_metric("RMSE", metrics_dict["RMSE"])
        mlflow.log_metric("R2", metrics_dict["R2"])
        if metrics_dict["train_time_sec"] is not None:
            mlflow.log_metric("train_time_sec", metrics_dict["train_time_sec"])

        # Log params
        mlflow.log_param("model_name", metrics_dict["model_name"])
        mlflow.log_param("encoder_checkpoint", os.path.basename(encoder_checkpoint_path))
        mlflow.log_param("split_checkpoint", os.path.basename(split_checkpoint_path))

        # Log model artifact
        mlflow.sklearn.log_model(
            model,
            # artifact_path="model",
            name="model",
            registered_model_name=None  # Don't auto-register; manual promotion later if needed
        )

        # Log checkpoint references as text artifacts (for traceability)
        with open("checkpoint_paths.txt", "w") as f:
            f.write(f"Encoder: {encoder_checkpoint_path}\n")
            f.write(f"Split: {split_checkpoint_path}\n")
        mlflow.log_artifact("checkpoint_paths.txt")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: {run_id}")
        return run_id


def train_decision_tree_regressor(X_train, y_train, random_state=None):
    """
    Train a Decision Tree Regressor baseline.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Encoded training features (output of fitted encoder.transform()).
    y_train : array-like, shape (n_samples,)
        Training target values.
    random_state : int, optional
        Seed for reproducibility. Defaults to config.RANDOM_SEED if None.

    Returns
    -------
    DecisionTreeRegressor (fitted)
    """
    if random_state is None:
        random_state = config.RANDOM_SEED

    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model
