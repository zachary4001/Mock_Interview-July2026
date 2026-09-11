# src/loans_preprocessing.py
# v1.0 - updated 23.08.26
# Dataset-specific cleaning and feature engineering for loans_modified.csv

import pandas as pd

from src.data_config import (
    CATEGORICAL_FILL_VALUES,
    DEBT_INCOME_RATIO_COLUMN,
    DROP_COLUMNS_AFTER_CLEANING,
    HAS_DEPENDENTS_COLUMN,
    NUMERIC_FILL_VALUES,
    REQUIRED_COLUMNS,
    TOTAL_INCOME_COLUMN,
    VALIDATION_BOUNDS,
)


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def remove_incomplete_applications(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    income_total = (
        df["applicant_income"].fillna(0)
        + df["coapplicant_income"].fillna(0)
    )
    return df.loc[income_total > 0].copy()


def validate_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    for column, bounds in VALIDATION_BOUNDS.items():
        if column not in df.columns:
            continue

        if "min" in bounds and (df[column] < bounds["min"]).any():
            raise ValueError(
                f"{column} contains values below {bounds['min']}"
            )
        if "max" in bounds and (df[column] > bounds["max"]).any():
            raise ValueError(
                f"{column} contains values above {bounds['max']}"
            )

    return df.copy()


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column, fill_value in CATEGORICAL_FILL_VALUES.items():
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)

    for column, fill_value in NUMERIC_FILL_VALUES.items():
        if column not in df.columns:
            continue
        if fill_value == "median":
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(fill_value)

    return df


def normalize_dependents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dependents"] = (
        df["dependents"].astype(str).replace({"3+": "3"})
    )
    df[HAS_DEPENDENTS_COLUMN] = (
        df["dependents"].astype(int) > 0
    ).astype(int)
    return df


def engineer_income_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TOTAL_INCOME_COLUMN] = (
        df["applicant_income"] + df["coapplicant_income"]
    )
    df[DEBT_INCOME_RATIO_COLUMN] = (
        df[TOTAL_INCOME_COLUMN] / df["loan_amount"]
    )
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=DROP_COLUMNS_AFTER_CLEANING,
        errors="ignore",
    ).copy()


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    df = remove_duplicate_rows(df)
    df = remove_incomplete_applications(df)
    return validate_numeric_values(df)


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = impute_missing_values(df)
    df = normalize_dependents(df)
    df = engineer_income_features(df)
    return drop_unused_columns(df)


def prepare_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    df = run_cleaning_pipeline(df)
    return run_preprocessing_pipeline(df)
