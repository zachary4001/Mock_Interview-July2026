# src/data_config.py
# Dataset-specific configuration for loans_modified.csv

EXPECTED_COLUMNS = [
    "loan_id", "gender", "married", "dependents", "education",
    "self_employed", "applicant_income", "coapplicant_income",
    "loan_amount", "loan_amount_term", "credit_history",
    "property_area", "loan_status",
]

TARGET_COLUMN = "loan_status"

# Raw column groups
RAW_NUMERIC_COLUMNS = [
    "applicant_income", "coapplicant_income", "loan_amount",
    "loan_amount_term", "credit_history",
]
RAW_CATEGORICAL_COLUMNS = [
    "gender", "married", "dependents", "education",
    "self_employed", "property_area",
]

# Backward-compatible names used by profiling utilities.
NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS
CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS

ID_COLUMNS = ["loan_id"]

# Rows missing these fields are treated as incomplete applications.
REQUIRED_COLUMNS = ["loan_status", "loan_amount", "loan_id"]

# MISSING-value notes (from EDA):
# self_employed: 34 missing (6.04%)
# coapplicant_income: 34 missing (6.04%)
# dependents: 32 missing (5.68%)
# loan_amount: 30 missing (5.33%)
# loan_id: 29 missing (5.15%)
# gender: 29 missing (5.15%)
# loan_amount_term: 28 missing (4.97%)
# loan_status: 28 missing (4.97%)
# applicant_income: 26 missing (4.62%)
# credit_history: 22 missing (3.91%)
# education: 22 missing (3.91%)
# property_area: 21 missing (3.73%)
# married: 19 missing (3.37%)
#
# Decisions anchored to the EDA markdown:
# - rows missing loan_status, loan_amount, or loan_id are removed;
# - applications with no positive total income are removed;
# - categorical values use explicit domain-informed fills;
# - coapplicant_income uses zero when missing;
# - credit_history uses zero when missing;
# - applicant_income and loan_amount_term use their training-data median;
# - target-dependent imputation is intentionally avoided to prevent leakage.

# Explicit imputation decisions.
CATEGORICAL_FILL_VALUES = {
    "gender": "Male",
    "married": "No",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "property_area": "Semiurban",
}
NUMERIC_FILL_VALUES = {
    "applicant_income": "median",
    "coapplicant_income": 0,
    "loan_amount_term": "median",
    "credit_history": 0,
}

# Derived feature names.
HAS_DEPENDENTS_COLUMN = "has_dependents"
TOTAL_INCOME_COLUMN = "total_income"
DEBT_INCOME_RATIO_COLUMN = "debt_income_ratio"
DERIVED_COLUMNS = [
    HAS_DEPENDENTS_COLUMN,
    TOTAL_INCOME_COLUMN,
    DEBT_INCOME_RATIO_COLUMN,
]

# Columns removed after their information has been used.
DROP_COLUMNS_AFTER_CLEANING = ["loan_id", "dependents"]
DROP_COLUMNS = DROP_COLUMNS_AFTER_CLEANING

# Validation bounds inferred from the raw EDA and data dictionary.
VALIDATION_BOUNDS = {
    "applicant_income": {"min": 0},
    "coapplicant_income": {"min": 0},
    "loan_amount": {"min": 0},
    "loan_amount_term": {"min": 12, "max": 480},
    "credit_history": {"min": 0, "max": 1},
    "loan_status": {"min": 0, "max": 1},
}
NOISY_FEATURES = []

# Model-ready feature groups.
CATEGORICAL_FEATURE_COLUMNS = [
    "gender", "married", "education", "self_employed", "property_area",
]
NUMERIC_FEATURE_COLUMNS = [
    "applicant_income", "coapplicant_income", "loan_amount",
    "loan_amount_term", "credit_history", HAS_DEPENDENTS_COLUMN,
    TOTAL_INCOME_COLUMN, DEBT_INCOME_RATIO_COLUMN,
]
FEATURE_COLUMNS = CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS

ENCODING_MAP = {
    column: "categorical"
    for column in CATEGORICAL_FEATURE_COLUMNS
}
ENCODING_MAP.update({
    column: "numeric"
    for column in NUMERIC_FEATURE_COLUMNS
})

ORDINAL_CATEGORIES = {}

# Baseline specifications preserve the initial broad feature configuration
# used for early model comparisons and reproducible exploratory analysis.
BASELINE_MODEL_SPECS = {
    "classification": {
        "feature_columns": FEATURE_COLUMNS,
        "encoding_map": ENCODING_MAP,
        "ordinal_categories": ORDINAL_CATEGORIES,
        "primary_metric": "roc_auc",
    },
    "regression": {
        "feature_columns": FEATURE_COLUMNS,
        "encoding_map": ENCODING_MAP,
        "ordinal_categories": ORDINAL_CATEGORIES,
        "primary_metric": "rmse",
    },
}

# Final classification definitions are selected through training-only
# validation and promoted here only after human review.
CLASSIFICATION_FEATURE_COLUMNS = [
    "credit_history",
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "married",
    "self_employed",
]

CLASSIFICATION_ENCODING_MAP = {
    column: "categorical"
    for column in ["married", "self_employed"]
}
CLASSIFICATION_ENCODING_MAP.update({
    column: "numeric"
    for column in [
        "credit_history",
        "applicant_income",
        "coapplicant_income",
        "loan_amount",
    ]
})

# MODEL_SPECS is the canonical final registry used by default by the model
# pipeline. BASELINE_MODEL_SPECS remains available for early comparisons.
MODEL_SPECS = {
    "classification": {
        "feature_columns": CLASSIFICATION_FEATURE_COLUMNS,
        "encoding_map": CLASSIFICATION_ENCODING_MAP,
        "ordinal_categories": ORDINAL_CATEGORIES,
        "primary_metric": "roc_auc",
    },
    "regression": {
        "feature_columns": FEATURE_COLUMNS,
        "encoding_map": ENCODING_MAP,
        "ordinal_categories": ORDINAL_CATEGORIES,
        "primary_metric": "rmse",
    },
}
