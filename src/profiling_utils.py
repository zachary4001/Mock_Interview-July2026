# profiling_utils.py
# v1.3 - updated 23.08.26

import re
from pathlib import Path

import pandas as pd

from preprocessing import (
    missing_values,
    duplicate_rows,
    numeric_columns,
    categorical_columns,
)
from eda_statistics import (dataset_summary, descriptive_statistics)


def infer_column_roles(df: pd.DataFrame, target_column: str) -> dict:
    numeric = [c for c in numeric_columns(df) if c != target_column]
    categorical = [c for c in categorical_columns(df) if c != target_column]

    likely_id = []
    for col in categorical:
        nunique = df[col].nunique(dropna=True)
        if len(df) > 0 and nunique / len(df) > 0.9:
            likely_id.append(col)
    categorical = [c for c in categorical if c not in likely_id]

    return {
        "numeric": numeric,
        "categorical": categorical,
        "likely_id": likely_id,
        "target": target_column if target_column in df.columns else None,
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    counts = missing_values(df)
    pct = (counts / len(df) * 100).round(2) if len(df) else counts.astype(float)

    return (
        pd.DataFrame({"column": counts.index, "missing_count": counts.values, "missing_pct": pct.values})
        .sort_values("missing_count", ascending=False)
        .reset_index(drop=True)
    )


def feature_completeness_report(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Summarize row-level completeness for a selected feature set.

    Completeness is calculated from the supplied dataframe before imputation
    or row removal. A row is incomplete when at least one selected feature is
    missing. The returned per-feature table reports individual missingness
    without confusing it with the row-level union.
    """
    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Feature columns are missing from dataframe: {missing_columns}"
        )
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one column")

    selected = df[feature_columns]
    row_missing = selected.isna().any(axis=1)
    total_rows = len(selected)
    incomplete_rows = int(row_missing.sum())
    complete_rows = total_rows - incomplete_rows

    per_feature = pd.DataFrame({
        "feature": feature_columns,
        "missing_count": selected.isna().sum().astype(int).values,
        "missing_pct": (selected.isna().mean() * 100).values,
    })

    return {
        "feature_columns": list(feature_columns),
        "total_rows": total_rows,
        "complete_rows": complete_rows,
        "incomplete_rows": incomplete_rows,
        "complete_pct": (complete_rows / total_rows * 100) if total_rows else 0.0,
        "incomplete_pct": (incomplete_rows / total_rows * 100) if total_rows else 0.0,
        "per_feature": per_feature,
    }


def generate_eda_summary(df: pd.DataFrame, target_column: str) -> str:
    roles = infer_column_roles(df, target_column)
    summary = dataset_summary(df)
    missing_report = missing_value_report(df)
    dup_count = duplicate_rows(df)

    lines = []
    lines.append("=" * 60)
    lines.append("EDA SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Shape: {summary['Rows']} rows x {summary['Columns']} columns")
    lines.append("")

    lines.append("Dtypes:")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col}: {dtype}")
    lines.append("")

    lines.append("Missing values:")
    non_zero = missing_report.query("missing_count > 0")
    if non_zero.empty:
        lines.append("  None")
    else:
        for _, row in non_zero.iterrows():
            lines.append(f"  {row['column']}: {row['missing_count']} ({row['missing_pct']}%)")
    lines.append("")

    lines.append(f"Duplicate rows: {dup_count}")
    lines.append("")

    if roles["target"] and target_column in df.columns:
        lines.append(f"Target balance ({target_column}):")
        counts = df[target_column].value_counts(dropna=False)
        pct = df[target_column].value_counts(normalize=True, dropna=False) * 100
        for value in counts.index:
            lines.append(f"  {value}: {counts[value]} ({pct[value]:.1f}%)")
        lines.append("")

    lines.append("Numeric column ranges:")
    if roles["numeric"]:
        desc = descriptive_statistics(df[roles["numeric"]])
        for col in roles["numeric"]:
            lines.append(f"  {col}: min={desc.loc['min', col]}, max={desc.loc['max', col]}, mean={desc.loc['mean', col]:.2f}")
    else:
        lines.append("  None")
    lines.append("")

    lines.append("Categorical column cardinality:")
    if roles["categorical"]:
        for col in roles["categorical"]:
            lines.append(f"  {col}: {df[col].nunique(dropna=True)} unique values")
    else:
        lines.append("  None")
    lines.append("")

    if roles["likely_id"]:
        lines.append("Likely-ID columns (high-cardinality, excluded from categorical):")
        for col in roles["likely_id"]:
            lines.append(f"  {col}")
        lines.append("")

    return "\n".join(lines)


def draft_data_config(df: pd.DataFrame, target_column: str, dataset_name: str) -> dict:
    roles = infer_column_roles(df, target_column)
    missing_report = missing_value_report(df)
    missing_notes = [
        f"{row['column']}: {row['missing_count']} missing ({row['missing_pct']}%)"
        for _, row in missing_report.query("missing_count > 0").iterrows()
    ]

    structural = {
        "dataset_name": dataset_name,
        "EXPECTED_COLUMNS": df.columns.tolist(),
        "NUMERIC_COLUMNS": roles["numeric"],
        "CATEGORICAL_COLUMNS": roles["categorical"],
        "TARGET_COLUMN": target_column,
        "missing_value_notes": missing_notes,
    }

    model_specs_template = {
        "classification": {
            "feature_columns": [],
            "encoding_map": None,
            "ordinal_categories": {},
            "primary_metric": "roc_auc",
        },
        "regression": {
            "feature_columns": [],
            "encoding_map": None,
            "ordinal_categories": {},
            "primary_metric": "rmse",
        },
    }

    needs_review = {
        "IMPUTATION_STRATEGY": None,
        "FEATURE_COLUMNS": [],
        "ENCODING_MAP": None,
        "BASELINE_MODEL_SPECS": model_specs_template,
        "MODEL_SPECS": model_specs_template.copy(),
        "NOISY_FEATURES": [],
        "VALIDATION_BOUNDS": None,
    }

    return {"structural": structural, "needs_review": needs_review}


def write_data_config(
    draft: dict,
    output_path: str,
    overwrite: bool = False,
) -> str:
    output_path = Path(output_path)

    # Never replace the curated dataset-specific configuration automatically.
    # A draft may be regenerated, but data_config.py must be changed manually.
    if output_path.name == "data_config.py" and output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing curated configuration: {output_path}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without overwrite=True: {output_path}"
        )

    structural = draft["structural"]
    needs_review = draft["needs_review"]

    lines = []
    lines.append(f"# Auto-generated data_config.py for dataset: {structural['dataset_name']}")
    lines.append("# Structural fields below were inferred directly from the data.")
    lines.append("")
    lines.append("# --- Structural (auto-populated) ---")
    lines.append(f"EXPECTED_COLUMNS = {structural['EXPECTED_COLUMNS']!r}")
    lines.append(f"NUMERIC_COLUMNS = {structural['NUMERIC_COLUMNS']!r}")
    lines.append(f"CATEGORICAL_COLUMNS = {structural['CATEGORICAL_COLUMNS']!r}")
    lines.append(f"TARGET_COLUMN = {structural['TARGET_COLUMN']!r}")
    lines.append("")
    if structural["missing_value_notes"]:
        lines.append("# Missing-value notes (from EDA):")
        for note in structural["missing_value_notes"]:
            lines.append(f"# {note}")
    else:
        lines.append("# No missing values detected.")
    lines.append("")

    lines.append("# --- Needs review (not auto-inferred; requires human judgment) ---")
    review_reasons = {
        "IMPUTATION_STRATEGY": "requires domain judgment on how each column's missing values should be handled",
        "FEATURE_COLUMNS": "requires deciding which columns to retain/engineer for modeling",
        "ENCODING_MAP": "requires deciding encoding strategy (ordinal vs. one-hot) per column",
        "BASELINE_MODEL_SPECS": "requires defining the initial broad feature and encoding registry for each task",
        "MODEL_SPECS": "requires human validation and promotion of the final feature and encoding registry",
        "NOISY_FEATURES": "requires manual inspection to flag unreliable columns",
        "VALIDATION_BOUNDS": "requires domain knowledge of valid ranges per column",
    }
    for key, value in needs_review.items():
        reason = review_reasons.get(key, "requires human review")
        lines.append(f"# TODO: {reason}")
        lines.append(f"{key} = {value!r}")
        lines.append("")

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def summarize_unresolved(draft: dict) -> pd.DataFrame:
    needs_review = draft["needs_review"]
    reasons = {
        "IMPUTATION_STRATEGY": "requires domain judgment on how each column's missing values should be handled",
        "FEATURE_COLUMNS": "requires deciding which columns to retain/engineer for modeling",
        "ENCODING_MAP": "requires deciding encoding strategy (ordinal vs. one-hot) per column",
        "BASELINE_MODEL_SPECS": "requires defining the initial broad feature and encoding registry for each task",
        "MODEL_SPECS": "requires human validation and promotion of the final feature and encoding registry",
        "NOISY_FEATURES": "requires manual inspection to flag unreliable columns",
        "VALIDATION_BOUNDS": "requires domain knowledge of valid ranges per column",
    }

    rows = [
        {"key": key, "reason": reasons.get(key, "requires human review")}
        for key in needs_review
    ]

    return pd.DataFrame(rows)


def classify_cardinality(
    series: pd.Series,
    max_groups: int = 10,
    coverage_threshold: float = 0.90,
) -> str | None:
    values = series.dropna()
    if values.empty:
        return None

    group_counts = values.value_counts()
    group_count = len(group_counts)
    total = len(values)
    top_two_coverage = group_counts.head(2).sum() / total
    top_groups_coverage = group_counts.head(max_groups).sum() / total

    if group_count <= 2 or top_two_coverage > coverage_threshold:
        return "binary"
    if group_count < max_groups or top_groups_coverage >= coverage_threshold:
        return "categorical"
    return None


def profile_low_cardinality_fields(
    df: pd.DataFrame,
    text_length_threshold: int = 15,
    max_groups: int = 10,
    coverage_threshold: float = 0.90,
) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        values = series.dropna()
        if values.empty:
            continue

        is_text = pd.api.types.is_string_dtype(series) or series.dtype == object
        short_text_fraction = None
        if is_text:
            short_text_fraction = values.astype(str).str.len().lt(text_length_threshold).mean()
            if short_text_fraction < coverage_threshold:
                continue

        potential_type = classify_cardinality(series, max_groups, coverage_threshold)
        if potential_type is None:
            continue

        counts = values.value_counts()
        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "non_null_count": len(values),
            "unique_count": values.nunique(),
            "top_group_count": int(counts.head(max_groups).sum()),
            "top_group_coverage": round(counts.head(max_groups).sum() / len(values), 4),
            "short_text_fraction": short_text_fraction,
            "potential_type": potential_type,
        })

    return pd.DataFrame(rows)


def _column_name_tokens(column: str) -> set[str]:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column))
    return {token for token in re.split(r"[^a-zA-Z0-9]+", name.lower()) if token}


def identify_related_columns(
    df: pd.DataFrame,
    min_shared_tokens: int = 1,
) -> pd.DataFrame:
    columns = list(df.columns)
    rows = []
    for index, column_a in enumerate(columns):
        tokens_a = _column_name_tokens(column_a)
        for column_b in columns[index + 1:]:
            shared_tokens = sorted(tokens_a & _column_name_tokens(column_b))
            if len(shared_tokens) < min_shared_tokens:
                continue

            prefix_a = str(column_a).lower().split("_")[0]
            prefix_b = str(column_b).lower().split("_")[0]
            relationship_type = "shared_prefix" if prefix_a == prefix_b else "shared_token"
            rows.append({
                "column_a": column_a,
                "column_b": column_b,
                "shared_tokens": ", ".join(shared_tokens),
                "relationship_type": relationship_type,
                "reason": "potential feature engineering and/or imputation relationship",
            })

    return pd.DataFrame(rows)


def missing_columns_by_row(df: pd.DataFrame) -> pd.DataFrame:
    missing_mask = df.isna()
    rows = []
    for row_index, row in missing_mask.iterrows():
        columns = row.index[row].tolist()
        if columns:
            rows.append({
                "row_index": row_index,
                "missing_count": len(columns),
                "missing_columns": ", ".join(map(str, columns)),
            })
    return pd.DataFrame(rows)


def build_missing_value_tables(
    df: pd.DataFrame,
    all_rows_limit: int = 1000,
) -> dict[str, pd.DataFrame]:
    row_profile = missing_columns_by_row(df)
    profile_columns = ["row_index", "missing_count", "missing_columns"]
    if row_profile.empty:
        empty = pd.DataFrame(columns=profile_columns)
        return {
            "all_missing_rows": empty,
            "multi_column_missing_rows": empty,
            "single_column_missing_rows": empty,
        }

    rows_with_missing = df.loc[row_profile["row_index"]].copy()
    rows_with_missing.insert(0, "row_index", row_profile["row_index"].tolist())
    rows_with_missing = rows_with_missing.merge(row_profile, on="row_index", how="left")

    all_missing_rows = (
        rows_with_missing.copy()
        if len(rows_with_missing) <= all_rows_limit
        else pd.DataFrame(columns=rows_with_missing.columns)
    )
    multi_column_missing_rows = rows_with_missing[
        rows_with_missing["missing_count"] > 1
    ].sort_values(["missing_count", "row_index"], ascending=[False, True])

    single_column_missing_rows = rows_with_missing[
        rows_with_missing["missing_count"] == 1
    ].copy()
    single_column_missing_rows["missing_column"] = (
        single_column_missing_rows["missing_columns"]
    )
    single_column_missing_rows = single_column_missing_rows.sort_values(
        ["missing_column", "row_index"]
    )

    return {
        "all_missing_rows": all_missing_rows,
        "multi_column_missing_rows": multi_column_missing_rows,
        "single_column_missing_rows": single_column_missing_rows,
    }


def profile_text_fields(
    df: pd.DataFrame,
    short_length_threshold: int = 5,
    numeric_fraction_threshold: float = 0.90,
) -> pd.DataFrame:
    rows = []
    for column in categorical_columns(df):
        values = df[column].dropna().astype(str)
        if values.empty:
            continue

        short_fraction = values.str.len().lt(short_length_threshold).mean()
        numeric_fraction = pd.to_numeric(values, errors="coerce").notna().mean()
        characteristics = []
        if short_fraction >= numeric_fraction_threshold:
            characteristics.append("short_text")
        if numeric_fraction > numeric_fraction_threshold:
            characteristics.append("primarily_numeric_text")

        if characteristics:
            rows.append({
                "column": column,
                "non_null_count": len(values),
                "unique_count": values.nunique(),
                "max_length": int(values.str.len().max()),
                "short_value_fraction": round(short_fraction, 4),
                "numeric_parse_fraction": round(numeric_fraction, 4),
                "characteristic": " and ".join(characteristics),
            })
    return pd.DataFrame(rows)


def profile_non_negative_numeric_fields(
    df: pd.DataFrame,
    lower_fraction: float = 0.90,
    upper_fraction: float = 1.00,
) -> pd.DataFrame:
    rows = []
    for column in numeric_columns(df):
        values = df[column].dropna()
        if values.empty:
            continue

        non_negative_count = int((values >= 0).sum())
        fraction = non_negative_count / len(values)
        if lower_fraction < fraction < upper_fraction:
            rows.append({
                "column": column,
                "non_null_count": len(values),
                "non_negative_count": non_negative_count,
                "negative_count": int((values < 0).sum()),
                "non_negative_fraction": round(fraction, 4),
            })
    return pd.DataFrame(rows)


def write_missing_value_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str,
    file_prefix: str = "missing_values",
) -> list[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for table_name, table in tables.items():
        if table.empty:
            continue
        table_path = output_path / f"{file_prefix}_{table_name}.csv"
        table.to_csv(table_path, index=False)
        written_paths.append(str(table_path))
    return written_paths


def generate_field_characteristics_report(
    df: pd.DataFrame,
    text_length_threshold: int = 15,
    short_length_threshold: int = 5,
    max_groups: int = 10,
    coverage_threshold: float = 0.90,
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
    return {
        "low_cardinality_fields": profile_low_cardinality_fields(
            df, text_length_threshold, max_groups, coverage_threshold
        ),
        "related_columns": identify_related_columns(df),
        "missing_value_tables": build_missing_value_tables(df),
        "text_field_profile": profile_text_fields(
            df, short_length_threshold, coverage_threshold
        ),
        "non_negative_numeric_fields": profile_non_negative_numeric_fields(df),
    }


def dominant_category_summary(
    df: pd.DataFrame,
    dominance_threshold: float = 0.60,
) -> pd.DataFrame:
    rows = []
    for column in categorical_columns(df):
        values = df[column].dropna()
        if values.empty:
            continue

        counts = values.value_counts()
        top_value = counts.index[0]
        top_value_pct = counts.iloc[0] / len(values)
        if top_value_pct >= dominance_threshold:
            imputation_note = "mode imputation reasonable"
        else:
            imputation_note = "low dominance; consider 'Unknown' category"

        rows.append({
            "column": column,
            "top_value": top_value,
            "top_value_pct": round(top_value_pct * 100, 2),
            "unique_count": values.nunique(),
            "imputation_note": imputation_note,
        })

    return pd.DataFrame(rows)


def numeric_distribution_summary(
    df: pd.DataFrame,
    skew_ratio_threshold: float = 0.10,
    zero_pct_threshold: float = 0.30,
) -> pd.DataFrame:
    rows = []
    for column in numeric_columns(df):
        values = df[column].dropna()
        if values.empty:
            continue

        mean = values.mean()
        median = values.median()
        difference = abs(mean - median)
        skew_ratio = difference if median == 0 else difference / abs(median)
        zero_count = int((values == 0).sum())
        zero_pct = zero_count / len(values)

        if skew_ratio >= skew_ratio_threshold:
            imputation_note = "median imputation preferred (skewed)"
        else:
            imputation_note = "mean/median comparable"
        if zero_pct >= zero_pct_threshold:
            imputation_note += "; zero-heavy; verify structural vs. missing"

        rows.append({
            "column": column,
            "mean": mean,
            "median": median,
            "skew_ratio": round(skew_ratio, 4),
            "zero_count": zero_count,
            "zero_pct": round(zero_pct * 100, 2),
            "imputation_note": imputation_note,
        })

    return pd.DataFrame(rows)


def surface_insights(
    df: pd.DataFrame,
    dominance_threshold: float = 0.60,
    skew_ratio_threshold: float = 0.10,
    zero_pct_threshold: float = 0.30,
) -> dict:
    categorical_summary = dominant_category_summary(df, dominance_threshold)
    numeric_summary = numeric_distribution_summary(
        df,
        skew_ratio_threshold,
        zero_pct_threshold,
    )

    narrative = []
    for _, row in categorical_summary.iterrows():
        narrative.append(
            f"{row['column']}: {row['top_value_pct']:.1f}% of non-null values "
            f"are '{row['top_value']}' -> {row['imputation_note']}"
        )
    for _, row in numeric_summary.iterrows():
        narrative.append(
            f"{row['column']}: mean={row['mean']:.2f}, median={row['median']:.2f}, "
            f"zero values={row['zero_pct']:.1f}% -> {row['imputation_note']}"
        )

    return {
        "categorical": categorical_summary,
        "numeric": numeric_summary,
        "narrative": narrative,
    }


def _safe_plot_filename(column: str, plot_type: str) -> str:
    safe_column = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(column)).strip("_")
    return f"{plot_type}_{safe_column}.png"


def route_visualizations(
    df: pd.DataFrame,
    target_column: str,
    figures_path: str,
) -> list[dict]:
    roles = infer_column_roles(df, target_column)
    plan = []
    figures_dir = Path(figures_path)

    for column in roles["numeric"]:
        plan.append({
            "type": "histogram",
            "columns": [column],
            "output_path": str(figures_dir / _safe_plot_filename(column, "hist")),
        })
        plan.append({
            "type": "boxplot",
            "columns": [column],
            "output_path": str(figures_dir / _safe_plot_filename(column, "box")),
        })

    for column in roles["categorical"]:
        plan.append({
            "type": "categorical",
            "columns": [column],
            "output_path": str(figures_dir / _safe_plot_filename(column, "count")),
        })

    if roles["target"] and classify_cardinality(df[target_column]) is not None:
        plan.append({
            "type": "target_distribution",
            "columns": [target_column],
            "output_path": str(
                figures_dir / _safe_plot_filename(target_column, "target")
            ),
        })

    if len(roles["numeric"]) >= 2:
        plan.append({
            "type": "heatmap",
            "columns": roles["numeric"],
            "output_path": str(figures_dir / "correlation_heatmap.png"),
        })

    return plan


def render_visualizations(
    df: pd.DataFrame,
    plan: list[dict],
) -> list[str]:
    from visualizations import (
        plot_categorical,
        plot_heatmap,
        plot_histogram,
        plot_boxplot,
        plot_target_distribution,
    )

    saved_paths = []
    for item in plan:
        plot_type = item["type"]
        columns = item["columns"]
        output_path = item["output_path"]

        if plot_type == "histogram":
            plot_histogram(df, columns[0], output=output_path)
        elif plot_type == "boxplot":
            plot_boxplot(df, columns[0], output=output_path)
        elif plot_type == "categorical":
            plot_categorical(df, columns[0], output=output_path)
        elif plot_type == "target_distribution":
            plot_target_distribution(df, columns[0], output=output_path)
        elif plot_type == "heatmap":
            plot_heatmap(df[columns], output=output_path)
        else:
            raise ValueError(f"Unsupported visualization type: {plot_type}")

        saved_paths.append(output_path)

    return saved_paths
