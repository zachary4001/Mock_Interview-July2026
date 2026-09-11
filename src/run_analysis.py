# run_analysis.py
# v1.1 -updated 23.08.26

from pathlib import Path

from src.config import TRAIN_FILE, FIGURES_DIR, REPORTS_DIR
from src.data_config import (
    HISTOGRAM_COLUMNS,
    BOXPLOT_COLUMNS,
    CATEGORICAL_PLOT_COLUMNS,
    HEATMAP,
    SCATTER_PLOTS,
    GROUPED_BOXPLOT_PLOTS,
    MEDIAN_BY_GROUP_EXPORTS,
    NOISY_FEATURES,
)
from src.data_loader import load_data
from src.eda_statistics import dataset_summary, descriptive_statistics, correlation_matrix, median_by_group
from src.preprocessing import missing_values
from src.visualizations import (
    plot_heatmap,
    plot_histogram,
    plot_boxplot,
    plot_categorical,
    plot_scatter,
    plot_grouped_boxplot,
)
from src.utils import create_output_folders


def main(preprocess_fn=None):
    create_output_folders(FIGURES_DIR, REPORTS_DIR)

    df = load_data(Path(TRAIN_FILE))

    if preprocess_fn is not None:
        df = preprocess_fn(df)

    print(dataset_summary(df))
    print(missing_values(df))
    print(descriptive_statistics(df))

    for column in HISTOGRAM_COLUMNS:
        plot_histogram(df, column, output=Path(FIGURES_DIR) / f"histogram_{column}.png")

    for column in BOXPLOT_COLUMNS:
        plot_boxplot(df, column, output=Path(FIGURES_DIR) / f"boxplot_{column}.png")

    for column in CATEGORICAL_PLOT_COLUMNS:
        plot_categorical(df, column, output=Path(FIGURES_DIR) / f"categorical_{column}.png")

    if HEATMAP:
        plot_heatmap(df, output=Path(FIGURES_DIR) / "heatmap.png")

    # EDA-04 — scatter plots
    for x, y in SCATTER_PLOTS:
        plot_scatter(df, x, y, output=Path(FIGURES_DIR) / f"scatter_{x}_vs_{y}.png")

    # EDA-03/05 — grouped boxplots
    for category_col, value_col, caveat in GROUPED_BOXPLOT_PLOTS:
        plot_grouped_boxplot(
            df, category_col, value_col,
            output=Path(FIGURES_DIR) / f"grouped_boxplot_{category_col}.png",
            caveat=caveat,
        )

    # Reports — median by group CSVs
    for group_col, value_col in MEDIAN_BY_GROUP_EXPORTS:
        result = median_by_group(df, group_col, value_col)
        result.to_csv(
            Path(REPORTS_DIR) / f"median_{value_col}_by_{group_col}.csv",
            index=False,
        )

    # Reports — summary stats and missing values
    import pandas as pd
    summary = pd.DataFrame([dataset_summary(df)])
    summary.to_csv(Path(REPORTS_DIR) / "dataset_summary.csv", index=False)

    mv = missing_values(df).reset_index()
    mv.columns = ["column", "missing_count"]
    mv.to_csv(Path(REPORTS_DIR) / "missing_values.csv", index=False)


if __name__ == "__main__":
    main()
