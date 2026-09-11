import pandas as pd


def dataset_summary(df: pd.DataFrame):

    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing values": int(df.isna().sum().sum()),
        "Duplicates": int(df.duplicated().sum())
    }


def descriptive_statistics(df: pd.DataFrame):

    return df.describe()


def correlation_matrix(df: pd.DataFrame):

    return df.select_dtypes(include="number").corr()


def median_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)[value_col]
        .median()
        .reset_index()
        .rename(columns={value_col: f"median_{value_col}"})
        .sort_values(f"median_{value_col}", ascending=False)
    )