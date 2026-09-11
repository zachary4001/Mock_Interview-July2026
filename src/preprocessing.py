import re
import pandas as pd


def missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(exclude="number").columns.tolist()


def majority_vote_fill(df, group_col, target_col, fill_value='unknown'):
    df = df.copy()

    def mode_or_fill(series):
        modes = series.mode()
        if len(modes) == 1:
            return modes.iloc[0]
        return fill_value

    group_modes = df.groupby(group_col)[target_col].transform(mode_or_fill)
    df[target_col] = df[target_col].fillna(group_modes)
    df[target_col] = df[target_col].fillna(fill_value)
    return df


def cap_column(df, col, cap_value):
    df = df.copy()
    df[col] = df[col].clip(upper=cap_value)
    return df


def bin_column(df, col, bins, labels, new_col):
    df = df.copy()
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


def extract_model_series(model_str):
    if pd.isna(model_str):
        return 'unknown'

    normalized = str(model_str).replace('-', ' ').lower()
    tokens = normalized.split()
    if not tokens:
        return 'unknown'
    token = tokens[0]

    letter_number_match = re.match(r'^([a-z]+)(\d+)$', token)
    if letter_number_match:
        return letter_number_match.group(1)

    if token.isdigit():
        n = int(token)
        if n <= 9:
            return str(n)
        elif n <= 99:
            return str(round(n / 10) * 10)
        elif n <= 999:
            return str(round(n / 100) * 100)
        else:
            return str(round(n / 1000) * 1000)

    return token