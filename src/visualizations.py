# visualizations.py
# v1.1 - updated 23.08.26

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _save_figure(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_target_distribution(df, target, output='', save_figure: bool = True, show_figure: bool = False):

    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x=target)
    plt.title("Target Distribution")

    if show_figure:
        plt.show()

    if save_figure:
        _save_figure(output)


def plot_histogram(df, column, output='', save_figure: bool = True, show_figure: bool = False):

    plt.figure(figsize=(7,5))
    sns.histplot(df[column], kde=True)
    plt.title(column)

    if show_figure:
        plt.show()

    if save_figure:
        _save_figure(output)


def plot_boxplot(df, column, output='', save_figure: bool = True, show_figure: bool = False):

    plt.figure(figsize=(7,5))
    sns.boxplot(y=df[column])
    plt.title(column)

    if show_figure:
        plt.show()

    if save_figure:
        _save_figure(output)


def plot_heatmap(df, output='', save_figure: bool = True, show_figure: bool = False):

    plt.figure(figsize=(10,8))
    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    if show_figure:
        plt.show()

    if save_figure:
        _save_figure(output)


def plot_categorical(df, column, output='', save_figure: bool = True, show_figure: bool = False):
    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x=column)
    plt.xticks(rotation=30)
    plt.title(column)

    if show_figure:
        plt.show()

    if save_figure:
        _save_figure(output)


def plot_scatter(df, x, y, output='', save_figure: bool = True, show_figure: bool = False):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(data=df, x=x, y=y, alpha=0.3, s=10)

    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)

    if show_figure:
        plt.show()

    if save_figure:
        _save_figure(output)


def plot_grouped_boxplot(df, category_col, value_col, output='', save_figure: bool = True,
                          show_figure: bool = False, caveat: str = None):

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=category_col, y=value_col)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{value_col} by {category_col}")

    if caveat is not None:
        plt.figtext(0.5, 0.01, caveat, ha='center', fontsize=9, color='gray')

    if show_figure:
        plt.show()

    if save_figure:
        _save_figure(output)