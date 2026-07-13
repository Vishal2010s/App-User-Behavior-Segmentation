"""
Visualization Module
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def plot_histogram(df, column):

    plt.figure(figsize=(8,5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()


def plot_boxplot(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"Outlier Detection - {column}"
    )

    fig.show()


def correlation_heatmap(df):

    plt.figure(figsize=(14,10))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=False,
        cmap="coolwarm"
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.show()


def plot_cluster_distribution(df):

    plt.figure(figsize=(7,5))

    sns.countplot(
        x="cluster",
        data=df
    )

    plt.title("Cluster Distribution")

    plt.show()