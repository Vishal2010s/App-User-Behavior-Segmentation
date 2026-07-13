"""
Data Preprocessing Module

Handles:
- Data Loading
- Dataset Inspection
- Missing Value Treatment
- Outlier Detection
- Outlier Treatment
"""

import numpy as np
import pandas as pd


def load_dataset(filepath):
    """Load dataset."""
    return pd.read_csv(filepath)


def dataset_summary(df):
    """Print dataset summary."""

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Shape : {df.shape}")
    print(f"Duplicates : {df.duplicated().sum()}")

    print("\nMissing Values")

    print(df.isnull().sum().sort_values(ascending=False))


def inspect_categories(df):
    """Inspect categorical columns."""

    for col in df.select_dtypes(include=["object", "category"]):

        print(f"\n{col}")

        print(df[col].value_counts())

        df[col] = df[col].astype("category")

    return df


def impute_rating(df):
    """Fill missing rating values."""

    df["rating_given"] = df["rating_given"].fillna(
        df["rating_given"].median()
    )

    return df


def detect_outliers(df):
    """
    Detect outliers using IQR.
    """

    report = {}

    for col in df.select_dtypes(include=["int64", "float64"]):

        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        report[col] = len(outliers)

    return report


def treat_outliers(df):
    """
    Apply selected outlier treatments.
    """

    outlier_treatments = {

        "avg_session_duration_min": "cap_log",

        "sessions_per_week": "keep",

        "daily_active_minutes": "log",

        "feature_clicks_per_session": "cap",

        "notifications_opened_per_week": "cap",

        "in_app_search_count": "log",

        "ads_clicked_last_30_days": "cap",

        "content_downloads": "log",

        "social_shares": "keep",

        "rating_given": "keep",

        "engagement_score": "keep"

    }

    for column, method in outlier_treatments.items():

        q1 = df[column].quantile(.25)

        q3 = df[column].quantile(.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        if method == "cap":

            df[column] = df[column].clip(lower, upper)

        elif method == "log":

            df[column] = np.log1p(df[column])

        elif method == "cap_log":

            df[column] = df[column].clip(lower, upper)

            df[column] = np.log1p(df[column])

    return df