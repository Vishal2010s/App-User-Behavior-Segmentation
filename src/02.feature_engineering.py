"""
Feature Engineering Module

Handles:
- Feature Selection
- Label Encoding
- Correlation Analysis
- Standard Scaling
"""

import pandas as pd
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def select_features(df):
    """
    Remove non-behavioral columns.
    """

    drop_cols = [

        "user_id",

        "app_version",

        "crash_events_last_30_days",

        "support_tickets_raised",

    ]

    df_selected = df.drop(columns=drop_cols)

    print(f"Selected Shape : {df_selected.shape}")

    return df_selected


def encode_features(df):
    """
    Encode categorical variables.
    """

    encoded = df.copy()

    for col in encoded.select_dtypes(include=["object", "category"]):

        encoder = LabelEncoder()

        encoded[col] = encoder.fit_transform(encoded[col])

    return encoded


def correlation_analysis(df):
    """
    Plot correlation matrix.
    """

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        width=900,
        height=900,
        title="Feature Correlation Matrix"
    )

    fig.show()

    print("\nHighly Correlated Features (|r| > 0.70)\n")

    found = False

    for i in range(len(corr.columns)):

        for j in range(i + 1, len(corr.columns)):

            if abs(corr.iloc[i, j]) > 0.70:

                found = True

                print(
                    f"{corr.columns[i]} ↔ "
                    f"{corr.columns[j]} : "
                    f"{corr.iloc[i, j]:.2f}"
                )

    if not found:

        print("No highly correlated features found.")


def scale_features(df):
    """
    Scale numerical features using StandardScaler.
    """

    selected_features = [

        "avg_session_duration_min",

        "sessions_per_week",

        "daily_active_minutes",

        "pages_viewed_per_session",

        "feature_clicks_per_session",

        "notifications_opened_per_week",

        "days_since_last_login",

        "account_age_days",

        "age",

        "ads_clicked_last_30_days",

        "in_app_search_count",

        "content_downloads",

        "churn_risk_score",

        "engagement_score",

    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(df[selected_features])

    scaled_df = pd.DataFrame(

        scaled,

        columns=selected_features,

    )

    return scaled_df, scaler