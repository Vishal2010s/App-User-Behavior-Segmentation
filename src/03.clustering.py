"""
Clustering Module

Handles:
- StandardScaler Validation
- Elbow Method
- Silhouette Score
- K-Means Clustering
- PCA
- Cluster Profiling
- Business Labels
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def find_optimal_clusters(df_scaled):
    """
    Determine optimal number of clusters
    using Elbow Method and Silhouette Score.
    """

    inertia = []

    k_range = range(1, 11)

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(df_scaled)

        inertia.append(model.inertia_)

    elbow_df = pd.DataFrame(
        {
            "k": list(k_range),
            "inertia": inertia,
        }
    )

    fig = px.line(
        elbow_df,
        x="k",
        y="inertia",
        markers=True,
        title="Elbow Method"
    )

    fig.show()

    print("\nSilhouette Scores\n")

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(df_scaled)

        score = silhouette_score(
            df_scaled,
            labels,
            sample_size=5000,
            random_state=42,
        )

        print(f"k={k} : {score:.4f}")


def build_clusters(df_original, df_scaled, n_clusters=4):
    """
    Train KMeans model.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    df_original["cluster"] = model.fit_predict(df_scaled)

    return df_original, model


def remap_clusters(df):
    """
    Rename clusters based on business meaning.
    """

    summary = df.groupby("cluster")[
        [
            "engagement_score",
            "churn_risk_score",
            "avg_session_duration_min",
            "sessions_per_week",
        ]
    ].mean()

    order = summary.sort_values(
        "churn_risk_score",
        ascending=False
    ).index.tolist()

    mapping = {

        order[0]: 2,

        order[1]: 3,

        order[2]: 1,

        order[3]: 0,

    }

    df["cluster"] = df["cluster"].map(mapping)

    return df


def assign_business_labels(df):
    """
    Assign business labels.
    """

    cluster_info = {

        0: {
            "label": "High Engagement",
            "priority": "P1",
            "action": "Loyalty Programs",
        },

        1: {
            "label": "Moderate",
            "priority": "P2",
            "action": "Personalized Campaigns",
        },

        2: {
            "label": "Low / At Risk",
            "priority": "P1",
            "action": "Retention Strategy",
        },

        3: {
            "label": "Occasional",
            "priority": "P3",
            "action": "Low Cost Re-engagement",
        },

    }

    df["cluster_label"] = df["cluster"].map(
        lambda x: cluster_info[x]["label"]
    )

    df["cluster_priority"] = df["cluster"].map(
        lambda x: cluster_info[x]["priority"]
    )

    df["cluster_action"] = df["cluster"].map(
        lambda x: cluster_info[x]["action"]
    )

    return df


def perform_pca(df):
    """
    Reduce dimensions for visualization.
    """

    features = [

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

    ]

    X = StandardScaler().fit_transform(df[features])

    pca = PCA(n_components=2)

    components = pca.fit_transform(X)

    return components


def plot_pca(df, components):
    """
    PCA Visualization.
    """

    plt.figure(figsize=(10, 7))

    plt.scatter(

        components[:, 0],

        components[:, 1],

        c=df["cluster"],

        alpha=.5,

        s=8,

    )

    plt.title("PCA Cluster Visualization")

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.show()


def cluster_profile(df):
    """
    Generate cluster profile.
    """

    profile = df.groupby("cluster").agg(

        users=("user_id", "count"),

        engagement=("engagement_score", "mean"),

        churn=("churn_risk_score", "mean"),

        avg_session=("avg_session_duration_min", "mean"),

        sessions=("sessions_per_week", "mean"),

    ).round(2)

    return profile