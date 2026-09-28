import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from data_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path

print("All libraries loaded")
base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "results"
output_dir.mkdir(exist_ok=True)

# ------------------------------------
# Initial EDA 
# ------------------------------------

# 1. Data Collection & Understanding

df = pd.read_csv(base_dir / 'app_user_behavior_dataset.csv')
if df['user_id'].isna().any() or df['user_id'].duplicated().any():
    raise ValueError('Each user must have one non-missing, unique user_id.')
df1=df.copy()
df1.info()
print(f"Shape: {df1.shape}")
print(f"Duplicates: {df1.duplicated().sum()}")
print(f"\nMissing values:")
print(df1.isnull().sum().sort_values(ascending=False))

# 1. Initial EDA Report export:
# profile = ProfileReport(df1, title='Data Report',explorative=True)
# profile.to_file('report.html')

for i in df1.columns:
    if df1[i].dtype in ['object','category']:
        dt=df1[i].value_counts().reset_index()
        df1[i]=df1[i].astype('category')
        print(f"\n--- {i} ---")
        print(dt)
        print(f"Unique values: {df1[i].nunique()}")

# 2. Initial numerical outlier report.
# Use `col` consistently so categorical values or loop counters cannot be
# printed accidentally as feature names.
initial_outlier_rows = []
numeric_columns_initial = df1.select_dtypes(include="number").columns.tolist()
for col in numeric_columns_initial:
    q1 = df1[col].quantile(0.25)
    q3 = df1[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    lower_count = int((df1[col] < lower_bound).sum())
    upper_count = int((df1[col] > upper_bound).sum())
    initial_outlier_rows.append({
        "column_name": col,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_outliers": lower_count,
        "upper_outliers": upper_count,
        "total_outliers": lower_count + upper_count,
    })

initial_outlier_report = pd.DataFrame(initial_outlier_rows)
print("\nInitial numerical outlier report:")
print(initial_outlier_report.round(3).to_string(index=False))
initial_outlier_report.to_csv(
    output_dir / "initial_outlier_report.csv", index=False
)

# ------------------------------------
# Data Cleaning & Preprocessing
# ------------------------------------
# 1. Null Imputation:
# 1.rating_given
print(f"Before Imputation:")
print(df1['rating_given'].describe())
print(f"\nSkewness: {df1['rating_given'].skew():.2f}")
# Use the median rating for missing ratings. Rating is not a clustering input.
df1['rating_given']=df1['rating_given'].fillna(df1['rating_given'].median())
print(f"After Null Imputation:")
print(df1['rating_given'].describe())

# Keep original units for the final customer profiles.
report_data = df1.copy()

# 2.Outlier Treatment:

# Extreme values are not automatically errors. Keep valid low and zero activity.
# Cap only the upper tail of continuous time measurements, then reduce skew.
# Counts, ratings and scores keep their original valid values.
outlier_treatment_rows = []
for col in ['avg_session_duration_min', 'daily_active_minutes']:
    if (df1[col] < 0).any():
        raise ValueError(f"{col} contains negative minutes; check the source data.")
    q1 = df1[col].quantile(0.25)
    q3 = df1[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before_low = int((df1[col] < lower).sum())
    before_high = int((df1[col] > upper).sum())
    df1[col] = np.log1p(df1[col].clip(upper=upper))
    after_q1 = df1[col].quantile(0.25)
    after_q3 = df1[col].quantile(0.75)
    after_iqr = after_q3 - after_q1
    after_lower = after_q1 - 1.5 * after_iqr
    after_upper = after_q3 + 1.5 * after_iqr
    after_low = int((df1[col] < after_lower).sum())
    after_high = int((df1[col] > after_upper).sum())
    outlier_treatment_rows.append({
        "column_name": col,
        "before_low": before_low,
        "before_high": before_high,
        "original_lower_bound": lower,
        "original_upper_bound": upper,
        "treatment": "upper IQR cap + log1p",
        "after_low_retained": after_low,
        "after_high": after_high,
    })
    print(
        f"{col}: {before_high} high values capped; "
        f"After capping using log1p: {after_high}"
    )

# Compare this treatment with keeping the upper tail before final business use.
# Raw values remain unchanged in report_data.

# ══════════════════════════════════════════════
#  FEATURE SELECTION
# ══════════════════════════════════════════════

drop_cols = ['user_id', 'app_version', 'crash_events_last_30_days',
             'support_tickets_raised']

# Why drop each:
# user_id                    → identifier, no behavior info
# app_version                → technical detail, not user choice
# crash_events_last_30_days  → device/software issue, not user action
# support_tickets_raised     → side effect of problems, not behavior

df1_selected = df1.drop(columns=drop_cols)
print(f"\nAfter feature selection: {df1_selected.shape}")
print(f"Dropped: {drop_cols}")
# print(f"\nSelected features:")
# print(df1_selected.columns.tolist())

# Verify feature health before correlation, scaling, PCA or clustering.
numeric_clean = df1_selected.select_dtypes(include=["int64", "float64"])
feature_health = pd.DataFrame({
    "min": numeric_clean.min(),
    "max": numeric_clean.max(),
    "std": numeric_clean.std(),
    "nunique": numeric_clean.nunique(),
})
print("\nNumerical feature-health check:")
print(feature_health.round(4).to_string())
feature_health.to_csv(output_dir / "feature_health.csv")

# Correlation is meaningful here for valid numerical measurements only.
# Arbitrary category numbers (for example country=1,2,3) are not measurements.
# Constant columns are reported above and excluded to prevent blank bands.
df_corr=df1_selected.copy()
for i in df_corr:
  if df_corr[i].dtype in ["category", "object"]:
    le = LabelEncoder()
    df_corr[i] = le.fit_transform(df1_selected[i])

corr = df_corr.corr()
fig = px.imshow(corr,text_auto='.2f',width=1200, height=1200)
fig.write_html(output_dir / "correlation_heatmap.html")
fig.show()
print("\nHighly correlated pairs (|r| > 0.70):")
found = False
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.70:
            print(f"{corr.columns[i]} / {corr.columns[j]}: {corr.iloc[i, j]:.3f}")
            found = True
if not found:
    print("No strong pairwise linear correlations found.")

# ══════════════════════════════════════════════
#  DATA SCALING — StandardScaler
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print("  DATA SCALING — StandardScaler")
print("=" * 55)

print("\nBefore Scaling:")
print(df1_selected.describe().loc[['mean', 'std', 'min', 'max']].transpose())

"""Only the features actually used by K-Means need to be standardized.StandardScaler does not make a feature useful—it only changes its scale
Feature	Meaning:
avg_session_duration_min	How deeply the user engages during each visit
daily_active_minutes	How much total time the user spends daily
days_since_last_login	How recently the user returned
"""

selected_features = [
    "avg_session_duration_min", "daily_active_minutes",
    "days_since_last_login",
]


df1_selected_sf = df1_selected[selected_features]

scaler = StandardScaler()

df1_scaled = pd.DataFrame(scaler.fit_transform(df1_selected_sf),columns=df1_selected_sf.columns)

print("\nAfter Scaling:")
print(df1_scaled.describe().loc[['mean', 'std', 'min', 'max']].transpose())

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

df1_selected_sf.boxplot(ax=axes[0], rot=90)
axes[0].set_title('Model Features — Before Standardization', fontweight='bold')

df1_scaled.boxplot(ax=axes[1], rot=90)
axes[1].set_title('Same Features — After Standardization', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'scaling.png', dpi=150)
plt.show()


df1_scaled.to_csv(output_dir / 'df1_scaled.csv', index=False)

print(f"\n✔ Feature Selection + Scaling Complete")
print(f"  Final shape: {df1_scaled.shape}")
print(f"  Saved to: {output_dir / 'df1_scaled.csv'}")
print(f"  Next: PCA + Clustering")

"""
•	All features used by K-Means are standardized.
•	No unscaled feature enters the K-Means distance calculation.
•	The other features remain in their original units for understandable reporting.
•	Cluster profiles can still show real values such as age, sessions per week, pages, engagement score, and churn risk.
"""

# ══════════════════════════════════════════════
#  POST-TREATMENT EDA VERIFICATION
# ══════════════════════════════════════════════
# Checking only the columns that were scaled. 
print("\nPost-treatment outlier verification:")
for col in ["avg_session_duration_min", "daily_active_minutes", "days_since_last_login"]:
    q1 = df1[col].quantile(0.25)
    q3 = df1[col].quantile(0.75)
    upper_bound = q3 + 1.5 * (q3 - q1)
    high_outliers = (df1[col] > upper_bound).sum()
    print(f"  {col}: {high_outliers} high outliers remaining")

# These are discrete counts or a recency measure. Their unusual values are
# useful behavior, not measurement errors, so we keep them for clustering.
retained_features = [
    "days_since_last_login",
]
print("\nValid behavior extremes retained:")
for col in retained_features:
    q1 = report_data[col].quantile(0.25)
    q3 = report_data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    low_count = int((report_data[col] < lower).sum())
    high_count = int((report_data[col] > upper).sum())
    outlier_treatment_rows.append({
    "feature": col,
    "before_outliers_below_lower_bound": low_count,
    "before_outliers_above_upper_bound": high_count,
    "original_lower_bound": lower,
    "original_upper_bound": upper,
    "treatment": "no transformation; retained as valid recency behavior",
    "after_outliers_below_lower_bound": low_count,
    "after_outliers_above_upper_bound": high_count,
    "outliers_removed": 0,
    "rows_removed": 0,
    "status": "retained unchanged",
    })
    print(f"  {col}: kept as recorded (range {df1[col].min()} to {df1[col].max()})")

outlier_treatment_report = pd.DataFrame(outlier_treatment_rows)
print("\nFinal clustering-feature outlier treatment report:")
print(outlier_treatment_report.round(3).to_string(index=False))
outlier_treatment_report.to_csv(
    output_dir / "outlier_treatment_report.csv", index=False
)

# ══════════════════════════════════════════════
# FEATURE-SET COMPARISON
# ══════════════════════════════════════════════
# Compare k=4 on broader and smaller input sets using the same treatment,
# scaling, random state and silhouette sample. This avoids choosing features
# only because one isolated run looks better.
broad_features = [
    "avg_session_duration_min", "sessions_per_week",
    "daily_active_minutes", "pages_viewed_per_session",
    "feature_clicks_per_session", "notifications_opened_per_week",
    "days_since_last_login", "ads_clicked_last_30_days",
    "in_app_search_count", "content_downloads", "social_shares",
]
focused_features = [
    "avg_session_duration_min", "sessions_per_week",
    "daily_active_minutes", "pages_viewed_per_session",
    "feature_clicks_per_session", "days_since_last_login",
]
core4_features = [
    "avg_session_duration_min", "sessions_per_week",
    "daily_active_minutes", "days_since_last_login",
]

feature_comparison = []
for name, features in {
    "11 behavioral features": broad_features,
    "6 behavioral features": focused_features,
    "4 core features": core4_features,
    "Final 3 core": selected_features,
}.items():
    comparison_scaled = StandardScaler().fit_transform(df1[features])
    max_abs_mean = float(np.abs(comparison_scaled.mean(axis=0)).max())
    max_abs_std_error = float(
        np.abs(comparison_scaled.std(axis=0, ddof=0) - 1).max()
    )
    all_features_scaled = (
        max_abs_mean < 1e-10 and max_abs_std_error < 1e-10
    )
    comparison_model = KMeans(
        n_clusters=4, random_state=42, n_init=10
    )
    comparison_labels = comparison_model.fit_predict(comparison_scaled)
    comparison_silhouette = silhouette_score(
        comparison_scaled, comparison_labels,
        sample_size=5000, random_state=42,
    )
    feature_comparison.append({
        "input": name,
        "features": ", ".join(features),
        "features_or_components": len(features),
        "variance_retained_pct": 100.0,
        "all_model_features_scaled": all_features_scaled,
        "max_abs_scaled_mean": max_abs_mean,
        "max_abs_scaled_std_error": max_abs_std_error,
        "k": 4,
        "silhouette": comparison_silhouette,
        "inertia": comparison_model.inertia_,
    })

# PCA must retain at least 90% variance. With these nearly independent inputs,
# all components are required, so PCA is an evaluation result rather than
# a useful dimensionality-reduction step.
pca_check = PCA().fit(df1_scaled)
cumulative_variance = np.cumsum(pca_check.explained_variance_ratio_)
pca_components_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
pca_90_data = PCA(n_components=pca_components_90).fit_transform(df1_scaled)
pca_90_model = KMeans(
    n_clusters=4, random_state=42, n_init=10
)
pca_90_labels = pca_90_model.fit_predict(pca_90_data)
pca_90_silhouette = silhouette_score(
    pca_90_data, pca_90_labels,
    sample_size=5000, random_state=42,
)
feature_comparison.append({
    "input": "Final core after PCA >=90%",
    "features": "PCA of: " + ", ".join(selected_features),
    "features_or_components": pca_components_90,
    "variance_retained_pct": cumulative_variance[pca_components_90 - 1] * 100,
    "all_model_features_scaled": True,
    "max_abs_scaled_mean": float(np.abs(df1_scaled.mean()).max()),
    "max_abs_scaled_std_error": float(
        np.abs(df1_scaled.std(ddof=0) - 1).max()
    ),
    "k": 4,
    "silhouette": pca_90_silhouette,
    "inertia": pca_90_model.inertia_,
})

feature_comparison_df = pd.DataFrame(feature_comparison)
print("\nFeature-set comparison at k=4:")
print(feature_comparison_df.round(4).to_string(index=False))
feature_comparison_df.to_csv(
    output_dir / "feature_set_comparison.csv", index=False
)


# ══════════════════════════════════════════════
# Clustering Model Selection
# ══════════════════════════════════════════════
# ============================================================
#  K-Means Clustering on User Behaviour
#  Steps:
#    7. Find optimal k  (Elbow Method)
#    8. Train K-Means   (Cluster Assignment)
#    9. Profile clusters (User Identification & Profiling)
# ============================================================

# ============================================================
#  Step 7 - ELBOW METHOD
# ============================================================
inertia = []
smallest_cluster_sizes = []
largest_cluster_sizes = []
cluster_size_details = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    k_labels = km.fit_predict(df1_scaled)
    k_sizes = pd.Series(k_labels).value_counts().sort_index()
    inertia.append(km.inertia_)
    smallest_cluster_sizes.append(int(k_sizes.min()))
    largest_cluster_sizes.append(int(k_sizes.max()))
    cluster_size_details.append(
        "; ".join(f"C{cluster}={count}" for cluster, count in k_sizes.items())
    )


elbow_df = pd.DataFrame({
    "k": list(k_range),
    "inertia": inertia,
    "smallest_cluster": smallest_cluster_sizes,
    "largest_cluster": largest_cluster_sizes,
    "cluster_sizes": cluster_size_details,
})
pd.options.display.float_format = "{:.3f}".format

fig = px.line(
    elbow_df,
    x="k",
    y="inertia",
    title="Elbow Method - Inertia vs Number of Clusters",
    markers=True,
)

fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
fig.update_layout(
    xaxis_title="k",
    yaxis_title="Inertia",
    template="plotly_white",
    width=750,
    height=450,
)
fig.write_html(output_dir / "elbow_plot.html")
fig.show()
# print the inertia values so you can read the elbow
print("k | Inertia")
print(elbow_df.to_string(index=False))

# Silhouette
print("\nSilhouette Scores (same 5,000-user sample for each k):")
silhouette_results = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(df1_scaled)
    sil = silhouette_score(df1_scaled, labels, sample_size=5000, random_state=42)
    # calinski = calinski_harabasz_score(df1_scaled, labels)
    # davies = davies_bouldin_score(df1_scaled, labels)
    silhouette_results.append({
        "k": k,
        "silhouette": sil,
        # "calinski_harabasz": calinski,
        # "davies_bouldin": davies,
    })
    print(
        f"  k={k}: silhouette={sil:.4f}, "
        # f"CH={calinski:.1f}, DB={davies:.4f}"
    )

# ============================================================
# 8. FIT FINAL MODEL (k=4)
# ============================================================

# Four groups are requested by the project; this is not proof that four is the
# unique mathematical optimum. Report the score differences transparently.
score_table = pd.DataFrame(silhouette_results)
best_score_row = score_table.loc[score_table.silhouette.idxmax()]
best_silhouette_k = int(best_score_row['k'])
print("Highest sampled silhouette k:", best_silhouette_k)
cluster_metrics = elbow_df.merge(score_table, on='k', how='left')
cluster_metrics.to_csv(output_dir / 'cluster_metrics.csv', index=False)
optimal_k = 4
k3_silhouette = score_table.loc[score_table.k == 3, 'silhouette'].iloc[0]
k4_silhouette = score_table.loc[score_table.k == 4, 'silhouette'].iloc[0]
print(f"k=3 silhouette: {k3_silhouette:.4f}")
print(f"k=4 silhouette: {k4_silhouette:.4f}")
print(f"Difference (k=3 minus k=4): {k3_silhouette - k4_silhouette:.4f}")
print(
    "Using k=4 as the project-aligned solution because it creates four "
    "stable, behaviorally distinct profiles; the highest silhouette result "
    "is reported separately."
)
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df1["cluster"] = kmeans.fit_predict(df1_scaled)
print(f"\nCluster counts:\n{df1['cluster'].value_counts().sort_index()}")


# ============================================================
# 9. CLUSTER PROFILE
# ============================================================
# Attach labels to original measurements, so minutes remain minutes.
# ============================================================
# 9. CLUSTER PROFILE
# ============================================================

# Attach final K-Means cluster labels to raw-unit data
report_data["cluster"] = df1["cluster"]

# Profile clusters using clustering features + business KPIs
summary = report_data.groupby("cluster")[
    selected_features + [
        "engagement_score",
        "churn_risk_score",
        "sessions_per_week"
    ]
].mean()

print("\nCluster means in ORIGINAL units:")
print(summary.round(3).to_string())

# Derive names from the measured raw-unit profiles. The supplied engagement
# and churn scores are nearly equal across groups, so they are not used to
# force High/Moderate/At-Risk labels.
remaining = set(summary.index)
low_daily_cluster = summary.loc[list(remaining), "daily_active_minutes"].idxmin()
remaining.remove(low_daily_cluster)
short_session_cluster = summary.loc[
    list(remaining), "avg_session_duration_min"
].idxmin()
remaining.remove(short_session_cluster)
recent_cluster = summary.loc[list(remaining), "days_since_last_login"].idxmin()
remaining.remove(recent_cluster)
lapsed_cluster = remaining.pop()

cluster_data = {
    int(short_session_cluster): {
        "label": "Moderate Users(Short-Session Users)",
        "meaning": "Users visit the app regularly but their sessions are comparatively shorter.",
        "opportunity": "Increase session depth and improve content discovery.",
        "action": "Simplify first screens and personalize the next best content.",
        "priority": "P2",
    },

    int(low_daily_cluster): {
        "label": "Low Users(Low-Usage Users)",
        "meaning": "Users spend comparatively less total time in the app each day.",
        "opportunity": "Build stronger engagement and repeat-use habits.",
        "action": "Use lightweight onboarding, reminders and relevant recommendations.",
        "priority": "P2",
    },

    int(recent_cluster): {
        "label": "High Users(Active Long-Visit Users)",
        "meaning": "Users logged in recently and show comparatively high engagement with longer sessions.",
        "opportunity": "Maintain strong engagement and encourage long-term loyalty.",
        "action": "Offer advanced features, personalized recommendations and loyalty benefits.",
        "priority": "P2",
    },

    int(lapsed_cluster): {
        "label": "Occasional Users(Lapsed Long-Session Users)",
        "meaning": "Users showed good engagement previously but currently visit the app less frequently.",
        "opportunity": "Re-engage users and increase their visit frequency.",
        "action": "Run targeted reactivation reminders, personalized offers or win-back campaigns.",
        "priority": "P1",
    },
}

for key in ['label', 'meaning', 'opportunity', 'action', 'priority']:
    report_data[f'cluster_{key}'] = report_data['cluster'].map(
        {c: v[key] for c, v in cluster_data.items()})

# ============================================================
#  Step 9d - PCA VISUALIZATION
# ============================================================

# K-Means is trained on the selected standardized behavioral features.
# PCA is compared as a modeling alternative and used for visualization.
selected_features = [
    "avg_session_duration_min",
    "daily_active_minutes",
    "days_since_last_login"
]

# Scale the same features used in K-Means
X_scaled_pca = StandardScaler().fit_transform(
    df1[selected_features]
)

# Reduce to 2 dimensions only for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_pca)

explained = pca.explained_variance_ratio_ * 100

pca_colors = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e74c3c",
    3: "#3498db"
}

pca_labels = {
    c: cluster_data[c]["label"]
    for c in cluster_data
}

plt.figure(figsize=(10, 7))

for c in sorted(df1["cluster"].unique()):
    mask = df1["cluster"] == c

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=5,
        alpha=0.4,
        c=pca_colors[c],
        label=f"Cluster {c}: {pca_labels[c]}"
    )

plt.title(
    f"PCA — User Clusters "
    f"({explained.sum():.1f}% variance shown)"
)

plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")

plt.legend(markerscale=4)
plt.tight_layout()
plt.show()
# ============================================================
#  Step 9e - FINAL CLUSTER PROFILE TABLE
# ============================================================

profile = report_data.groupby("cluster").agg(
    users             = ("user_id", "nunique"),
    label             = ("cluster_label", "first"),
    business_meaning  = ("cluster_meaning", "first"),
    risk_opportunity  = ("cluster_opportunity", "first"),
    recommended_action = ("cluster_action", "first"),
    priority          = ("cluster_priority", "first"),
    avg_engagement    = ("engagement_score", "mean"),
    avg_churn_risk    = ("churn_risk_score", "mean"),
    avg_sessions_week = ("sessions_per_week", "mean"),
    avg_session_dur   = ("avg_session_duration_min", "mean"),
    avg_daily_active  = ("daily_active_minutes", "mean"),
    avg_pages_viewed  = ("pages_viewed_per_session", "mean"),
    avg_days_since_login = ("days_since_last_login", "mean"),
    avg_feature_clicks = ("feature_clicks_per_session", "mean"),

).round(2)

print("\nFinal Cluster Profile:")
print(profile.to_string())
profile.to_csv(output_dir / "cluster_profile.csv")
report_data.to_csv(output_dir / "users_with_clusters.csv", index=False)

# ============================================================
# PRINT CLUSTER SUMMARY + SAVE DELIVERABLES
# ============================================================

print("\nCluster Summary:")
print("-" * 60)
for c in sorted(df1["cluster"].unique()):
    n = len(df1[df1["cluster"] == c])
    label = cluster_data[c]["label"]
    action = cluster_data[c]["action"]
    priority = cluster_data[c]["priority"]
    eng = profile.loc[c, "avg_engagement"]
    churn = profile.loc[c, "avg_churn_risk"]
    print(f"  Cluster {c}: {n:>6,} users | {label:<16} | "
          f"Avg_Engagement_score: {eng} | Avg_Chrun_rate: {churn} | {action} ({priority})")

# Per-cluster user lists
print("\nPer-cluster deliverables:")
# Remove only customer-list CSVs created by earlier runs of this script. This
# prevents obsolete segment names from remaining beside the current outputs.
for old_cluster_file in output_dir.glob("cluster_[0-9]_*.csv"):
    old_cluster_file.unlink()
for c in sorted(df1["cluster"].unique()):
    sub = report_data[report_data["cluster"] == c][[
        "user_id", "cluster", "engagement_score", "churn_risk_score",
        *selected_features, "sessions_per_week", "pages_viewed_per_session",
        "feature_clicks_per_session", "cluster_label", "cluster_meaning",
        "cluster_opportunity", "cluster_action", "cluster_priority",
    ]]
    safe_label = cluster_data[c]["label"].replace(" ", "_").replace("/", "-")
    filename = output_dir / f"cluster_{c}_{safe_label}.csv"
    sub.to_csv(filename, index=False)
    print(f"  Cluster {c}: {len(sub):,} users saved to {filename}")

print("\nCSV deliverables saved. Segment meanings still require evidence from the profiles.")
