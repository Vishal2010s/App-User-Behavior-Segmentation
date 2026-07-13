# %%
import pandas as pd
from sqlalchemy import create_engine
import  plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
# from ydata_profiling import ProfileReport
from data_profiling import ProfileReport
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pdfkit
import os

print("\N{white heavy check mark}","All libraries loaded")

# %%
df=pd.read_csv('app_user_behavior_dataset.csv')
df1=df.copy()
print(f"Shape: {df1.shape}")
print(f"Duplicates: {df1.duplicated().sum()}")
print(f"\nMissing values:")
print(df1.isnull().sum().sort_values(ascending=False))

# %%
# profile = ProfileReport(df1, title='Data Report',explorative=True)
# profile.to_file('report.html')

# %%
df1.info()

# %%
for i in df1.columns:
    if df1[i].dtype in ['object','category']:
        dt=df1[i].value_counts().reset_index()
        df1[i]=df1[i].astype('category')
        print(f"\n--- {i} ---")
        print(dt)

# %%
df1.describe()

# %%
# 1.rating_given
print(f"Before Imputation:")
print(df1['rating_given'].describe())
print(f"\nSkewness: {df1['rating_given'].skew():.2f}")
# As skewness (-0.68) is less than 1, the values are slightly left skewed and as the 50%-75% vales are btween 4-5, hence taking median to fill values
df1['rating_given']=df1['rating_given'].fillna(df1['rating_given'].median())
print(f"After Null Imputation:")
df1['rating_given'].describe()

# %%
df1.isna().sum()

# %%
for i in df1.columns:
    if df1[i].dtype in ["category", "object"]:
        print(f"\n{i}:")
        print(df1[i].value_counts())
        print(f"Unique values: {df[i].nunique()}")

# %%
for i in df1.columns:
    if df1[i].dtype in ['float64','int64']:
        Q1 = df1[i].quantile(0.25)
        Q3 = df1[i].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df1[(df1[i] < lower_bound) | (df1[i] > upper_bound)]

        # df1[i].plot.hist(bins=30, title=f'Distribution of {i}')
        # plt.xlabel(i)
        # plt.ylabel('Frequency')
        # plt.show()
        #  # Visualization: Boxplot with custom dimensions
        # print(f"Visualization for outlier detection in column: {i}")
        # fig = px.box(df1, y=i, title=f"{i} Outlier Detection", width=900, height=600)

        # fig.add_hline(y=lower_bound, line_color="red", line_dash="dash", annotation_text=f"Lower Bound: {lower_bound:.2f}", annotation_position="top left")
        # fig.add_hline(y=upper_bound, line_color="green", line_dash="dash", annotation_text=f"Upper Bound: {upper_bound:.2f}", annotation_position="top right")
        # fig.show()
        # print(f"____________________________________________________________________________________________________________________________________________________\n")

        if outliers.shape[0] > 0:          
            print(f"Column name: {i}")
            print(f"Before: {outliers.shape[0]} outliers found")
        else:
            print(f"Column name: ***{i}***: No outliers")
        print("-" * 60)

# %%
# normalization:

outlier_treatments = {
    'avg_session_duration_min':      'cap_log',
    'sessions_per_week':             'keep',
    'daily_active_minutes':          'log',
    'feature_clicks_per_session':    'cap',
    'notifications_opened_per_week': 'cap',
    'in_app_search_count':           'log',
    'ads_clicked_last_30_days':      'cap',
    'content_downloads':             'log',
    'social_shares':                 'keep',
    'rating_given':                  'keep',
    'engagement_score':              'keep',
}

for i, j in outlier_treatments.items():

    # Count outliers before
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = ((df1[i] < lower) | (df1[i] > upper)).sum()

    # Apply treatment
    if j == 'keep':
        pass

    elif j == 'cap':
        df1[i] = df1[i].clip(lower, upper)

    elif j == 'log':
        df1[i] = np.log1p(df1[i])
        Q1 = df1[i].quantile(0.25)
        Q3 = df1[i].quantile(0.75)
        IQR = Q3 - Q1
        df1[i] = df1[i].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)   


    elif j == 'cap_log':
        df1[i] = df1[i].clip(lower, upper)
        df1[i] = np.log1p(df1[i])
        Q1 = df1[i].quantile(0.25)
        Q3 = df1[i].quantile(0.75)
        IQR = Q3 - Q1
        df1[i] = df1[i].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Count outliers after
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    after = ((df1[i] < lower) | (df1[i] > upper)).sum()

    pct = f"{(before - after) / before * 100:.1f}%" if before > 0 else "N/A"
    print(f"{i:<32} {j:<10} Before: {before:<6} After: {after:<6} {pct}")

# %%
# df1.to_csv('app_interim.csv', index=False)

# %%
df1.describe()

# %%
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
print(f"\nSelected features:")
print(df1_selected.columns.tolist())

# Encoding
df_corr=df1_selected.copy()
for i in df_corr:
  if df_corr[i].dtype in ["category", "object"]:
    le = LabelEncoder()
    df_corr[i] = le.fit_transform(df1_selected[i])

corr=df_corr.corr()
fig = px.imshow(corr,text_auto='.2f',width=1200, height=1200)
fig.show()

print("\nHighly correlated pairs (|r| > 0.70):")
found = False
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.70:
            print(f"  {corr.columns[i]} ↔ {corr.columns[j]}: {corr.iloc[i, j]:.3f}")
            found = True
if not found:
    print("  None found. All selected features are independent.")


# %%
# ══════════════════════════════════════════════
#  DATA SCALING — StandardScaler
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print("  DATA SCALING — StandardScaler")
print("=" * 55)

print("\nBefore Scaling:")
print(df1_selected.describe().loc[['mean', 'std', 'min', 'max']].transpose())

selected_features = [
    "avg_session_duration_min", "sessions_per_week",
    "daily_active_minutes", "pages_viewed_per_session",
    "feature_clicks_per_session", "notifications_opened_per_week",
    "days_since_last_login", "account_age_days",
    "age", "ads_clicked_last_30_days",
    "in_app_search_count", "content_downloads",
    "churn_risk_score", "engagement_score",
]

df1_selected_sf = df1_selected[selected_features]

scaler = StandardScaler()

df1_scaled = pd.DataFrame(scaler.fit_transform(df1_selected_sf),columns=df1_selected_sf.columns)

print("\nAfter Scaling:")
print(df1_scaled.describe().loc[['mean', 'std', 'min', 'max']].transpose())

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

df1_selected.boxplot(ax=axes[0], rot=90)
axes[0].set_title('Before Scaling — Different Ranges', fontweight='bold')

df1_scaled.boxplot(ax=axes[1], rot=90)
axes[1].set_title('After Scaling — All Normalized', fontweight='bold')

plt.tight_layout()

plt.show()


# # ══════════════════════════════════════════════
# #  SAVE — Ready for next stage
# # ══════════════════════════════════════════════

# df1_scaled.to_csv('data/df1_scaled.csv', index=False)

# print(f"\n✔ Feature Selection + Scaling Complete")
# print(f"  Final shape: {df1_scaled.shape}")
# print(f"  Saved to: data/df1_scaled.csv")
# print(f"  Next: PCA + Clustering")

# %%
df1_selected.describe()


# %%
# Reruningoutlier detection after dropping columns

for i in df1_selected.columns:
    if df1_selected[i].dtype in ['float64','int64']:
        Q1 = df1_selected[i].quantile(0.25)
        Q3 = df1_selected[i].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df1_selected[(df1_selected[i] < lower_bound) | (df1_selected[i] > upper_bound)]

        # df1_selected[i].plot.hist(bins=30, title=f'Distribution of {i}')
        # plt.xlabel(i)
        # plt.ylabel('Frequency')
        # plt.show()
        #  # Visualization: Boxplot with custom dimensions
        # print(f"Visualization for outlier detection in column: {i}")
        # fig = px.box(df1_selected, y=i, title=f"{i} Outlier Detection", width=900, height=600)

        # fig.add_hline(y=lower_bound, line_color="red", line_dash="dash", annotation_text=f"Lower Bound: {lower_bound:.2f}", annotation_position="top left")
        # fig.add_hline(y=upper_bound, line_color="green", line_dash="dash", annotation_text=f"Upper Bound: {upper_bound:.2f}", annotation_position="top right")
        # fig.show()
        # print(f"____________________________________________________________________________________________________________________________________________________\n")

        if outliers.shape[0] > 0:          
            print(f"Column name: {i}")
            print(f"Before: {outliers.shape[0]} outliers found")
        else:
            print(f"Column name: ***{i}***: No outliers")
        print("-" * 60)

   

# %%
print(df1.columns)
print("-"*70)
print(df1_scaled.columns)

# %%
df1_scaled_stats = df1_scaled.describe().loc[['mean', 'std', 'min', 'max']].T
df1_scaled_stats['mean_rounded'] = df1_scaled_stats['mean'].round(12)
df1_scaled_stats['std_rounded'] = df1_scaled_stats['std'].round(4)
print(df1_scaled_stats[['mean_rounded', 'std_rounded', 'min', 'max']])


# %%
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
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df1_scaled)
    inertia.append(km.inertia_)


elbow_df = pd.DataFrame({"k": list(k_range), "inertia": inertia})
pd.options.display.float_format = "{:,.0f}".format

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
fig.show()
# print the inertia values so you can read the elbow
print("k | Inertia")
print(elbow_df.to_string(index=False))

# Silhouette
print("\nSilhouette Scores:")
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(df1_scaled)
    sil = silhouette_score(df1_scaled, labels, sample_size=5000, random_state=42)
    print(f"  k={k}: {sil:.4f}")

# ============================================================
# 8. FIT FINAL MODEL (k=4)
# ============================================================

optimal_k = 4                      # change this after looking at the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df1["cluster"] = kmeans.fit_predict(df1_scaled)
print(f"\nCluster counts:\n{df1['cluster'].value_counts().sort_index()}")

# %%
# ============================================================
# 9. CLUSTER PROFILE
# ============================================================
summary = df1.groupby("cluster")[
    ["engagement_score", "churn_risk_score",
     "avg_session_duration_min", "sessions_per_week"]
].mean().round(2)

print("\nOriginal cluster means:")
print(summary)

# ============================================================
#  Step 9 - REMAP CLUSTERS TO MATCH NARRATIVE
#     Highest churn → 2 (At-Risk)
#     Lowest churn  → 0 (High Engagement)
# ============================================================

sorted_clusters = summary.sort_values(
    "churn_risk_score", ascending=False
).index.tolist()

remap = {
    sorted_clusters[0]: 2,   # highest churn  → Low Engagement / At-Risk
    sorted_clusters[1]: 3,   # second highest → Occasional
    sorted_clusters[2]: 1,   # second lowest  → Moderate
    sorted_clusters[3]: 0,   # lowest churn   → High Engagement
}

print(f"\nRemap (old → new): {remap}")
df1["cluster"] = df1["cluster"].map(remap).astype(int)

# Verify remapped means
print("\nRemapped cluster means:")
print(df1.groupby("cluster")[
    ["engagement_score", "churn_risk_score",
     "avg_session_duration_min", "sessions_per_week"]
].mean().round(2))

# ============================================================
#   BUILD CLUSTER DATA (label, action, priority)
#============================================================

cluster_data = {
    0: {"label": "High Engagement", "action": "Loyalty and Premium Offers",  "priority": "P1"},
    1: {"label": "Moderate",        "action": "Personalized Engagement",   "priority": "P2"},
    2: {"label": "Low / At-Risk",   "action": "Retention and Re-engagement", "priority": "P1"},
    3: {"label": "Occasional",      "action": "	Minimal Outreach and Low-effort Re-activation ",       "priority": "P3"},
}

for key in ["label", "action", "priority"]:
    df1[f"cluster_{key}"] = df1["cluster"].map({c: v[key] for c, v in cluster_data.items()})




# %%
# ============================================================
#  Step 9d - PCA VISUALIZATION
# ============================================================

selected_features = [
    "avg_session_duration_min", "sessions_per_week",
    "daily_active_minutes", "pages_viewed_per_session",
    "feature_clicks_per_session", "notifications_opened_per_week",
    "days_since_last_login", "account_age_days",
    "age", "ads_clicked_last_30_days",
    "in_app_search_count", "content_downloads",
    "churn_risk_score",
]

X_scaled_pca = StandardScaler().fit_transform(df1[selected_features])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_pca)

pca_colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c", 3: "#3498db"}
pca_labels = {0: "High", 1: "Moderate", 2: "Low/At-Risk", 3: "Occasional"}

plt.figure(figsize=(10, 7))
for c in sorted(df1["cluster"].unique()):
    mask = df1["cluster"] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                s=5, alpha=0.4, c=pca_colors[c],
                label=f"Cluster {c}: {pca_labels[c]}")

plt.title(f"PCA — User Clusters (all {len(df1):,} users)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(markerscale=4)
plt.tight_layout()
plt.show()


# ============================================================
#  Step 9e - FINAL CLUSTER PROFILE TABLE
# ============================================================

profile = df1.groupby("cluster").agg(
    users             = ("user_id", "nunique"),
    label             = ("cluster_label", "first"),
    avg_engagement    = ("engagement_score", "mean"),
    avg_churn_risk    = ("churn_risk_score", "mean"),
    avg_sessions_week = ("sessions_per_week", "mean"),
    avg_session_dur   = ("avg_session_duration_min", "mean"),
    avg_daily_active  = ("daily_active_minutes", "mean"),
    avg_pages_viewed  = ("pages_viewed_per_session", "mean"),
).round(2)

print("\nFinal Cluster Profile:")
print(profile)


# %%
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
          f"Eng: {eng} | Churn: {churn} | {action} ({priority})")

# Per-cluster user lists
print("\nPer-cluster deliverables:")
for c in sorted(df1["cluster"].unique()):
    sub = df1[df1["cluster"] == c][[
        "user_id", "engagement_score", "churn_risk_score",
        "cluster_label", "cluster_action",
    ]]
    safe_label = cluster_data[c]["label"].replace(" ", "_").replace("/", "-")
    print(f"  Cluster {c}: {len(sub):,} users → cluster_{c}_{safe_label}.csv")

print("\nAll deliverables ready.")


