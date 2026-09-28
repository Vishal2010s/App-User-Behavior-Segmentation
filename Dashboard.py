from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="App User Behavior Business Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "results"

MODEL_FEATURES = [
    "avg_session_duration_min",
    "daily_active_minutes",
    "days_since_last_login",
]

SEGMENT_COLORS = {
    "High Users(Active Long-Visit Users)": "#2563EB",
    "Low Users(Low-Usage Users)": "#F59E0B",
    "Moderate Users(Short-Session Users)": "#8B5CF6",
    "Occasional Users(Lapsed Long-Session Users)": "#EF4444",
}

DISPLAY_NAMES = {
    "sessions_per_week": "Sessions per week",
    "avg_session_duration_min": "Average session duration",
    "daily_active_minutes": "Daily active minutes",
    "pages_viewed_per_session": "Pages viewed per session",
    "feature_clicks_per_session": "Feature clicks per session",
    "notifications_opened_per_week": "Notifications opened per week",
    "in_app_search_count": "In-app searches",
    "days_since_last_login": "Days since last login",
    "engagement_score": "Engagement score",
    "churn_risk_score": "Churn-risk score",
    "rating_given": "Rating given",
    "content_downloads": "Content downloads",
    "social_shares": "Social shares",
}

STRATEGY_DETAILS = {
    "High Users(Active Long-Visit Users)": {
        "objective": "Protect valuable usage and increase loyalty",
        "actions": [
            "Offer advanced features and personalized recommendations.",
            "Promote premium plans or loyalty benefits to suitable users.",
            "Use this segment for early access and product-feedback programs.",
        ],
        "watch": "Monitor login recency and session duration for early signs of decline.",
    },
    "Low Users(Low-Usage Users)": {
        "objective": "Build a repeat-use habit",
        "actions": [
            "Use a short onboarding journey focused on one clear benefit.",
            "Send relevant reminders at a controlled frequency.",
            "Recommend simple content based on the user's previous activity.",
        ],
        "watch": "Track daily active minutes and whether reminders improve return behavior.",
    },
    "Moderate Users(Short-Session Users)": {
        "objective": "Increase the value delivered early in each visit",
        "actions": [
            "Simplify the first screens and reduce time to useful content.",
            "Personalize the next best action or content recommendation.",
            "Test session-start messages and measure session-duration improvement.",
        ],
        "watch": "Track average session duration without relying only on visit frequency.",
    },
    "Occasional Users(Lapsed Long-Session Users)": {
        "objective": "Reactivate users who have not returned recently",
        "actions": [
            "Run a targeted reactivation or win-back campaign.",
            "Use personalized offers tied to earlier usage patterns.",
            "Limit repeated messages when a user does not respond.",
        ],
        "watch": "Track days since last login, return rate, and post-return session depth.",
    },
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1550px;}
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DFE7F1;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetricLabel"] {color: #475569; font-weight: 600;}
    [data-testid="stMetricValue"] {color: #0F172A;}
    .info-note {
        background: #EFF6FF; border-left: 4px solid #2563EB;
        border-radius: 8px; padding: 12px 16px; color: #1E3A8A;
        margin: 0.2rem 0 1rem 0;
    }
    .warning-note {
        background: #FFF7ED; border-left: 4px solid #F59E0B;
        border-radius: 8px; padding: 12px 16px; color: #7C2D12;
        margin: 0.2rem 0 1rem 0;
    }
    .strategy-title {font-size: 1.08rem; font-weight: 700; color: #0F172A;}
    .priority-p1 {color: #B91C1C; font-weight: 700;}
    .priority-p2 {color: #92400E; font-weight: 700;}
    h1, h2, h3 {color: #0F172A;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    users = pd.read_csv(DATA_DIR / "users_with_clusters.csv")
    profiles = pd.read_csv(DATA_DIR / "cluster_profile.csv")
    metrics = pd.read_csv(DATA_DIR / "cluster_metrics.csv")
    feature_sets = pd.read_csv(DATA_DIR / "feature_set_comparison.csv")
    pca_models = pd.read_csv(DATA_DIR / "pca_modeling_comparison.csv")

    required = {
        "user_id", "cluster", "cluster_label", "cluster_action", "cluster_priority",
        "country", "device_type", "subscription_type", "engagement_score",
        "churn_risk_score", *MODEL_FEATURES,
    }
    missing = required.difference(users.columns)
    if missing:
        raise ValueError(f"users_with_clusters.csv is missing: {sorted(missing)}")
    if users["user_id"].isna().any() or users["user_id"].duplicated().any():
        raise ValueError("user_id must be complete and unique.")
    if len(users) == 0:
        raise ValueError("users_with_clusters.csv contains no records.")
    return users, profiles, metrics, feature_sets, pca_models


@st.cache_data
def pca_coordinates(users):
    model_data = users[MODEL_FEATURES].copy()
    for column in ["avg_session_duration_min", "daily_active_minutes"]:
        q1 = model_data[column].quantile(0.25)
        q3 = model_data[column].quantile(0.75)
        upper = q3 + 1.5 * (q3 - q1)
        model_data[column] = np.log1p(model_data[column].clip(upper=upper))
    scaled = StandardScaler().fit_transform(model_data)
    model = PCA(n_components=2)
    coords = model.fit_transform(scaled)
    result = users[["user_id", "cluster_label"]].copy()
    result["PC1"] = coords[:, 0]
    result["PC2"] = coords[:, 1]
    return result, model.explained_variance_ratio_ * 100


def apply_filters(users):
    st.sidebar.header("Business filters")
    st.sidebar.caption(
        "Business KPIs, exploration charts and downloads follow the selected users. "
        "Model-quality evidence remains global."
    )
    labels = sorted(users["cluster_label"].dropna().unique())
    countries = sorted(users["country"].dropna().unique())
    devices = sorted(users["device_type"].dropna().unique())
    plans = sorted(users["subscription_type"].dropna().unique())

    selected_labels = st.sidebar.multiselect("User segments", labels, default=labels)
    selected_countries = st.sidebar.multiselect("Countries", countries, default=countries)
    selected_devices = st.sidebar.multiselect("Devices", devices, default=devices)
    selected_plans = st.sidebar.multiselect("Subscription types", plans, default=plans)
    minimum_age, maximum_age = int(users["age"].min()), int(users["age"].max())
    age_range = st.sidebar.slider(
        "Age range", minimum_age, maximum_age, (minimum_age, maximum_age)
    )

    mask = (
        users["cluster_label"].isin(selected_labels)
        & users["country"].isin(selected_countries)
        & users["device_type"].isin(selected_devices)
        & users["subscription_type"].isin(selected_plans)
        & users["age"].between(*age_range)
    )
    result = users.loc[mask].copy()
    st.sidebar.divider()
    st.sidebar.caption(f"Selected users: {len(result):,} of {len(users):,}")
    return result


def make_profile(frame):
    return (
        frame.groupby(["cluster", "cluster_label"], as_index=False)
        .agg(
            users=("user_id", "count"),
            avg_engagement=("engagement_score", "mean"),
            avg_churn_risk=("churn_risk_score", "mean"),
            avg_sessions_week=("sessions_per_week", "mean"),
            avg_session_duration=("avg_session_duration_min", "mean"),
            avg_daily_minutes=("daily_active_minutes", "mean"),
            avg_days_since_login=("days_since_last_login", "mean"),
            avg_pages=("pages_viewed_per_session", "mean"),
        )
    )


def explain_relative_score(row, overall, measure):
    definitions = {
        "avg_engagement": ("engagement score", "higher", "lower"),
        "avg_churn_risk": ("churn-risk score", "higher", "lower"),
        "avg_session_duration": ("session duration", "longer", "shorter"),
        "avg_daily_minutes": ("daily usage", "higher", "lower"),
        "avg_days_since_login": ("login gap", "longer", "shorter"),
    }
    name, high_word, low_word = definitions[measure]
    value = row[measure]
    reference = overall[measure]
    difference = value - reference
    if abs(difference) < max(abs(reference) * 0.01, 0.01):
        return f"Its {name} is close to the selected-user average, so the difference is not material."
    direction = high_word if difference > 0 else low_word
    return f"Its {name} is {direction} than the selected-user average by {abs(difference):.2f}."


users, profiles, metrics, feature_sets, pca_models = load_data()
filtered = apply_filters(users)

st.title("App User Behavior Business Dashboard")
st.caption("Understand user behavior, compare segments, and choose practical business actions.")


if filtered.empty:
    st.warning("No users match the current filters. Select at least one value in every filter.")
    st.stop()

profile = make_profile(filtered)
overall = {
    "avg_engagement": filtered["engagement_score"].mean(),
    "avg_churn_risk": filtered["churn_risk_score"].mean(),
    "avg_session_duration": filtered["avg_session_duration_min"].mean(),
    "avg_daily_minutes": filtered["daily_active_minutes"].mean(),
    "avg_days_since_login": filtered["days_since_last_login"].mean(),
}

kpi_columns = st.columns(6)
kpi_columns[0].metric("Users", f"{len(filtered):,}", f"{len(filtered)/len(users):.1%} of all users")
kpi_columns[1].metric("Segments", f"{filtered['cluster_label'].nunique()}")
kpi_columns[2].metric("Sessions per week", f"{filtered['sessions_per_week'].mean():.1f}")
kpi_columns[3].metric("Daily active time", f"{filtered['daily_active_minutes'].mean():.1f} min")
kpi_columns[4].metric("Days since login", f"{filtered['days_since_last_login'].mean():.1f}")
overview_login_gap = pd.to_numeric(filtered["days_since_last_login"], errors="coerce",).dropna()
overview_inactive_pct = (overview_login_gap.ge(30).mean() * 100
    if not overview_login_gap.empty
    else None
)
kpi_columns[5].metric("Inactive users (30+ days)",f"{overview_inactive_pct:.1f}%"
    if overview_inactive_pct is not None
    else "N/A",
)


overview_tab, explorer_tab, strategy_tab, customer_tab, evidence_tab = st.tabs(
    [
        "Executive overview", "Segment explorer", "Strategy center",
        "Customer lookup", "Model evidence",
    ]
)

with overview_tab:
    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Users by segment")
        counts = profile[["cluster_label", "users"]].copy()
        counts["share"] = counts["users"] / counts["users"].sum()
        counts = counts.sort_values("users")
        fig = px.bar(
            counts, x="users", y="cluster_label", orientation="h",
            color="cluster_label", color_discrete_map=SEGMENT_COLORS,
            text=counts["share"].map(lambda value: f"{value:.1%}"),
            labels={"users": "Users", "cluster_label": "Segment"},
        )
        fig.update_traces(textposition="outside", hovertemplate="%{y}<br>Users: %{x:,}<extra></extra>")
        fig.update_layout(showlegend=False, height=390, margin=dict(l=0, r=55, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Behavior comparison")
        behavior = profile.melt(
            id_vars=["cluster", "cluster_label", "users"],
            value_vars=["avg_session_duration", "avg_daily_minutes", "avg_days_since_login"],
            var_name="measure", value_name="average",
        )
        behavior["measure"] = behavior["measure"].map({
            "avg_session_duration": "Session duration (min)",
            "avg_daily_minutes": "Daily active minutes",
            "avg_days_since_login": "Days since last login",
        })
        fig = px.bar(
            behavior, x="measure", y="average", color="cluster_label", barmode="group",
            color_discrete_map=SEGMENT_COLORS,
            labels={"measure": "", "average": "Average", "cluster_label": "Segment"},
        )
        fig.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Where the business opportunity is")

    heading_colors = {
        "High Users(Active Long-Visit Users)": "#60A5FA",        # Blue
        "Low Users(Low-Usage Users)": "#FBBF24",                # Amber
        "Moderate Users(Short-Session Users)": "#C4B5FD",       # Purple
        "Occasional Users(Lapsed Long-Session Users)": "#F87171",  # Red
    }

    opportunity_rows = profiles.loc[
        profiles["label"].isin(profile["cluster_label"])
    ].copy()

    opportunity_rows = opportunity_rows.sort_values(
        ["priority", "users"],
        ascending=[True, False],
    )

    for _, row in opportunity_rows.iterrows():
        with st.container(border=True):
            heading, priority = st.columns([5, 1])

            label = str(row["label"]).strip()
            if label.startswith("High Users"):
                heading_color = "#60A5FA"  # Blue
            elif label.startswith("Low Users"):
                heading_color = "#FBBF24"  # Amber
            elif label.startswith("Moderate Users"):
                heading_color = "#C4B5FD"  # Purple
            elif label.startswith("Occasional Users"):
                heading_color = "#F87171"  # Red
            else:
                heading_color = "#FFFFFF"

            heading.markdown(
                f'<div style="font-size: 1.08rem; font-weight: 700; '
                f'color: {heading_color} !important;">{label}</div>',
                unsafe_allow_html=True,
            )

            priority.markdown(
                f"""
                <span class='priority-{str(row['priority']).lower()}'>
                    {row['priority']} priority
                </span>
                """,
                unsafe_allow_html=True,
            )

            st.write(row["business_meaning"])
            st.write(f"**Opportunity:** {row['risk_opportunity']}")
            st.write(f"**Recommended action:** {row['recommended_action']}")

with explorer_tab:
    st.subheader("Segment explorer")
    selected_segment = st.selectbox("Choose a segment", sorted(filtered["cluster_label"].unique()))
    segment_users = filtered.loc[filtered["cluster_label"] == selected_segment]
    segment_row = profile.loc[profile["cluster_label"] == selected_segment].iloc[0]
    reference = users if len(filtered) == len(users) else filtered

    segment_metrics = st.columns(5)
    segment_metrics[0].metric("Users", f"{len(segment_users):,}", f"{len(segment_users)/len(filtered):.1%} of selected")
    segment_metrics[1].metric("Session duration", f"{segment_users['avg_session_duration_min'].mean():.1f} min")
    segment_metrics[2].metric("Daily activity", f"{segment_users['daily_active_minutes'].mean():.1f} min")
    segment_metrics[3].metric("Login gap", f"{segment_users['days_since_last_login'].mean():.1f} days")

    login_gap = pd.to_numeric(
        segment_users["days_since_last_login"],
        errors="coerce",
    ).dropna()
    inactive_users = int((login_gap >= 30).sum())
    inactive_pct = (
        inactive_users / len(login_gap) * 100
        if len(login_gap) > 0
        else None
    )

    segment_metrics[4].metric(
        "Inactive users (30+ days)",
        f"{inactive_pct:.1f}%" if inactive_pct is not None else "N/A",
        help=(
            f"{inactive_users:,} of {len(login_gap):,} users with known "
            "login recency have not logged in for at least 30 days. "
            "This indicates inactivity, not confirmed churn."
        ),
    )
        # segment_metrics[4].metric("Sessions/week", f"{segment_users['sessions_per_week'].mean():.1f}")

    profile_fields = [
        "sessions_per_week", "avg_session_duration_min", "daily_active_minutes",
        "pages_viewed_per_session", "feature_clicks_per_session", "days_since_last_login",
        "engagement_score", "churn_risk_score",
    ]
    comparison = pd.DataFrame({
        "Measure": [DISPLAY_NAMES[field] for field in profile_fields],
        "Segment": [segment_users[field].mean() for field in profile_fields],
        "All selected users": [reference[field].mean() for field in profile_fields],
    })
    comparison["Index versus selected average"] = comparison["Segment"] / comparison["All selected users"] * 100

    chart_col, reason_col = st.columns([1.4, 1])
    with chart_col:
        fig = px.bar(
            comparison.sort_values("Index versus selected average"),
            x="Index versus selected average", y="Measure", orientation="h",
            color="Index versus selected average", color_continuous_scale="RdYlBu_r",
            color_continuous_midpoint=100,
            labels={"Index versus selected average": "Index (selected average = 100)"},
        )
        fig.add_vline(x=100, line_dash="dash", line_color="#0F172A")
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with reason_col:
        st.markdown("**Why this segment looks high or low**")
        for measure in [
            "avg_session_duration", "avg_daily_minutes", "avg_days_since_login",
            "avg_engagement", "avg_churn_risk",
        ]:
            st.write("• " + explain_relative_score(segment_row, overall, measure))
        matched = profiles.loc[profiles["label"] == selected_segment]
        if not matched.empty:
            item = matched.iloc[0]
            st.info(f"Business meaning: {item['business_meaning']}")
            st.success(f"Recommended action: {item['recommended_action']}")

    st.subheader("Who is in this segment")
    d1, d2, d3 = st.columns(3)
    for column, holder, title in [
        ("subscription_type", d1, "Subscription mix"),
        ("device_type", d2, "Device mix"),
        ("marketing_source", d3, "Marketing-source mix"),
    ]:
        mix = segment_users[column].value_counts(normalize=True).mul(100).reset_index()
        mix.columns = [column, "share"]
        fig = px.bar(mix, x=column, y="share", text_auto=".1f", title=title)
        fig.update_layout(showlegend=False, height=330, yaxis_title="Share (%)", xaxis_title="")
        holder.plotly_chart(fig, width="stretch")

with strategy_tab:
    st.subheader("Business strategy center")
    st.write("Use each strategy as a testable plan with a clear audience, action, and measurement.")
    selected_counts = (
        filtered["cluster_label"].value_counts().rename_axis("label").reset_index(name="selected_users")
    )
    ordered = profiles.loc[profiles["label"].isin(filtered["cluster_label"].unique())].merge(
        selected_counts, on="label", how="inner"
    ).sort_values(
        ["priority", "selected_users"], ascending=[True, False]
    )
    for _, row in ordered.iterrows():
        detail = STRATEGY_DETAILS.get(row["label"], {})
        with st.expander(f"{row['priority']}  {row['label']}  ·  {int(row['selected_users']):,} selected users", expanded=row["priority"] == "P1"):
            st.markdown(f"**Objective:** {detail.get('objective', row['risk_opportunity'])}")
            st.markdown("**Recommended actions**")
            for action in detail.get("actions", [row["recommended_action"]]):
                st.write(f"• {action}")
            st.markdown(f"**What to monitor:** {detail.get('watch', 'Segment size and behavior after the action.')}")

    st.subheader("Campaign planning table")
    campaign = profiles[[
        "label", "users", "priority", "business_meaning", "risk_opportunity", "recommended_action"
    ]].merge(selected_counts, on="label", how="inner").rename(columns={
        "label": "Segment", "users": "All users", "selected_users": "Selected users", "priority": "Priority",
        "business_meaning": "Observed behavior", "risk_opportunity": "Opportunity",
        "recommended_action": "Recommended action",
    })
    st.dataframe(campaign, width="stretch", hide_index=True)

# with drivers_tab:
#     st.subheader("High and low score explanations")
#     st.markdown(
#         '<div class="info-note"><b>Interpret carefully:</b> These relationships are descriptive correlations, '
#         "not proof that one behavior causes engagement or churn.</div>",
#         unsafe_allow_html=True,
#     )

#     candidate_drivers = [
#         "sessions_per_week", "avg_session_duration_min", "daily_active_minutes",
#         "pages_viewed_per_session", "feature_clicks_per_session",
#         "notifications_opened_per_week", "in_app_search_count", "days_since_last_login",
#         "content_downloads", "social_shares", "rating_given",
#     ]
#     corr = filtered[candidate_drivers + ["engagement_score", "churn_risk_score"]].corr()
#     engagement_drivers = corr["engagement_score"].drop(["engagement_score", "churn_risk_score"]).sort_values()
#     churn_drivers = corr["churn_risk_score"].drop(["engagement_score", "churn_risk_score"]).sort_values()

#     e_col, c_col = st.columns(2)
#     with e_col:
#         st.markdown("**Strongest observed relationships with engagement score**")
#         chart = engagement_drivers.reset_index()
#         chart.columns = ["feature", "correlation"]
#         chart["feature"] = chart["feature"].map(DISPLAY_NAMES)
#         fig = px.bar(
#             chart, x="correlation", y="feature", orientation="h",
#             color="correlation", color_continuous_scale="RdBu", color_continuous_midpoint=0,
#             labels={"correlation": "Correlation", "feature": ""},
#         )
#         fig.update_layout(height=470, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
#         st.plotly_chart(fig, width="stretch")
#     with c_col:
#         st.markdown("**Strongest observed relationships with churn-risk score**")
#         chart = churn_drivers.reset_index()
#         chart.columns = ["feature", "correlation"]
#         chart["feature"] = chart["feature"].map(DISPLAY_NAMES)
#         fig = px.bar(
#             chart, x="correlation", y="feature", orientation="h",
#             color="correlation", color_continuous_scale="RdBu", color_continuous_midpoint=0,
#             labels={"correlation": "Correlation", "feature": ""},
#         )
#         fig.update_layout(height=470, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
#         st.plotly_chart(fig, width="stretch")

#     st.subheader("Score distribution by segment")
#     score_sample = filtered.sample(min(12000, len(filtered)), random_state=42)
#     score_name = st.radio("Score", ["engagement_score", "churn_risk_score"], horizontal=True, format_func=lambda item: DISPLAY_NAMES[item])
#     fig = px.box(
#         score_sample, x="cluster_label", y=score_name, color="cluster_label",
#         color_discrete_map=SEGMENT_COLORS, points=False,
#         labels={"cluster_label": "Segment", score_name: DISPLAY_NAMES[score_name]},
#     )
#     fig.update_layout(showlegend=False, height=430)
#     st.plotly_chart(fig, width="stretch")
#     segment_spread = profile["avg_engagement"].max() - profile["avg_engagement"].min()
#     churn_spread = profile["avg_churn_risk"].max() - profile["avg_churn_risk"].min()
#     st.caption(
#         f"Across the current segment profiles, the engagement-average spread is only {segment_spread:.3f} points "
#         f"and the churn-risk-average spread is only {churn_spread:.3f}. Segment labels should therefore be "
#         "explained by behavior rather than by small score differences."
#     )

with customer_tab:
    st.subheader("Customer lookup")
    user_text = st.text_input("Enter a user ID", placeholder="Example: 100000")
    if user_text:
        try:
            user_id = int(user_text)
            match = users.loc[users["user_id"] == user_id]
            if match.empty:
                st.warning("That user ID was not found.")
            else:
                row = match.iloc[0]
                st.success(f"User {user_id:,} belongs to {row['cluster_label']}")
                a, b, c = st.columns(3)
                a.write(f"**Country:** {row['country']}")
                a.write(f"**Device:** {row['device_type']}")
                a.write(f"**Subscription:** {row['subscription_type']}")
                b.write(f"**Sessions/week:** {row['sessions_per_week']:.0f}")
                b.write(f"**Session duration:** {row['avg_session_duration_min']:.2f} min")
                b.write(f"**Daily active time:** {row['daily_active_minutes']:.2f} min")
                c.write(f"**Days since login:** {row['days_since_last_login']:.0f}")
                c.write(f"**Engagement:** {row['engagement_score']:.2f}")
                c.write(f"**Churn risk:** {row['churn_risk_score']:.1%}")
                st.info(f"Recommended action: {row['cluster_action']}")
        except ValueError:
            st.error("Enter a numeric user ID.")

    st.subheader("Selected-customer export")
    export_columns = [
        "user_id", "cluster_label", "cluster_priority", "cluster_action",
        "country", "device_type", "subscription_type", "marketing_source",
        "sessions_per_week", "avg_session_duration_min", "daily_active_minutes",
        "days_since_last_login", "engagement_score", "churn_risk_score",
    ]
    st.dataframe(filtered[export_columns].head(500), width="stretch", hide_index=True)
    st.caption("The table previews 500 rows. The download contains every user matching the filters.")
    st.download_button(
        "Download selected users", filtered[export_columns].to_csv(index=False).encode("utf-8"),
        "selected_app_users.csv", "text/csv", width="stretch",
    )

with evidence_tab:
    st.subheader("Cluster separation")
    coordinates, explained = pca_coordinates(users)
    coordinates = coordinates[coordinates["user_id"].isin(filtered["user_id"])]
    if len(coordinates) > 15000:
        coordinates = coordinates.sample(15000, random_state=42)
    fig = px.scatter(
        coordinates, x="PC1", y="PC2", color="cluster_label",
        color_discrete_map=SEGMENT_COLORS, opacity=0.50, render_mode="webgl",
        hover_data={"user_id": True, "PC1": ":.2f", "PC2": ":.2f"},
        labels={
            "PC1": f"PC1 ({explained[0]:.2f}% variance)",
            "PC2": f"PC2 ({explained[1]:.2f}% variance)",
            "cluster_label": "Segment",
        },
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=590, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"The two PCA axes display {explained.sum():.2f}% of the three-feature variance. "
        "Overlap is expected because a two-dimensional chart does not show all three model dimensions."
    )

    k4 = metrics.loc[metrics["k"] == 4].iloc[0]
    q1, q2, q3 = st.columns(3)
    q1.metric("Selected k", "4")
    q2.metric("Silhouette", f"{k4['silhouette']:.3f}")
    q3.metric("Smallest segment", f"{int(k4['smallest_cluster']):,} users")

    left, right = st.columns(2)
    with left:
        st.markdown("**Elbow and silhouette comparison**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics["k"], y=metrics["inertia"], mode="lines+markers", name="Inertia", yaxis="y1"))
        silhouette_rows = metrics.dropna(subset=["silhouette"])
        fig.add_trace(go.Scatter(x=silhouette_rows["k"], y=silhouette_rows["silhouette"], mode="lines+markers", name="Silhouette", yaxis="y2"))
        fig.add_vline(x=4, line_dash="dash", line_color="#111827")
        fig.update_layout(
            height=430, xaxis_title="Number of clusters k",
            yaxis=dict(title="Inertia"),
            yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12), margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Feature-set evidence**")
        feature_view = feature_sets[["input", "features_or_components", "silhouette"]].copy()
        feature_view = feature_view.rename(columns={
            "input": "Feature set", "features_or_components": "Inputs", "silhouette": "Silhouette"
        })
        feature_view["Silhouette"] = feature_view["Silhouette"].round(4)
        st.dataframe(feature_view, width="stretch", hide_index=True)
        st.caption(
            "The final three-feature input has the strongest k=4 silhouette among the tested feature sets. "
            "The elbow is gradual, so k=4 is supported by combined evidence and interpretability."
        )

