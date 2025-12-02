

# pages/4_Dashboard_Insights.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Resume Analytics Dashboard")

# Load dataset
df = pd.read_csv("Resume_ATS_Fairness.csv")
df["label_str"] = df["y_pred"].map({0: "Weak", 1: "Strong"})

# ---------------------------------------------------------
# TOP KPIs (Dashboard Style)
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

total_resumes = len(df)
strong_resumes = (df["label_str"] == "Strong").sum()
weak_resumes = (df["label_str"] == "Weak").sum()
avg_ats = df["ATS_score"].mean()

col1.metric("📄 Total Resumes", total_resumes)
col2.metric("🌟 Strong Resumes", strong_resumes)
col3.metric("⚠️ Weak Resumes", weak_resumes)
col4.metric("📈 Avg ATS Score", f"{avg_ats:.2f}")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 1 — PLATFORM ANALYTICS ROW
# ---------------------------------------------------------
st.subheader("🧭 Platform Performance Overview")

c1, c2 = st.columns(2)

with c1:
    platform_stats = df.groupby("platform")["y_pred"].mean().reset_index()
    platform_stats["strong_rate"] = platform_stats["y_pred"]

    fig1 = px.bar(
        platform_stats,
        x="platform",
        y="strong_rate",
        title="Strong Resume Rate by Platform",
        color="strong_rate",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("🔍 Two-word insight: **High volume**")

with c2:
    fig2 = px.pie(
        df,
        names="platform",
        title="Resume Distribution by Platform",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔍 Two-word insight: **Naukri leads**")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 2 — SKILL & EDUCATION VISUAL ROW
# ---------------------------------------------------------
st.subheader("🎓 Skills, Education & ATS Behaviour")

left, right = st.columns(2)

with left:
    fig3 = px.box(
        df,
        x="label_str",
        y="skill_count",
        color="label_str",
        title="Skill Count Distribution",
        points="all"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🔍 Two-word insight: **Skills matter**")

with right:
    fig4 = px.box(
        df,
        x="label_str",
        y="education_count",
        color="label_str",
        title="Education Count Distribution",
        points="all"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("🔍 Two-word insight: **Higher education**")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 3 — 3D VISUAL + BUBBLE CHART
# ---------------------------------------------------------
st.subheader("🧊 Advanced 3D & Bubble Insights")

colA, colB = st.columns(2)

with colA:
    fig5 = px.scatter_3d(
        df,
        x="skill_count",
        y="education_count",
        z="ATS_score",
        color="label_str",
        symbol="label_str",
        title="3D Scatter: Skill × Education × ATS",
        height=650
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("🔍 Two-word insight: **Strong cluster**")

with colB:
    fig6 = px.scatter(
        df,
        x="skill_count",
        y="ATS_score",
        size="word_count",
        color="label_str",
        hover_data=["platform", "Category"],
        title="Bubble Chart: Skills vs ATS vs Word Count"
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("🔍 Two-word insight: **Long resumes**")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 4 — CORRELATION + CATEGORY VIOLIN PLOT
# ---------------------------------------------------------
st.subheader("📌 Category & Correlation Trends")

colX, colY = st.columns(2)

with colX:
    features = ["skill_count", "education_count", "certification_count", "word_count", "ATS_score"]
    corr = df[features].corr()

    fig7 = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("🔍 Two-word insight: **Skill–ATS**")

with colY:
    fig8 = px.violin(
        df,
        x="Category",
        y="ATS_score",
        color="label_str",
        box=True,
        title="ATS Score Across Job Categories"
    )
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("🔍 Two-word insight: **IT dominates**")
# ---------------------------------------------------------
# SECTION 5 — ATS SCORE DEEP ANALYSIS
# ---------------------------------------------------------

st.markdown("## 🧠 ATS Score Deep Analysis")

ats_row1, ats_row2 = st.columns(2)

# -------------------------------------
# 1️⃣ ATS Score Distribution Histogram
# -------------------------------------
with ats_row1:
    fig_ats1 = px.histogram(
        df,
        x="ATS_score",
        nbins=30,
        color="label_str",
        title="ATS Score Distribution by Class",
        marginal="box",
        color_discrete_map={"Strong": "green", "Weak": "red"}
    )
    st.plotly_chart(fig_ats1, use_container_width=True)
    st.caption("🔍 Two-word insight: **Score clusters**")

# -------------------------------------
# 2️⃣ ATS Score by Platform
# -------------------------------------
with ats_row2:
    df_platform_ats = df.groupby("platform")["ATS_score"].mean().reset_index()

    fig_ats2 = px.bar(
        df_platform_ats,
        x="platform",
        y="ATS_score",
        color="ATS_score",
        title="Average ATS Score by Platform",
        color_continuous_scale="Plasma"
    )
    st.plotly_chart(fig_ats2, use_container_width=True)
    st.caption("🔍 Two-word insight: **Naukri optimized**")

st.markdown("---")

# -------------------------------------
# 3️⃣ ATS Score by Category (Violin Plot)
# -------------------------------------
st.subheader("🎭 ATS Scores by Job Category")

fig_ats3 = px.violin(
    df,
    x="Category",
    y="ATS_score",
    color="label_str",
    box=True,
    title="ATS Score Spread Across Job Categories",
    color_discrete_map={"Strong": "green", "Weak": "red"}
)
st.plotly_chart(fig_ats3, use_container_width=True)
st.caption("🔍 Two-word insight: **IT leads**")

st.markdown("---")

# -------------------------------------
# 4️⃣ ATS Score vs Word Count (Correlation)
# -------------------------------------
st.subheader("📝 Does Resume Length Affect ATS Score?")

fig_ats4 = px.scatter(
    df,
    x="word_count",
    y="ATS_score",
    color="label_str",
    size="skill_count",
    trendline="ols",
    title="ATS Score vs Word Count (With Trendline)"
)
st.plotly_chart(fig_ats4, use_container_width=True)
st.caption("🔍 Two-word insight: **Length matters**")

st.markdown("---")

# -------------------------------------
# 5️⃣ ATS Score vs Skill Count Heatmap
# -------------------------------------
st.subheader("🔥 Skill Count Heatmap")

df_heat = df.groupby("skill_count")["ATS_score"].mean().reset_index()

fig_ats5 = px.density_heatmap(
    df,
    x="skill_count",
    y="ATS_score",
    nbinsx=20,
    nbinsy=20,
    title="Skill Count vs ATS Score Density Map",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_ats5, use_container_width=True)
st.caption("🔍 Two-word insight: **Skill boost**")

st.markdown("---")

# -------------------------------------
# 6️⃣ ATS Score: Strong vs Weak Comparison
# -------------------------------------
st.subheader("⚖️ ATS Score Comparison (Strong vs Weak)")

ats_compare = df.groupby("label_str")["ATS_score"].mean().reset_index()

fig_ats6 = px.bar(
    ats_compare,
    x="label_str",
    y="ATS_score",
    color="label_str",
    title="Average ATS Score — Strong vs Weak",
    color_discrete_map={"Strong": "green", "Weak": "red"},
    text="ATS_score"
)
fig_ats6.update_traces(texttemplate="%{text:.2f}")
st.plotly_chart(fig_ats6, use_container_width=True)
st.caption("🔍 Two-word insight: **Clear gap**")

st.markdown("---")

# End of file (NO INVALID LINES BELOW)

