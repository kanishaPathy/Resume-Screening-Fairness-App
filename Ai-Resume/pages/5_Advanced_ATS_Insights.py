

# pages/5_Advanced_ATS_Insights.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF

st.set_page_config(page_title="Advanced ATS Insights", layout="wide")
st.title("🔥 Advanced ATS Intelligence Dashboard")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("Resume_ATS_Fairness.csv")
df["label_str"] = df["y_pred"].map({0: "Weak", 1: "Strong"})

st.success(f"📄 {len(df)} resumes loaded.")

# ---------------------------------------------------------
# AUTO-DETECT RESUME COLUMN
# ---------------------------------------------------------
resume_cols = [c for c in df.columns if "resume" in c.lower()]
if not resume_cols:
    st.error("❌ Could not find resume text column.")
    st.stop()
resume_col = resume_cols[0]

st.info(f"Using resume column: **{resume_col}**")

# ---------------------------------------------------------
# ADD KEYWORD DENSITY
# ---------------------------------------------------------
KEYWORDS = [
    "python","sql","excel","tableau","power bi","aws",
    "azure","developer","analyst","engineer",
    "machine learning","ml","cloud","data"
]

def keyword_density(x):
    x = str(x).lower()
    return sum(1 for k in KEYWORDS if k in x)

df["keyword_density"] = df[resume_col].apply(keyword_density)

# ---------------------------------------------------------
# DATE SUPPORT (OPTIONAL)
# ---------------------------------------------------------
date_cols = [c for c in df.columns if "date" in c.lower()]
date_col = date_cols[0] if date_cols else None
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
else:
    st.warning("⚠ No date column found – timeline analysis skipped.")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 1 — KEYWORD DENSITY VISUAL
# ---------------------------------------------------------
st.subheader("🔎 Keyword Density Distribution")

fig_kw = px.histogram(
    df, x="keyword_density", color="label_str",
    title="Keyword Density by Resume Strength",
    nbins=20, color_discrete_sequence=px.colors.sequential.Plasma
)
st.plotly_chart(fig_kw, use_container_width=True)

st.write("Example missing keywords:", df.sample(1)["keyword_density"].iloc[0])

st.markdown("---")

# ---------------------------------------------------------
# SECTION 2 — PLATFORM BIAS
# ---------------------------------------------------------
st.subheader("⚖ Platform Bias Detection")

platform_stats = df.groupby("platform")["y_pred"].mean().reset_index()
platform_stats["strong_rate"] = platform_stats["y_pred"]

fig_bias = px.bar(
    platform_stats, x="platform", y="strong_rate",
    title="Strong Resume Rate by Platform",
    color="strong_rate", color_continuous_scale="Teal"
)
st.plotly_chart(fig_bias, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# SECTION 3 — ATS SCORE ANALYSIS
# ---------------------------------------------------------
st.subheader("📊 ATS Score by Strong/Weak")

fig_ats = px.violin(
    df, x="label_str", y="ATS_score", color="label_str",
    box=True, title="ATS Score Distribution"
)
st.plotly_chart(fig_ats, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# SECTION 4 — TIMELINE (IF AVAILABLE)
# ---------------------------------------------------------
if date_col:
    st.subheader("⏳ ATS Score Timeline")

    df_t = df.dropna(subset=[date_col]).sort_values(date_col)
    fig_time = px.line(
        df_t, x=date_col, y="ATS_score", color="label_str",
        title="ATS Score Trend Over Time"
    )
    st.plotly_chart(fig_time, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# SECTION 5 — AI SUMMARY GENERATION
# ---------------------------------------------------------
st.subheader("🧠 AI-Generated Summary of Dataset")

def generate_ai_summary(df):
    """Simple rule-based AI summary generator (no API needed)."""
    strong_rate = round(df['y_pred'].mean() * 100, 2)
    avg_ats = round(df["ATS_score"].mean(), 2)
    most_common_platform = df["platform"].value_counts().idxmax()
    hardest_category = df.groupby("Category")["y_pred"].mean().idxmin()

    return (
        f"📌 **AI Summary:**\n"
        f"- Strong resume ratio: **{strong_rate}%**\n"
        f"- Avg ATS Score: **{avg_ats}**\n"
        f"- Most resumes come from: **{most_common_platform}**\n"
        f"- Most rejected category: **{hardest_category}**\n"
        f"- Tip: Improve keyword density & certifications to boost strong predictions.\n"
    )

ai_summary = generate_ai_summary(df)
st.info(ai_summary)

st.markdown("---")

# ---------------------------------------------------------
# SECTION 6 — EXPORT PDF REPORT
# ---------------------------------------------------------
st.subheader("📄 Export Full ATS Report as PDF")

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "ATS Intelligence Report", ln=1, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Generated via ATS Dashboard", align="C")

def create_pdf(df, ai_summary):
    pdf = PDF()
    pdf.add_page()

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, ai_summary)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Dataset Overview", ln=1)

    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, str(df.describe()))

    return pdf.output(dest="S").encode("latin1")

if st.button("📥 Download ATS PDF Report"):
    pdf_data = create_pdf(df, ai_summary)
    st.download_button(
        label="Download PDF",
        data=pdf_data,
        file_name="ATS_Report.pdf",
        mime="application/pdf"
    )

st.success("✅ Report ready!")


