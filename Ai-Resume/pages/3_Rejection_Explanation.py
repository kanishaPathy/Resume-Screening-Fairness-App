
# pages/3_Rejection_Explanation.py
# =============================================================
#  Resume Rejection Explanation & Improvement Dashboard
# =============================================================

import streamlit as st
import numpy as np
from io import BytesIO

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try PDF
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


st.set_page_config(page_title="Resume Rejection & Improvement", layout="wide")
st.title("❗ Resume Rejection Explanation & Improvement Plan")


# -------------------------------------------------------------
# 1. CHECK DATA FROM PAGE 2
# -------------------------------------------------------------
required_keys = [
    "shap_vals_class1",
    "feature_names",
    "numeric",
    "category",
    "pred_class"
]

missing = [k for k in required_keys if k not in st.session_state]
if missing:
    st.error(
        f"Missing data from Resume Evaluation page. "
        f"Go back to **Resume Evaluation** and run it first.\nMissing: {missing}"
    )
    st.stop()


# Load session data
shap_vals_class1 = np.array(st.session_state["shap_vals_class1"])
feature_names = st.session_state["feature_names"]
numeric = st.session_state["numeric"]
category = st.session_state["category"]
pred_class = st.session_state["pred_class"]


# Extract features
skill_count = numeric.get("skill_count", 0)
designation_count = numeric.get("designation_count", 0)
education_count = numeric.get("education_count", 0)
certification_count = numeric.get("certification_count", 0)
word_count = numeric.get("word_count", 0)
ats_score = numeric.get("ATS_score", 0)
cat = str(category).upper()


# -------------------------------------------------------------
# LAYOUT: LEFT = TEXT / RIGHT = VISUALS
# -------------------------------------------------------------
left_col, right_col = st.columns([1.15, 1.25])


# =============================================================
# LEFT COLUMN
# =============================================================
with left_col:

    # ------------------ Model Explanation -------------------
    st.markdown("### 🔍 Why the Resume Was Classified as Weak/Strong?")

    explanation_points = []

    if pred_class == 0:
        st.warning("The model predicted this resume as **WEAK**.")
        explanation_points.append(
            "The resume did not align well with strong candidate patterns."
        )
    else:
        st.success("The model predicted this resume as **STRONG**.")
        explanation_points.append(
            "The resume aligns with strong profiles, but some areas still weaken confidence."
        )

    # Worst SHAP features
    try:
        worst_idx = np.argsort(shap_vals_class1)[:3]
        worst_features = [feature_names[i] for i in worst_idx]
    except Exception:
        worst_features = []

    for f in worst_features:
        explanation_points.append(
            f"**{f}** negatively affected the strength score (SHAP)."
        )

    for e in explanation_points:
        st.write("• " + e)

    st.markdown("---")

    # =============================================================
    #  Dynamic Improvement Checklist (CATEGORY-AWARE)
    # =============================================================
    st.markdown("### ✅ Improvement Checklist (Dynamic & Category-Specific)")

    checklist_good = []
    checklist_warn = []

    # CATEGORY-SPECIFIC CERTIFICATION SUGGESTIONS
    CATEGORY_CERTS = {
        "INFORMATION-TECHNOLOGY": [
            "AWS Cloud Practitioner",
            "Azure Fundamentals",
            "Google Data Analytics",
            "Scrum / Agile Certification"
        ],
        "DATA-ANALYTICS": [
            "IBM Data Analyst",
            "Google Data Analytics",
            "Power BI / Tableau Certification"
        ],
        "HR": [
            "SHRM-CP / SCP",
            "HRCI aPHR / PHR",
            "HR Analytics Certification"
        ],
        "FINANCE": [
            "CFA Level 1",
            "CPA",
            "Financial Modelling Certification"
        ],
        "SALES": [
            "Salesforce Certification",
            "HubSpot Sales Certification"
        ],
        "HEALTHCARE": [
            "BLS / ACLS Certification",
            "HIPAA Compliance Certification",
            "Medical Coding / CNA / RBT"
        ]
    }

    # --------------------- SKILLS ---------------------
    if skill_count >= 6:
        checklist_good.append("Skills section looks **strong**.")
    elif skill_count >= 3:
        checklist_warn.append("Add more **role-relevant technical skills**.")
    else:
        checklist_warn.append("Very few skills — add **8–12 strong domain skills**.")

    # --------------------- CERTIFICATIONS ---------------------
    if certification_count >= 2:
        checklist_good.append("Certifications section is **strong**.")
    elif certification_count == 1:
        checklist_warn.append("Add **one more certification** for better credibility.")
    else:
        # Category-aware recommendation
        checklist_warn.append("Add **role-relevant certifications** such as:")
        if cat in CATEGORY_CERTS:
            for cert in CATEGORY_CERTS[cat]:
                checklist_warn.append(f"→ {cert}")
        else:
            checklist_warn.append("→ Add at least one certification relevant to your field.")

    # --------------------- EDUCATION ---------------------
    if education_count > 0:
        checklist_good.append("Education section looks good.")
    else:
        checklist_warn.append("Add your **degree, university, year, specialization**.")

    # --------------------- DESIGNATIONS ---------------------
    if designation_count == 0:
        checklist_warn.append("Add clear **job titles** (Data Analyst, HR Intern, Engineer, etc.).")

    # --------------------- LENGTH ---------------------
    if word_count < 150:
        checklist_warn.append("Resume is **too short** — add achievements & responsibilities.")
    elif word_count > 900:
        checklist_warn.append("Resume is **too long** — shorten older content.")

    # --------------------- ATS SCORE ---------------------
    if ats_score < 50:
        checklist_warn.append("ATS score is low — add **keywords** from the job description.")
    else:
        checklist_good.append("ATS keyword match is **acceptable**.")

    # --------------------- DISPLAY CHECKLIST ---------------------
    if checklist_good:
        st.markdown("#### ✔ Already Strong:")
        for item in checklist_good:
            st.write("🟢 " + item)

    if checklist_warn:
        st.markdown("#### ⚠ Needs Improvement:")
        for item in checklist_warn:
            st.write("🟡 " + item)

    st.markdown("---")

    # Category-specific recommendation box
    st.markdown("### 🧩 Category-Specific Recommendations")

    CATEGORY_TIPS = {
        "INFORMATION-TECHNOLOGY": """
- Add GitHub / GitLab links  
- Highlight programming languages & cloud tools  
- Add measurable project outcomes  
""",
        "DATA-ANALYTICS": """
- Add dashboards (Power BI / Tableau)  
- Show SQL queries & analytics workflow  
- Include ML/statistics exposure  
""",
        "HR": """
- Add HRIS tools (Workday, SAP)  
- Show recruitment pipeline metrics  
- Show onboarding / payroll / compliance tasks  
""",
        "FINANCE": """
- Add financial modelling & forecasting  
- Use measurable impact metrics  
- Mention tools like SAP, ERP, QuickBooks  
""",
        "SALES": """
- Add conversion %, revenue metrics  
- Mention CRM tools (HubSpot, Salesforce)  
""",
        "HEALTHCARE": """
- Add EMR/EHR tools (Epic, Cerner)  
- Mention HIPAA compliance & documentation  
- Highlight patient-care tasks & certifications  
"""
    }

    if cat in CATEGORY_TIPS:
        st.info(CATEGORY_TIPS[cat])
    else:
        st.info("Add measurable achievements, action verbs & relevant tools.")


# =============================================================
# RIGHT COLUMN — VISUALS
# =============================================================
with right_col:

    # ---------------- Radar Chart ----------------
    st.markdown("### 🌐 ATS Profile Radar")

    if PLOTLY_AVAILABLE:
        max_words = 900
        word_pct = min(100, int((word_count / max_words) * 100))
        ats_pct = min(100, ats_score)

        labels = [
            "Skills", "Designations", "Education",
            "Certifications", "Word Count", "ATS Score"
        ]
        values = [
            skill_count * 10,
            designation_count * 10,
            education_count * 20,
            certification_count * 25,
            word_pct,
            ats_pct
        ]

        values.append(values[0])
        labels.append(labels[0])

        fig = go.Figure(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself'
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Install plotly for radar chart.")

    st.markdown("---")

    # ---------------- Strength Gauge ----------------
    st.markdown("### 🎯 Resume Strength Score")

    base_score = ats_score + skill_count*3 + certification_count*5
    if education_count == 0:
        base_score -= 5
    base_score = max(0, min(100, int(base_score)))

    if PLOTLY_AVAILABLE:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=base_score,
            gauge={"axis": {"range": [0, 100]}},
            title={"text": "Resume Strength (0–100)"}
        ))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write(f"Score: {base_score}/100")

    st.markdown("---")

    # ---------------- Progress Bars ----------------
    st.markdown("### 📈 Key ATS Metrics")

    st.write(f"**Skills ({skill_count})**")
    st.progress(min(1.0, skill_count / 8))

    st.write(f"**Certifications ({certification_count})**")
    st.progress(min(1.0, certification_count / 3))

    st.write(f"**ATS Score ({ats_score}/100)**")
    st.progress(min(1.0, ats_score / 100))

    st.write(f"**Word Count ({word_count})**")
    st.progress(min(1.0, word_count / 900))


# =============================================================
# PDF Export
# =============================================================
st.markdown("---")
st.markdown("### 📄 Download ATS Report")

if not FPDF_AVAILABLE:
    st.info("Install `fpdf` to enable PDF download.")
else:
    if st.button("📥 Generate PDF"):

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "ATS Resume Analysis Report", ln=1)

        pdf.multi_cell(0, 7, f"Prediction: {'Strong' if pred_class==1 else 'Weak'}")
        pdf.multi_cell(0, 7, f"Category: {category}")
        pdf.multi_cell(0, 7, f"Strength Score: {base_score}/100")
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Recommended Improvements:", ln=1)
        pdf.set_font("Arial", size=11)

        for i in checklist_warn:
            pdf.multi_cell(0, 6, f"- {i}")

        pdf_bytes = pdf.output(dest="S").encode("latin-1")

        st.download_button(
            "⬇ Download PDF",
            data=pdf_bytes,
            file_name="ATS_Resume_Report.pdf",
            mime="application/pdf"
        )

st.success("🎯 Improvements applied will significantly increase your ATS score & model confidence.")

