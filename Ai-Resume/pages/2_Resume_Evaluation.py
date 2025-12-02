

# pages/2_Resume_Evaluation.py

# pages/2_Resume_Evaluation.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
import shap

import pdfplumber
import docx2txt
from PyPDF2 import PdfReader

# ---------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Resume Evaluation", layout="wide")
st.title("📝 Resume Evaluation with ATS + SHAP Explainability")

BASE_DIR = r"C:\Users\Kanisha Pathy\Downloads\Research - Practicum"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMERIC_FEATURES = [
    "skill_count",
    "designation_count",
    "education_count",
    "certification_count",
    "word_count",
    "ATS_score"
]

CATEGORICAL_FEATURES = ["platform", "Category"]


# ---------------------------------------------------------
# TABTRANSFORMER MODEL DEFINITION
# ---------------------------------------------------------
class TabTransformer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities,
                 d_model=32, n_heads=4, n_layers=2,
                 dropout=0.1, num_classes=2):
        super().__init__()

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_model) for card in cat_cardinalities
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_num, x_cat):
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embs = torch.stack(cat_embs, dim=1)

        cat_ctx = self.transformer(cat_embs)
        cat_pooled = cat_ctx.mean(dim=1)

        num_repr = self.numeric_proj(x_num)

        h = torch.cat([cat_pooled, num_repr], dim=1)
        logits = self.fc(h)
        return logits


# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(BASE_DIR + r"\TabTransformer_Scaler.pkl")
    cat_maps = joblib.load(BASE_DIR + r"\TabTransformer_Cat_Maps.pkl")

    with open(BASE_DIR + r"\TabTransformer_Hyperparams.json", "r") as f:
        best_params = json.load(f)

    cat_cardinalities = [len(cat_maps[col]) for col in CATEGORICAL_FEATURES]

    model = TabTransformer(
        num_numeric=len(NUMERIC_FEATURES),
        cat_cardinalities=cat_cardinalities,
        d_model=best_params["d_model"],
        n_heads=best_params["n_heads"],
        n_layers=best_params["n_layers"],
        dropout=best_params["dropout"],
        num_classes=2
    ).to(DEVICE)

    state = torch.load(
        BASE_DIR + r"\TabTransformer_Final_Model.pt",
        map_location=DEVICE
    )
    model.load_state_dict(state)
    model.eval()

    df = pd.read_csv(BASE_DIR + r"\Resume_ATS_Final.csv")

    if "word_count" not in df.columns:
        df["word_count"] = df["clean_resume"].astype(str).apply(lambda x: len(x.split()))

    if "auto_selected" in df.columns:
        df_model = df.dropna(subset=["auto_selected"]).copy()
    else:
        df_model = df.copy()

    return model, scaler, cat_maps, cat_cardinalities, df_model


model, scaler, cat_maps, cat_cardinalities, df_model = load_artifacts()


# ---------------------------------------------------------
# FILE TEXT EXTRACTION
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    mime = uploaded_file.type.lower()

    # PDF
    if name.endswith(".pdf") or "pdf" in mime:
        text = ""
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages)
        except:
            text = ""

        if not text.strip():
            try:
                uploaded_file.seek(0)
                reader = PdfReader(uploaded_file)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
            except:
                text = ""

        return text

    # DOCX
    if name.endswith(".docx") or "word" in mime or "officedocument" in mime:
        try:
            return docx2txt.process(uploaded_file)
        except:
            return ""

    # TXT
    if name.endswith(".txt") or "text" in mime:
        try:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except:
            return ""

    return ""


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
SKILL_KEYWORDS = [
    "python", "sql", "java", "ml", "machine learning", "deep learning",
    "excel", "pandas", "numpy", "power bi", "tableau",
    "aws", "azure", "cloud", "nlp"
]
DESIGNATION_KEYWORDS = ["developer", "engineer", "analyst", "consultant", "manager", "scientist", "intern"]
EDU_KEYWORDS = ["bachelor", "master", "b.sc", "m.sc", "phd", "degree", "university", "college"]
CERT_KEYWORDS = ["certified", "certificate", "pmp", "aws certified", "azure certified", "google certified"]


def count_matches(text, keywords):
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def compute_ats_score(text):
    t = text.lower()
    score = 0
    if any(k in t for k in SKILL_KEYWORDS): score += 20
    if any(k in t for k in CERT_KEYWORDS): score += 20
    if any(k in t for k in EDU_KEYWORDS): score += 20
    if any(k in t for k in DESIGNATION_KEYWORDS): score += 20
    if len(t.split()) > 150: score += 20
    return score


def extract_features_from_resume(text, platform, category):
    wc = len(text.split())
    numeric = {
        "skill_count": count_matches(text, SKILL_KEYWORDS),
        "designation_count": count_matches(text, DESIGNATION_KEYWORDS),
        "education_count": count_matches(text, EDU_KEYWORDS),
        "certification_count": count_matches(text, CERT_KEYWORDS),
        "word_count": wc,
        "ATS_score": compute_ats_score(text)
    }
    categorical = {"platform": platform, "Category": category}
    return numeric, categorical


def encode_features(numeric, categorical):
    x_num = np.array([[numeric[col] for col in NUMERIC_FEATURES]], dtype=float)
    x_num_scaled = scaler.transform(x_num).astype(np.float32)

    x_cat = []
    for col in CATEGORICAL_FEATURES:
        x_cat.append(cat_maps[col].get(str(categorical[col]), 0))

    x_cat = np.array([x_cat], dtype=np.int64)
    full_input = np.concatenate([x_num_scaled, x_cat.astype(float)], axis=1)
    return x_num_scaled, x_cat, full_input


# ---------------------------------------------------------
# MODEL PREDICT
# ---------------------------------------------------------
def model_predict(x_numpy):
    x_numpy = np.array(x_numpy, dtype=float)
    if x_numpy.ndim == 1: x_numpy = x_numpy.reshape(1, -1)

    x_num = x_numpy[:, :len(NUMERIC_FEATURES)]
    x_cat = x_numpy[:, len(NUMERIC_FEATURES):].astype(np.int64)

    x_num_t = torch.tensor(x_num, dtype=torch.float32).to(DEVICE)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(x_num_t, x_cat_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


# ---------------------------------------------------------
# STREAMLIT INPUT FORM
# ---------------------------------------------------------
with st.form("resume_eval_form"):
    st.subheader("📤 Upload Resume or Paste Text")

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF, DOCX, or TXT):",
        type=["pdf", "docx", "txt"]
    )

    resume_text_box = st.text_area(
        "Or paste your resume text here:",
        height=220,
        help="Paste full resume text here."
    )

    col1, col2 = st.columns(2)
    with col1:
        platform = st.selectbox("Platform:", sorted(cat_maps["platform"].keys()))
    with col2:
        category = st.selectbox("Job Category:", sorted(cat_maps["Category"].keys()))

    submitted = st.form_submit_button("🔍 Evaluate Resume")
# ---------------------------------------------------------
# MAIN PIPELINE + STRONG RESUME VALIDATION
# ---------------------------------------------------------
if submitted:

    resume_text = ""

    # ---------------------------------------------------------
    # 1) HANDLE UPLOADED FILE
    # ---------------------------------------------------------
    if uploaded_file:
        name = uploaded_file.name.lower()

        if not (name.endswith(".pdf") or name.endswith(".docx") or name.endswith(".txt")):
            st.error("❌ Invalid file type. Only PDF, DOCX, or TXT resumes allowed.")
            st.stop()

        resume_text = extract_text_from_file(uploaded_file)

        if not resume_text.strip():
            st.error("❌ Unable to read text. Upload a proper resume file.")
            st.stop()

    # ---------------------------------------------------------
    # 2) HANDLE TEXT BOX
    # ---------------------------------------------------------
    elif resume_text_box.strip():
        resume_text = resume_text_box

    else:
        st.warning("⚠ Upload a resume file OR paste your resume text.")
        st.stop()

    # ---------------- STRONG RESUME VALIDATION -----------------

    text_lower = resume_text.lower()

    # ✓ MINIMUM LENGTH
    if len(resume_text.split()) < 30:
        st.error("❌ Resume text is too short. Upload a valid resume.")
        st.stop()

    # ✓ MAX LENGTH (avoid reports/research papers)
    if len(resume_text.split()) > 3000:
        st.error("❌ This file looks too large (more than 3000 words). It does NOT appear to be a resume.")
        st.stop()

    # ✓ MUST HAVE SECTION HEADINGS
    resume_sections = [
        "experience", "work experience",
        "education", "skills", "projects",
        "certifications", "summary", "objective"
    ]
    if not any(sec in text_lower for sec in resume_sections):
        st.error("❌ This file does not contain any resume sections (Experience, Education, Skills, Projects). "
                 "It does NOT appear to be a resume.")
        st.stop()

    # ✓ MUST HAVE EMAIL OR PHONE
    import re
    has_email = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text))
    has_phone = bool(re.search(r"\b\d{10}\b|\+?\d{1,3}[-.\s]?\d{10}", resume_text))

    if not (has_email or has_phone):
        st.error("❌ No email or phone number found. This is not a valid resume.")
        st.stop()

    # ✓ REJECT ACADEMIC / PROJECT REPORTS
    academic_keywords = [
        "methodology", "results", "findings", "analysis",
        "dataset", "research", "proposed", "framework",
        "introduction", "conclusion"
    ]
    academic_hits = sum(1 for w in academic_keywords if w in text_lower)

    if academic_hits >= 6:
        st.error("❌ The content looks like an academic/project document, NOT a resume.")
        st.stop()

    # ---------------------------------------------------------
    # PREVIEW
    # ---------------------------------------------------------
    st.subheader("📝 Resume Text Preview")
    st.write(resume_text[:800] + ("..." if len(resume_text) > 800 else ""))

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------
    numeric, categorical = extract_features_from_resume(resume_text, platform, category)
    x_num_scaled, x_cat, full_input = encode_features(numeric, categorical)

    # ---------------------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------------------
    probs = model_predict(full_input)[0]
    pred_class = int(np.argmax(probs))
    prob_strong = float(probs[1])

    if (
        numeric["skill_count"] == 0
        or numeric["ATS_score"] < 30
        or numeric["word_count"] < 150
        or numeric["education_count"] == 0
    ):
        pred_class = 0
        prob_strong = 0.0

    # ---------------------------------------------------------
    # DISPLAY RESULT
    # ---------------------------------------------------------
    st.subheader("📌 ATS Prediction Result")
    if pred_class == 1:
        st.success(f"🌟 STRONG Resume (Probability: {prob_strong:.2f})")
    else:
        st.error(f"❗ WEAK Resume (Probability: {prob_strong:.2f})")

    # ---------------------------------------------------------
    # SHAP FIXED BACKGROUND (NO MORE ERRORS)
    # ---------------------------------------------------------
    bg_sample = df_model.sample(50, random_state=42).copy()

    bg_num = scaler.transform(bg_sample[NUMERIC_FEATURES])

    bg_cat = np.zeros((50, len(CATEGORICAL_FEATURES)), dtype=int)
    for i, col in enumerate(CATEGORICAL_FEATURES):
        bg_cat[:, i] = bg_sample[col].astype(str).map(
            lambda v: cat_maps[col].get(v, 0)
        ).astype(int)

    background = np.hstack([bg_num, bg_cat])

    with st.spinner("Computing SHAP values..."):
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(full_input)

    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    if isinstance(shap_values, list):
        shap_vals_class1 = np.array(shap_values[1]).reshape(-1)
        expected_value_class1 = float(np.array(explainer.expected_value[1]).flatten()[0])
    else:
        shap_vals_class1 = np.array(shap_values).reshape(-1)
        expected_value_class1 = float(np.array(explainer.expected_value).flatten()[0])

    shap_vals_class1 = np.resize(shap_vals_class1, len(feature_names))
    # ---------------------------------------------------------
    # PAGE-3 SESSION SAVES (FOR REJECTION EXPLANATION PAGE)
    # ---------------------------------------------------------
    st.session_state["shap_vals_class1"] = shap_vals_class1
    st.session_state["feature_names"] = feature_names
    st.session_state["numeric"] = numeric
    st.session_state["category"] = category
    st.session_state["pred_class"] = pred_class
    # ---------------------------------------------------------
    # SHAP WATERFALL PLOT
    # ---------------------------------------------------------
    st.write("### 📉 SHAP Waterfall Plot")
    fig = shap.plots._waterfall.waterfall_legacy(
        expected_value_class1, shap_vals_class1, feature_names=feature_names, show=False
    )
    st.pyplot(fig, clear_figure=True)
    # ---------------------------------------------------------
    # SHAP FORCE PLOT
    # ---------------------------------------------------------
    st.write("### ⚡ SHAP Force Plot")
    force_fig = shap.force_plot(
        expected_value_class1, shap_vals_class1, feature_names, matplotlib=True, show=False
    )
    st.pyplot(force_fig, clear_figure=True)

    st.success("✅ SHAP explanation generated successfully!")

