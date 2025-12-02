
# pages/6_Resume_Compare.py

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import joblib
import json
import pandas as pd

st.set_page_config(page_title="Resume Compare", layout="wide")
st.title("🆚 Resume Comparison Tool")

BASE_DIR = r"C:\Users\Kanisha Pathy\Downloads\Research - Practicum"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMERIC_FEATURES = [
    "skill_count", "designation_count", "education_count",
    "certification_count", "word_count", "ATS_score"
]
CATEGORICAL_FEATURES = ["platform", "Category"]

# -------------------------------------------------------------
# LOAD MODEL + ENCODERS
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(BASE_DIR + r"\TabTransformer_Scaler.pkl")
    cat_maps = joblib.load(BASE_DIR + r"\TabTransformer_Cat_Maps.pkl")
    with open(BASE_DIR + r"\TabTransformer_Hyperparams.json", "r") as f:
        best = json.load(f)

    # Build TabTransformer
    class TabTransformer(torch.nn.Module):
        def __init__(self, num_numeric, cat_cardinalities,
                     d_model=32, n_heads=4, n_layers=2,
                     dropout=0.1, num_classes=2):
            super().__init__()
            self.cat_embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(card, d_model) for card in cat_cardinalities
            ])
            enc = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=128, dropout=dropout,
                batch_first=True
            )
            self.transformer = torch.nn.TransformerEncoder(enc, num_layers=n_layers)
            self.numeric_proj = torch.nn.Sequential(
                torch.nn.Linear(num_numeric, d_model),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(2 * d_model, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(64, num_classes)
            )

        def forward(self, x_num, x_cat):
            cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_embs = torch.stack(cat_embs, dim=1)
            cat_ctx = self.transformer(cat_embs)
            cat_pooled = cat_ctx.mean(dim=1)
            num_repr = self.numeric_proj(x_num)
            h = torch.cat([cat_pooled, num_repr], dim=1)
            return self.fc(h)

    cat_cardinalities = [len(cat_maps[c]) for c in CATEGORICAL_FEATURES]

    model = TabTransformer(
        num_numeric=len(NUMERIC_FEATURES),
        cat_cardinalities=cat_cardinalities,
        d_model=best["d_model"],
        n_heads=best["n_heads"],
        n_layers=best["n_layers"],
        dropout=best["dropout"],
    ).to(DEVICE)

    state = torch.load(BASE_DIR + r"\TabTransformer_Final_Model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, cat_maps

model, scaler, cat_maps = load_artifacts()


# -------------------------------------------------------------
# FEATURE EXTRACTORS
# -------------------------------------------------------------
SKILL_KEYWORDS = [
    "python","sql","java","ml","machine learning","deep learning","excel","pandas","numpy",
    "power bi","tableau","aws","azure","cloud","nlp"
]
DESIGNATION_KEYWORDS = ["developer","engineer","analyst","consultant","manager","scientist","intern"]
EDU_KEYWORDS = ["bachelor","master","b.sc","m.sc","phd","degree","university","college"]
CERT_KEYWORDS = ["certified","certificate","pmp","aws certified","azure certified","google certified"]

def count_matches(text, keywords):
    t = text.lower()
    return sum(1 for k in keywords if k in t)

def compute_ats_score(text):
    t = text.lower()
    score = 0
    if any(k in t for k in SKILL_KEYWORDS):       score += 20
    if any(k in t for k in CERT_KEYWORDS):        score += 20
    if any(k in t for k in EDU_KEYWORDS):         score += 20
    if any(k in t for k in DESIGNATION_KEYWORDS): score += 20
    if len(t.split()) > 150:                      score += 20
    return score

def extract_features(text, platform, category):
    numeric = {
        "skill_count": count_matches(text, SKILL_KEYWORDS),
        "designation_count": count_matches(text, DESIGNATION_KEYWORDS),
        "education_count": count_matches(text, EDU_KEYWORDS),
        "certification_count": count_matches(text, CERT_KEYWORDS),
        "word_count": len(text.split()),
        "ATS_score": compute_ats_score(text),
    }
    categorical = {"platform": platform, "Category": category}
    return numeric, categorical

def encode(numeric, categorical):
    x_num = np.array([[numeric[c] for c in NUMERIC_FEATURES]], dtype=float)
    x_scaled = scaler.transform(x_num).astype(np.float32)

    x_cat = []
    for col in CATEGORICAL_FEATURES:
        mapping = cat_maps[col]
        val = str(categorical[col])
        x_cat.append(mapping.get(val, 0))
    x_cat = np.array([x_cat], dtype=np.int64)

    full = np.concatenate([x_scaled, x_cat.astype(float)], axis=1)
    return full


def model_predict(x_numpy):
    x_numpy = np.array(x_numpy, dtype=float)
    if x_numpy.ndim == 1:
        x_numpy = x_numpy.reshape(1, -1)

    n = len(NUMERIC_FEATURES)
    x_num = x_numpy[:, :n]
    x_cat = x_numpy[:, n:].astype(np.int64)

    x_num_t = torch.tensor(x_num, dtype=torch.float32).to(DEVICE)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(x_num_t, x_cat_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# -------------------------------------------------------------
# USER INPUT SECTION
# -------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.markdown("### 📄 Resume A")
    textA = st.text_area("Paste Resume A", height=250)
    platformA = st.selectbox("Platform A", sorted(cat_maps["platform"].keys()))
    categoryA = st.selectbox("Category A", sorted(cat_maps["Category"].keys()))

with colB:
    st.markdown("### 📄 Resume B")
    textB = st.text_area("Paste Resume B", height=250)
    platformB = st.selectbox("Platform B", sorted(cat_maps["platform"].keys()))
    categoryB = st.selectbox("Category B", sorted(cat_maps["Category"].keys()))

run_compare = st.button("🔍 Compare Resumes")


# -------------------------------------------------------------
# PROCESS BOTH RESUMES
# -------------------------------------------------------------
if run_compare:

    if not textA.strip() or not textB.strip():
        st.warning("Paste both resumes to compare.")
        st.stop()

    def process_resume(text, platform, category):
        numeric, categorical = extract_features(text, platform, category)
        full = encode(numeric, categorical)
        probs = model_predict(full)[0]
        pred = int(np.argmax(probs))
        prob_strong = float(probs[1])

        # ----------------------------------------------------
        #  SANITY CORRECTION — SAME LOGIC AS PAGE 2
        # ----------------------------------------------------
        if (
            numeric["skill_count"] == 0
            or numeric["education_count"] == 0
            or numeric["ATS_score"] < 30
            or numeric["word_count"] < 150
        ):
            pred = 0
            prob_strong = 0.0

        # Strength Score
        strength = (
            numeric["ATS_score"]
            + numeric["skill_count"] * 3
            + numeric["certification_count"] * 5
            + (1 if numeric["education_count"] > 0 else -5)
        )
        strength = max(0, min(100, int(strength)))

        return numeric, categorical, pred, prob_strong, strength

    numA, catA, predA, probA, scoreA = process_resume(textA, platformA, categoryA)
    numB, catB, predB, probB, scoreB = process_resume(textB, platformB, categoryB)

    # -------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------
    st.markdown("## 🔎 Comparison Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resume A")
        st.write(f"Prediction: **{'STRONG' if predA == 1 else 'WEAK'}**")
        st.write(f"Strength Score: **{scoreA}/100**")

    with col2:
        st.subheader("Resume B")
        st.write(f"Prediction: **{'STRONG' if predB == 1 else 'WEAK'}**")
        st.write(f"Strength Score: **{scoreB}/100**")

    # -------------------------------------------------------------
    # RECOMMENDATION
    # -------------------------------------------------------------
    st.markdown("### 🏆 Recommended Resume")

    if scoreA > scoreB:
        st.success("Resume **A** is stronger overall.")
    elif scoreB > scoreA:
        st.success("Resume **B** is stronger overall.")
    else:
        st.info("Both resumes are equally strong.")

    # -------------------------------------------------------------
    # RADAR CHARTS
    # -------------------------------------------------------------
    def radar_chart(title, numeric):
        labels = ["Skills","Designation","Education","Certification","Length","ATS"]
        values = [
            numeric["skill_count"] * 10,
            numeric["designation_count"] * 10,
            numeric["education_count"] * 20,
            numeric["certification_count"] * 25,
            min(100, int((numeric["word_count"] / 900) * 100)),
            numeric["ATS_score"],
        ]
        values.append(values[0])
        labels.append(labels[0])

        fig = go.Figure(go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name=title
        ))
        fig.update_layout(height=350, showlegend=False)
        return fig

    st.markdown("### 🌐 ATS Radar Comparison")

    colr1, colr2 = st.columns(2)
    with colr1:
        st.subheader("Resume A Shape")
        st.plotly_chart(radar_chart("A", numA), use_container_width=True)
    with colr2:
        st.subheader("Resume B Shape")
        st.plotly_chart(radar_chart("B", numB), use_container_width=True)


