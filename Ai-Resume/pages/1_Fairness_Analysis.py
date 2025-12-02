

# pages/1_Fairness_Analysis.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

st.title("📊 Fairness Analysis – HR View")

st.markdown("""
This page shows whether the ATS model is **fair** across:
- Different **platforms** (LinkedIn, Indeed, Company Website, etc.)
- Different **job categories** (IT, Finance, Aviation, etc.)
""")

# Load the saved df_test from your notebook
df = pd.read_csv(r"C:\Users\Kanisha Pathy\Downloads\Research - Practicum\Resume_ATS_Fairness.csv")

st.write("### Sample of Evaluation Data")
st.dataframe(df[["platform", "Category", "y_true", "y_pred"]].head())

# === Fairness functions ===
def disparate_impact(df, feature, positive_label=1):
    groups = df[feature].unique()
    selection_rates = {}
    for g in groups:
        sr = (df[df[feature] == g]["y_pred"] == positive_label).mean()
        selection_rates[g] = round(sr, 3)
    reference = max(selection_rates.values())
    di_scores = {g: round(selection_rates[g] / reference, 3)
                 for g in selection_rates}
    return selection_rates, di_scores

def equal_opportunity(df, feature):
    groups = df[feature].unique()
    tpr_scores = {}
    for g in groups:
        gdf = df[df[feature] == g]
        tp = ((gdf["y_true"] == 1) & (gdf["y_pred"] == 1)).sum()
        fn = ((gdf["y_true"] == 1) & (gdf["y_pred"] == 0)).sum()
        tpr = tp / (tp + fn + 1e-6)
        tpr_scores[g] = round(tpr, 3)
    return tpr_scores

def demographic_parity(df, feature):
    groups = df[feature].unique()
    dp_scores = {}
    for g in groups:
        dp_scores[g] = round((df[df[feature] == g]["y_pred"] == 1).mean(), 3)
    return dp_scores

def plot_bar(title, score_dict, ylabel):
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=list(score_dict.keys()), y=list(score_dict.values()), ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

def fairness_heatmap(sr, di, tpr, dp, feature_name):
    df_metric = pd.DataFrame({
        "Selection Rate": sr,
        "Disparate Impact": di,
        "TPR": tpr,
        "Demographic Parity": dp
    })
    fig, ax = plt.subplots(figsize=(7,4))
    sns.heatmap(df_metric.T, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(f"{feature_name} Fairness Heatmap")
    st.pyplot(fig)

# ===== PLATFORM FAIRNESS =====
st.subheader("Platform Fairness")
sr_plat, di_plat = disparate_impact(df, "platform")
tpr_plat = equal_opportunity(df, "platform")
dp_plat = demographic_parity(df, "platform")

st.markdown("**Selection Rate by Platform**")
plot_bar("Selection Rate – Platform", sr_plat, "Selection Rate")

st.markdown("**Disparate Impact (Target ≥ 0.80)**")
plot_bar("Disparate Impact – Platform", di_plat, "DI Score")

st.markdown("**Equal Opportunity (TPR)**")
plot_bar("Equal Opportunity – Platform", tpr_plat, "TPR")

st.markdown("**Demographic Parity (P(y_pred = 1))**")
plot_bar("Demographic Parity – Platform", dp_plat, "P(y_pred=1)")

st.markdown("**Combined Platform Fairness Heatmap**")
fairness_heatmap(sr_plat, di_plat, tpr_plat, dp_plat, "Platform")

# ===== CATEGORY FAIRNESS =====
st.subheader("Category Fairness")
sr_cat, di_cat = disparate_impact(df, "Category")
tpr_cat = equal_opportunity(df, "Category")
dp_cat = demographic_parity(df, "Category")

st.markdown("**Selection Rate by Category**")
plot_bar("Selection Rate – Category", sr_cat, "Selection Rate")

st.markdown("**Disparate Impact – Category**")
plot_bar("Disparate Impact – Category", di_cat, "DI Score")

st.markdown("**Equal Opportunity (TPR)**")
plot_bar("Equal Opportunity – Category", tpr_cat, "TPR")

st.markdown("**Demographic Parity**")
plot_bar("Demographic Parity – Category", dp_cat, "P(y_pred=1)")

st.markdown("**Combined Category Fairness Heatmap**")
fairness_heatmap(sr_cat, di_cat, tpr_cat, dp_cat, "Category")

