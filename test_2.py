"""
Startup Success Predictor
- Uses modest trees + stronger L1/L2 + scale_pos_weight to reduce overfitting.
- Shows fit-quality, 5-fold CV, confusion matrix, ROC, learning curve.
- SHAP local explanations (TreeExplainer).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

st.set_page_config(page_title="Startup Success Prediction", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef6ff 0%, #fdf2f8 35%, #f0fdf4 100%);
    }
    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1d4ed8 35%, #7c3aed 100%);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 16px 40px rgba(76, 29, 149, 0.22);
        margin-bottom: 1.2rem;
    }
    .hero-card h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .hero-card p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.55rem;
        margin-bottom: 0;
        font-size: 1.02rem;
    }
    div[data-testid="stMetricContainer"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(239,246,255,0.9));
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.12);
    }
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        box-shadow: 0 10px 20px rgba(239, 68, 68, 0.2);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.2);
    }
    h1, h2, h3 {
        color: #111827;
    }
    .stHeader, .stSubheader {
        color: #312e81;
    }
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(30, 41, 59, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Configuration
# ----------------------
DATA_FILE = "startup data.csv"   # put your CSV in the same folder or update path
MODEL_FILE = "startup_model.pkl"
STATS_FILE = "startup_stats.pkl"

FEATURES = [
    "has_VC",
    "has_angel",
    "has_roundA",
    "has_roundB",
    "has_roundC",
    "has_roundD",
    "funding_rounds",
    "funding_total_usd",
    "avg_participants",
    "is_top500",
    "milestones",
]
TARGET = "status"

# ----------------------
# Helpers
# ----------------------
def safe_num(x, default=0.0):
    try:
        return float(x)
    except:
        return float(default)

@st.cache_data(show_spinner=False)
def prepare_model_and_stats(data_path, features, target):
    """
    Load data, clean, train XGBoost (stronger regularization + smaller trees),
    compute stats (medians, importances, correlations), compute train/test accuracies,
    and compute 5-fold CV accuracy.
    """
    df = pd.read_csv(data_path)

    # Keep only features present in the CSV
    keep_features = [c for c in features if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV file.")
    df = df[keep_features + [target]].copy()

    # Map target
    df[target] = df[target].map({"acquired": 1, "closed": 0})
    df = df.dropna(subset=[target])

    # Binary columns cleanup
    binary_cols = [
        c for c in ["has_VC","has_angel","has_roundA","has_roundB","has_roundC","has_roundD","is_top500"]
        if c in df.columns
    ]
    for c in binary_cols:
        if df[c].dtype == object:
            df[c] = df[c].str.lower().map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0}).fillna(df[c])
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Numeric columns: use median to fill missing
    numeric_cols = [c for c in ["funding_rounds","funding_total_usd","avg_participants","milestones"] if c in df.columns]
    medians = {}
    for c in numeric_cols:
        medians[c] = float(df[c].median()) if not df[c].dropna().empty else 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(medians[c])

    # Final X, y
    X = df[[c for c in features if c in df.columns]]
    y = df[target].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Stronger regularization + smaller trees (Option A)
    # -------------------------
    # compute class weight ratio
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg) / max(1.0, float(pos))

    model = XGBClassifier(
        n_estimators=200,           # smaller number of trees
        learning_rate=0.03,         # gentle learning rate
        max_depth=3,                # shallower trees to reduce complexity
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=2.0,              # stronger L1
        reg_lambda=5.0,             # stronger L2
        min_child_weight=8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)  # simple fit — compatible with all versions

    # -----------------------------
    # TRAIN vs TEST accuracy check
    # -----------------------------
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    fit_gap = float(train_acc - test_acc)

    # Evaluate on test
    y_pred = test_pred
    acc = test_acc
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance and correlation
    try:
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    except Exception:
        booster = model.get_booster()
        fdict = booster.get_score(importance_type='gain')
        imp_series = pd.Series({k: fdict.get(k, 0.0) for k in X.columns})
        feat_imp = imp_series.sort_values(ascending=False)

    corr = X.join(y).corr()[target].drop(target).reindex(feat_imp.index)

    # 5-fold CV accuracy for overall estimate
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
    except Exception:
        cv_mean, cv_std = None, None

    stats = {
        "accuracy": float(acc),
        "report": report_dict,
        "feature_importances": feat_imp.to_dict(),
        "correlations": corr.to_dict(),
        "medians": {c: float(medians[c]) if c in medians else 0.0 for c in features},
        "train_shape": X.shape,
        "fit_quality": {
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "fit_gap": fit_gap
        },
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        # Keep snapshots for diagnostics/learning curve
        "X_snapshot": X.copy(),
        "y_snapshot": y.copy(),
        "model_wrapper": model
    }

    return model, stats

@st.cache_resource(show_spinner=False)
def load_model_if_exists(model_file):
    """Load a previously saved model (if available). Cache the loaded object."""
    if os.path.exists(model_file):
        try:
            with open(model_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    """Return a TreeExplainer for an XGBoost model (leading underscore avoids hashing)."""
    return shap.TreeExplainer(_model)


def normalize_feature_frame(df, feature_cols, medians):
    """Standardize a DataFrame to the model feature layout and fill missing values."""
    cleaned = df.copy()

    for col in feature_cols:
        if col not in cleaned.columns:
            cleaned[col] = 0

    cleaned = cleaned[feature_cols].copy()

    binary_cols = [
        c for c in ["has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD", "is_top500"]
        if c in cleaned.columns
    ]
    for c in binary_cols:
        if cleaned[c].dtype == object:
            cleaned[c] = cleaned[c].astype(str).str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce").fillna(0).astype(int)

    numeric_cols = [c for c in ["funding_rounds", "funding_total_usd", "avg_participants", "milestones"] if c in cleaned.columns]
    for c in numeric_cols:
        default = float(medians.get(c, 0.0)) if c in medians else 0.0
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce").fillna(default)

    return cleaned.apply(pd.to_numeric, errors="coerce").fillna(0)


def predict_single_row(model, input_dict, feature_cols, medians):
    """Return probability and class for a single startup input."""
    X_input = pd.DataFrame([input_dict])
    X_input = normalize_feature_frame(X_input, feature_cols, medians)

    try:
        feature_order = list(model.feature_names_in_)
    except Exception:
        feature_order = feature_cols

    if len(feature_order) > 0:
        for c in feature_order:
            if c not in X_input.columns:
                X_input[c] = 0
        X_input = X_input[feature_order]

    proba = float(model.predict_proba(X_input)[0, 1])
    pred_class = int(model.predict(X_input)[0])
    return proba, pred_class


def generate_recommendations(input_dict, medians):
    """Return short, rule-based recommendations for the startup profile."""
    recommendations = []

    if input_dict.get("funding_total_usd", 0) < float(medians.get("funding_total_usd", 0.0)):
        recommendations.append("Funding is below the median. Raising more capital could improve the odds of acquisition.")

    if input_dict.get("milestones", 0) < float(medians.get("milestones", 0.0)):
        recommendations.append("Milestones are below the median. Reaching more milestones before the next funding round may strengthen the startup profile.")

    if input_dict.get("has_roundA", 0) != 1:
        recommendations.append("Series A funding is missing. Securing a Series A round could improve traction and investor confidence.")

    if input_dict.get("has_VC", 0) != 1:
        recommendations.append("VC backing is absent. Improving VC support could strengthen the startup profile.")

    if input_dict.get("avg_participants", 0) < float(medians.get("avg_participants", 0.0)):
        recommendations.append("Investor participation is below the median. Increasing investor engagement may signal stronger market confidence.")

    if input_dict.get("funding_rounds", 0) < float(medians.get("funding_rounds", 0.0)):
        recommendations.append("Funding rounds are below the median. Adding a new round could improve capital readiness and growth potential.")

    if not recommendations:
        recommendations.append("The startup profile already looks strong relative to the median benchmarks. Focus on sustaining momentum and execution quality.")

    return recommendations


def build_investor_report(input_dict, probability, pred_class, medians):
    """Build a text report snapshot for the startup prediction and recommendations."""
    forecast = "Acquired (SUCCESS)" if pred_class == 1 else "Closed (FAIL)"
    recs = generate_recommendations(input_dict, medians)
    top_recs = recs[:3]

    lines = [
        "Startup Investor Report",
        "=======================",
        f"Forecast: {forecast}",
        f"Probability of success: {probability:.3f}",
        "",
        "Key inputs:",
    ]

    for key in [
        "has_VC",
        "has_angel",
        "has_roundA",
        "has_roundB",
        "has_roundC",
        "has_roundD",
        "funding_rounds",
        "funding_total_usd",
        "avg_participants",
        "is_top500",
        "milestones",
    ]:
        if key in input_dict:
            lines.append(f"- {key}: {input_dict[key]}")

    lines.extend(["", "Recommended next actions:"])
    for idx, rec in enumerate(top_recs, start=1):
        lines.append(f"{idx}. {rec}")

    lines.extend(["", "Report generated by Startup Success Predictor"])
    return "\n".join(lines)

# ----------------------
# App start
# ----------------------
st.markdown(
    """
    <div class="hero-card">
        <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
            <div style="font-size: 2.3rem;">🚀</div>
            <div>
                <h1>Startup Success Prediction</h1>
                <p>AI-powered startup evaluation dashboard for assessing acquisition likelihood, business traction, and investor-readiness.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# NOTE: sidebar model info & fit quality display intentionally removed per user request.

if not os.path.exists(DATA_FILE):
    st.error(f"Dataset file '{DATA_FILE}' not found. Put it in the working directory or update DATA_FILE.")
    st.stop()

with st.spinner("Training or loading model (cached)..."):
    model, stats = prepare_model_and_stats(DATA_FILE, FEATURES, TARGET)

# Save model & stats (best-effort)
try:
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(STATS_FILE, "wb") as f:
        pickle.dump(stats, f)
except Exception:
    pass

# UI inputs (use medians as defaults)
med = stats["medians"]

st.header("Batch prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV with startup features", type=["csv"])

if uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file)
        feature_cols = [c for c in FEATURES if c in upload_df.columns] if upload_df.columns is not None else []
        if not feature_cols:
            st.warning("No expected startup feature columns found in the uploaded CSV. Check the column names.")
        else:
            prepared_batch = normalize_feature_frame(upload_df, FEATURES, med)
            batch_probs = model.predict_proba(prepared_batch)[:, 1]
            batch_preds = ["Acquired" if p >= 0.5 else "Closed" for p in batch_probs]
            batch_results = upload_df.copy()
            batch_results["Probability_of_success"] = batch_probs
            batch_results["Predicted_label"] = batch_preds
            st.dataframe(batch_results)

            csv_data = batch_results.to_csv(index=False)
            st.download_button(
                label="Download batch prediction results",
                data=csv_data,
                file_name="startup_batch_predictions.csv",
                mime="text/csv",
            )

            st.subheader("Batch dashboard")
            best = batch_results.sort_values("Probability_of_success", ascending=False).head(5)
            st.dataframe(best, use_container_width=True)

            fig_hist, ax = plt.subplots(figsize=(8, 4))
            ax.hist(batch_results["Probability_of_success"], bins=10, color="steelblue", edgecolor="black")
            ax.set_title("Distribution of predicted success probabilities")
            ax.set_xlabel("Probability of success")
            ax.set_ylabel("Count")
            st.pyplot(fig_hist)
            plt.clf()

            fig_scatter, ax = plt.subplots(figsize=(8, 4))
            funding_values = batch_results.get("funding_total_usd", pd.Series([0] * len(batch_results)))
            ax.scatter(funding_values, batch_results["Probability_of_success"], alpha=0.7)
            ax.set_title("Funding vs success probability")
            ax.set_xlabel("funding_total_usd")
            ax.set_ylabel("Probability of success")
            st.pyplot(fig_scatter)
            plt.clf()

            st.subheader("Batch recommendations")
            for _, row in best.head(3).iterrows():
                label = row.get("startup", "Startup") if "startup" in row else "Startup"
                st.write(f"**{label}**")
                row_dict = row.to_dict()
                row_dict = {k: v for k, v in row_dict.items() if isinstance(v, (int, float, str))}
                recs = generate_recommendations(row_dict, med)
                for rec in recs[:3]:
                    st.write(f"- {rec}")

            batch_report = "\n".join([
                "Startup Batch Report",
                "===================",
                *[f"{idx + 1}. {row.get('startup', f'Startup {idx + 1}')} | Probability={row['Probability_of_success']:.3f} | Label={row['Predicted_label']}" for idx, row in best.iterrows()],
                "",
                "Recommendations:",
                *[f"- {rec}" for _, row in best.head(3).iterrows() for rec in generate_recommendations({k: v for k, v in row.to_dict().items() if isinstance(v, (int, float, str))}, med)[:2]]
            ])
            st.download_button(
                label="Download batch report",
                data=batch_report,
                file_name="startup_batch_report.txt",
                mime="text/plain",
            )

            st.subheader("Portfolio dashboard")
            min_prob = st.slider("Minimum success probability", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
            max_funding = st.slider("Maximum funding filter (USD)", min_value=0, max_value=int(batch_results["funding_total_usd"].max()) if not batch_results.empty and "funding_total_usd" in batch_results.columns else 10000000, value=int(batch_results["funding_total_usd"].max()) if not batch_results.empty and "funding_total_usd" in batch_results.columns else 10000000)

            filtered = batch_results.copy()
            if "funding_total_usd" in filtered.columns:
                filtered = filtered[filtered["funding_total_usd"] <= max_funding]
            filtered = filtered[filtered["Probability_of_success"] >= min_prob]

            st.metric("Total startups in view", len(filtered))
            colA, colB, colC = st.columns(3)
            colA.metric("Acquired count", int((filtered["Predicted_label"] == "Acquired").sum()))
            colB.metric("Closed count", int((filtered["Predicted_label"] == "Closed").sum()))
            colC.metric("Average success probability", f"{filtered['Probability_of_success'].mean():.3f}" if not filtered.empty else "0.000")

            st.dataframe(filtered.sort_values("Probability_of_success", ascending=False), use_container_width=True)

            if not filtered.empty:
                fig_portfolio, ax = plt.subplots(figsize=(8, 4))
                ax.bar(filtered["Predicted_label"].value_counts().index, filtered["Predicted_label"].value_counts().values, color=["#4CAF50", "#F44336"])
                ax.set_title("Predicted outcomes in filtered portfolio")
                ax.set_xlabel("Label")
                ax.set_ylabel("Count")
                st.pyplot(fig_portfolio)
                plt.clf()
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")

st.header("What-if scenario simulator")
scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    scen_vc = st.selectbox("Scenario: Has VC funding?", options=[1,0], index=0 if med.get("has_VC",0)>=0.5 else 1)
    scen_angel = st.selectbox("Scenario: Has Angel funding?", options=[1,0], index=0 if med.get("has_angel",0)>=0.5 else 1)
    scen_roundA = st.selectbox("Scenario: Has Round A?", options=[1,0], index=0 if med.get("has_roundA",0)>=0.5 else 1)
    scen_roundB = st.selectbox("Scenario: Has Round B?", options=[1,0], index=0 if med.get("has_roundB",0)>=0.5 else 1)
    scen_roundC = st.selectbox("Scenario: Has Round C?", options=[1,0], index=0 if med.get("has_roundC",0)>=0.5 else 1)
    scen_roundD = st.selectbox("Scenario: Has Round D?", options=[1,0], index=0 if med.get("has_roundD",0)>=0.5 else 1)

with scenario_col2:
    scen_funding_rounds = st.number_input("Scenario: funding_rounds", min_value=0.0, value=float(med.get("funding_rounds", 0.0)))
    scen_funding_total_usd = st.number_input("Scenario: funding_total_usd (USD)", min_value=0.0, value=float(med.get("funding_total_usd", 0.0)), step=1000.0)
    scen_avg_participants = st.number_input("Scenario: avg_participants", min_value=0.0, value=float(med.get("avg_participants", 1.0)), step=0.1)
    scen_top500 = st.selectbox("Scenario: Is Top 500 company?", options=[1,0], index=0 if med.get("is_top500",0)>=0.5 else 1)
    scen_milestones = st.number_input("Scenario: milestones", min_value=0.0, value=float(med.get("milestones", 0.0)), step=1.0)

if st.button("Run scenario simulation"):
    scenario_dict = {
        "has_VC": int(scen_vc),
        "has_angel": int(scen_angel),
        "has_roundA": int(scen_roundA),
        "has_roundB": int(scen_roundB),
        "has_roundC": int(scen_roundC),
        "has_roundD": int(scen_roundD),
        "funding_rounds": float(scen_funding_rounds),
        "funding_total_usd": float(scen_funding_total_usd),
        "avg_participants": float(scen_avg_participants),
        "is_top500": int(scen_top500),
        "milestones": float(scen_milestones),
    }

    scenario_proba, scenario_pred = predict_single_row(model, scenario_dict, FEATURES, med)
    st.subheader("Scenario result")
    if scenario_pred == 1:
        st.success(f"This scenario predicts: **Acquired (SUCCESS)** with probability {scenario_proba:.3f}")
    else:
        st.error(f"This scenario predicts: **Closed (FAIL)** with probability {scenario_proba:.3f}")

    baseline = sum(float(v) for v in med.values()) if med else 0.0
    impact = scenario_proba - 0.5
    if impact > 0:
        st.info("This scenario is stronger than a neutral baseline and is improving the odds of acquisition.")
    else:
        st.info("This scenario is weaker than a neutral baseline and reduces the odds of acquisition.")

    st.subheader("Recommended actions for this scenario")
    scenario_recs = generate_recommendations(scenario_dict, med)
    for idx, rec in enumerate(scenario_recs, start=1):
        st.write(f"{idx}. {rec}")

    st.markdown("**Scenario highlights:**")
    if scen_funding_total_usd >= float(med.get("funding_total_usd", 0.0)):
        st.write("- Higher funding than the dataset median supports a stronger success signal.")
    else:
        st.write("- Funding is below the dataset median, which may reduce the predicted success probability.")

    if scen_milestones >= float(med.get("milestones", 0.0)):
        st.write("- Milestone count is above the dataset median, which generally strengthens the startup profile.")
    else:
        st.write("- Milestone count is below the dataset median, which may weaken the startup profile.")

    if scen_roundA == 1:
        st.write("- Series A funding is present, which adds a positive signal.")
    if scen_vc == 1:
        st.write("- VC backing is present, which strengthens investor credibility.")
    if scen_top500 == 1:
        st.write("- Top 500 status contributes a positive signal.")

st.header("Startup comparison dashboard")
compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    st.subheader("Startup A")
    compA_vc = st.selectbox("Startup A: Has VC funding?", options=[1,0], index=0 if med.get("has_VC",0)>=0.5 else 1, key="compA_vc")
    compA_angel = st.selectbox("Startup A: Has Angel funding?", options=[1,0], index=0 if med.get("has_angel",0)>=0.5 else 1, key="compA_angel")
    compA_roundA = st.selectbox("Startup A: Has Round A?", options=[1,0], index=0 if med.get("has_roundA",0)>=0.5 else 1, key="compA_roundA")
    compA_roundB = st.selectbox("Startup A: Has Round B?", options=[1,0], index=0 if med.get("has_roundB",0)>=0.5 else 1, key="compA_roundB")
    compA_roundC = st.selectbox("Startup A: Has Round C?", options=[1,0], index=0 if med.get("has_roundC",0)>=0.5 else 1, key="compA_roundC")
    compA_roundD = st.selectbox("Startup A: Has Round D?", options=[1,0], index=0 if med.get("has_roundD",0)>=0.5 else 1, key="compA_roundD")
    compA_funding_rounds = st.number_input("Startup A: funding_rounds", min_value=0.0, value=float(med.get("funding_rounds", 0.0)), key="compA_funding_rounds")
    compA_funding_total_usd = st.number_input("Startup A: funding_total_usd (USD)", min_value=0.0, value=float(med.get("funding_total_usd", 0.0)), step=1000.0, key="compA_funding_total_usd")
    compA_avg_participants = st.number_input("Startup A: avg_participants", min_value=0.0, value=float(med.get("avg_participants", 1.0)), step=0.1, key="compA_avg_participants")
    compA_top500 = st.selectbox("Startup A: Is Top 500 company?", options=[1,0], index=0 if med.get("is_top500",0)>=0.5 else 1, key="compA_top500")
    compA_milestones = st.number_input("Startup A: milestones", min_value=0.0, value=float(med.get("milestones", 0.0)), step=1.0, key="compA_milestones")

with compare_col2:
    st.subheader("Startup B")
    compB_vc = st.selectbox("Startup B: Has VC funding?", options=[1,0], index=0 if med.get("has_VC",0)>=0.5 else 1, key="compB_vc")
    compB_angel = st.selectbox("Startup B: Has Angel funding?", options=[1,0], index=0 if med.get("has_angel",0)>=0.5 else 1, key="compB_angel")
    compB_roundA = st.selectbox("Startup B: Has Round A?", options=[1,0], index=0 if med.get("has_roundA",0)>=0.5 else 1, key="compB_roundA")
    compB_roundB = st.selectbox("Startup B: Has Round B?", options=[1,0], index=0 if med.get("has_roundB",0)>=0.5 else 1, key="compB_roundB")
    compB_roundC = st.selectbox("Startup B: Has Round C?", options=[1,0], index=0 if med.get("has_roundC",0)>=0.5 else 1, key="compB_roundC")
    compB_roundD = st.selectbox("Startup B: Has Round D?", options=[1,0], index=0 if med.get("has_roundD",0)>=0.5 else 1, key="compB_roundD")
    compB_funding_rounds = st.number_input("Startup B: funding_rounds", min_value=0.0, value=float(med.get("funding_rounds", 0.0)), key="compB_funding_rounds")
    compB_funding_total_usd = st.number_input("Startup B: funding_total_usd (USD)", min_value=0.0, value=float(med.get("funding_total_usd", 0.0)), step=1000.0, key="compB_funding_total_usd")
    compB_avg_participants = st.number_input("Startup B: avg_participants", min_value=0.0, value=float(med.get("avg_participants", 1.0)), step=0.1, key="compB_avg_participants")
    compB_top500 = st.selectbox("Startup B: Is Top 500 company?", options=[1,0], index=0 if med.get("is_top500",0)>=0.5 else 1, key="compB_top500")
    compB_milestones = st.number_input("Startup B: milestones", min_value=0.0, value=float(med.get("milestones", 0.0)), step=1.0, key="compB_milestones")

if st.button("Compare startups"):
    startup_a = {
        "has_VC": int(compA_vc),
        "has_angel": int(compA_angel),
        "has_roundA": int(compA_roundA),
        "has_roundB": int(compA_roundB),
        "has_roundC": int(compA_roundC),
        "has_roundD": int(compA_roundD),
        "funding_rounds": float(compA_funding_rounds),
        "funding_total_usd": float(compA_funding_total_usd),
        "avg_participants": float(compA_avg_participants),
        "is_top500": int(compA_top500),
        "milestones": float(compA_milestones),
    }
    startup_b = {
        "has_VC": int(compB_vc),
        "has_angel": int(compB_angel),
        "has_roundA": int(compB_roundA),
        "has_roundB": int(compB_roundB),
        "has_roundC": int(compB_roundC),
        "has_roundD": int(compB_roundD),
        "funding_rounds": float(compB_funding_rounds),
        "funding_total_usd": float(compB_funding_total_usd),
        "avg_participants": float(compB_avg_participants),
        "is_top500": int(compB_top500),
        "milestones": float(compB_milestones),
    }

    prob_a, pred_a = predict_single_row(model, startup_a, FEATURES, med)
    prob_b, pred_b = predict_single_row(model, startup_b, FEATURES, med)

    comparison = pd.DataFrame([
        {"Startup": "Startup A", "Probability_of_success": prob_a, "Prediction": "Acquired" if pred_a == 1 else "Closed"},
        {"Startup": "Startup B", "Probability_of_success": prob_b, "Prediction": "Acquired" if pred_b == 1 else "Closed"},
    ])

    st.subheader("Comparison summary")
    st.dataframe(comparison.sort_values("Probability_of_success", ascending=False), use_container_width=True)

    if prob_a > prob_b:
        st.success("Startup A has a stronger predicted outcome based on the current inputs.")
    elif prob_b > prob_a:
        st.success("Startup B has a stronger predicted outcome based on the current inputs.")
    else:
        st.info("Both startups have very similar predicted probability of success.")

st.header("Startup inputs (important features only)")
col1, col2 = st.columns(2)

with col1:
    has_VC = st.selectbox("Has VC funding?", options=[1,0], index=0 if med.get("has_VC",0)>=0.5 else 1)
    has_angel = st.selectbox("Has Angel funding?", options=[1,0], index=0 if med.get("has_angel",0)>=0.5 else 1)
    has_roundA = st.selectbox("Has Round A?", options=[1,0], index=0 if med.get("has_roundA",0)>=0.5 else 1)
    has_roundB = st.selectbox("Has Round B?", options=[1,0], index=0 if med.get("has_roundB",0)>=0.5 else 1)
    has_roundC = st.selectbox("Has Round C?", options=[1,0], index=0 if med.get("has_roundC",0)>=0.5 else 1)
    has_roundD = st.selectbox("Has Round D?", options=[1,0], index=0 if med.get("has_roundD",0)>=0.5 else 1)

with col2:
    funding_rounds = st.number_input("funding_rounds (count)", min_value=0.0, value=float(med.get("funding_rounds", 0.0)))
    funding_total_usd = st.number_input("funding_total_usd (USD, e.g. 2000000 for $2M)", min_value=0.0, value=float(med.get("funding_total_usd", 0.0)), step=1000.0)
    avg_participants = st.number_input("avg_participants (investors per round)", min_value=0.0, value=float(med.get("avg_participants", 1.0)), step=0.1)
    is_top500 = st.selectbox("Is Top 500 company?", options=[1,0], index=0 if med.get("is_top500",0)>=0.5 else 1)
    milestones = st.number_input("milestones (major achievements)", min_value=0.0, value=float(med.get("milestones", 0.0)), step=1.0)

if st.button("Predict"):
    input_dict = {
        "has_VC": int(has_VC),
        "has_angel": int(has_angel),
        "has_roundA": int(has_roundA),
        "has_roundB": int(has_roundB),
        "has_roundC": int(has_roundC),
        "has_roundD": int(has_roundD),
        "funding_rounds": safe_num(funding_rounds, med.get("funding_rounds", 0.0)),
        "funding_total_usd": safe_num(funding_total_usd, med.get("funding_total_usd", 0.0)),
        "avg_participants": safe_num(avg_participants, med.get("avg_participants", 0.0)),
        "is_top500": int(is_top500),
        "milestones": safe_num(milestones, med.get("milestones", 0.0))
    }

    X_input = pd.DataFrame([input_dict])
    # Ensure column order matches model
    try:
        cols_order = list(model.feature_names_in_)
    except Exception:
        cols_order = X_input.columns.tolist()
    for c in cols_order:
        if c not in X_input.columns:
            X_input[c] = 0
    X_input = X_input[cols_order].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Predict using sklearn wrapper
    proba = float(model.predict_proba(X_input)[0,1])
    pred_class = int(model.predict(X_input)[0])

    st.markdown("---")
    st.subheader("Prediction")
    if pred_class == 1:
        st.success(f"Predicted: **Acquired (SUCCESS)** — probability {proba:.3f}")
    else:
        st.error(f"Predicted: **Closed (FAIL)** — probability {proba:.3f}")

    st.metric("Probability of success (acquired)", f"{proba:.3f}")

    st.subheader("Recommendation engine")
    recs = generate_recommendations(input_dict, med)
    for idx, rec in enumerate(recs, start=1):
        st.write(f"{idx}. {rec}")

    report_text = build_investor_report(input_dict, proba, pred_class, med)
    st.download_button(
        label="Download investor report",
        data=report_text,
        file_name="startup_investor_report.txt",
        mime="text/plain",
    )

    # Global feature importances
    st.subheader("Global feature importances (model)")
    feat_imp_series = pd.Series(stats["feature_importances"]).reindex(cols_order).fillna(0)
    fig, ax = plt.subplots(figsize=(6,4))
    feat_imp_series.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
    plt.clf()

    # ---------------------------
    # Diagnostics: confusion matrix + ROC
    # ---------------------------
    st.subheader("Diagnostics (test set)")
    X_snap = stats.get("X_snapshot")
    y_snap = stats.get("y_snapshot")
    if X_snap is not None and y_snap is not None:
        # split same as training
        _, X_test_full, _, y_test_full = train_test_split(X_snap, y_snap, test_size=0.2, random_state=42, stratify=y_snap)
        y_test_pred = model.predict(X_test_full)
        y_test_proba = model.predict_proba(X_test_full)[:,1]

        # confusion matrix
        cm = confusion_matrix(y_test_full, y_test_pred)
        fig_cm, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title("Confusion matrix (test)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center', color="white" if cm.max()>50 else "black")
        st.pyplot(fig_cm)
        plt.clf()

        # ROC-AUC
        try:
            roc = roc_auc_score(y_test_full, y_test_proba)
            st.write(f"ROC-AUC (test): {roc:.3f}")
            fig_roc, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test_full, y_test_proba, ax=ax)
            st.pyplot(fig_roc)
            plt.clf()
        except Exception:
            st.write("ROC-AUC could not be computed.")
    else:
        st.write("No snapshot data for diagnostics available.")

    # ---------------------------
    # Learning curve (cross-validated)
    # ---------------------------
    st.subheader("Learning curve (5-fold CV)")
    try:
        X_lc = stats["X_snapshot"]
        y_lc = stats["y_snapshot"]
        train_sizes = np.linspace(0.1, 1.0, 5)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_lc, y_lc, cv=5, train_sizes=train_sizes, scoring="accuracy", n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        fig_lc, ax = plt.subplots()
        ax.plot(train_sizes, train_mean, label="Train score")
        ax.plot(train_sizes, test_mean, label="CV score")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning curve")
        ax.legend()
        st.pyplot(fig_lc)
        plt.clf()
    except Exception as e:
        st.write("Learning curve could not be computed:", e)

    # ---------------------------
    # SHAP local explanation (TreeExplainer)
    # ---------------------------
    st.subheader("SHAP local explanation (why this prediction?)")
    try:
        explainer = get_shap_explainer(model)  # shap accepts sklearn wrapper
        shap_values = explainer(X_input)

        # Waterfall plot for the single prediction
        fig_wf = plt.figure(figsize=(8,4))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        st.pyplot(fig_wf)
        plt.clf()

        # Top positive/negative contributors
        sv = shap_values.values[0]
        feat_names = X_input.columns.tolist()
        contrib = list(zip(feat_names, sv))
        contrib_sorted = sorted(contrib, key=lambda x: -abs(x[1]))

        st.markdown("**Top contributors (positive push toward success / negative push toward failure):**")
        pos = [(f, v) for f, v in contrib_sorted if v > 0]
        neg = [(f, v) for f, v in contrib_sorted if v < 0]

        if pos:
            st.markdown("**Top supporting features:**")
            for f, v in pos[:6]:
                st.write(f"- {f}: +{v:.4f} (increases predicted probability)")
        else:
            st.write("No strong supporting features detected.")

        if neg:
            st.markdown("**Top opposing features:**")
            for f, v in neg[:6]:
                st.write(f"- {f}: {v:.4f} (decreases predicted probability)")
        else:
            st.write("No strong opposing features detected.")

    except Exception as e:
        st.error("SHAP explanation failed: " + str(e))
        st.info("If error persists, ensure `shap` is installed (`pip install shap`) and restart the app.")

    # Save input for reference
    try:
        pd.DataFrame([input_dict]).to_csv("last_input_example.csv", index=False)
        st.caption("Saved input to last_input_example.csv")
    except Exception:
        pass

    # Plain-English explanation summary
    explanation_lines = []
    if input_dict["funding_total_usd"] >= med.get("funding_total_usd", 0.0):
        explanation_lines.append("Funding is above the dataset median, which supports a healthier startup profile.")
    else:
        explanation_lines.append("Funding is below the dataset median, which may reduce the predicted probability of acquisition.")

    if input_dict["milestones"] >= med.get("milestones", 0.0):
        explanation_lines.append("Milestone count is above the median, showing stronger business execution.")
    else:
        explanation_lines.append("Milestone count is below the median, which can weaken the startup’s success outlook.")

    if input_dict["has_roundA"] == 1:
        explanation_lines.append("Series A funding is present, which is a positive signal for traction.")
    else:
        explanation_lines.append("Series A funding is missing, which may limit later-stage growth potential.")

    if input_dict["has_VC"] == 1:
        explanation_lines.append("VC backing is present, which generally improves credibility and capital access.")

    st.markdown("### Plain-English explanation")
    for line in explanation_lines:
        st.write("- " + line)

st.markdown("---")
st.markdown(
    "Notes: This configuration uses stronger regularization and smaller trees to reduce overfitting. "
    "If you want further improvements, we can run a randomized hyperparameter search (longer) or add more features."
)
