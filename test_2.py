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

st.set_page_config(page_title="Startup Success Predictor (Option A - Reg)", layout="centered")

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

# ----------------------
# App start
# ----------------------
st.title("🚀 Startup Success Predictor — Option A (Regularized)")
st.markdown("This run uses stronger regularization + smaller trees to reduce overfitting. Use the UI to predict and inspect diagnostics.")

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

st.markdown("---")
st.markdown(
    "Notes: This configuration uses stronger regularization and smaller trees to reduce overfitting. "
    "If you want further improvements, we can run a randomized hyperparameter search (longer) or add more features."
)
