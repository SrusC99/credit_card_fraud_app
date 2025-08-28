# app.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

import pickle

APP_DIR = Path(__file__).parent.resolve()
MODEL_DIR = APP_DIR / "models"
MODEL_NAME = "rf_best_model.pkl"   # exact filename in your repo
MODEL_PATH = MODEL_DIR / MODEL_NAME

# --- Diagnostics: show what Streamlit sees ---
st.caption(f"App dir: {APP_DIR}")
st.caption(f"Models dir exists: {MODEL_DIR.exists()}")
if MODEL_DIR.exists():
    st.caption(f"Files in models/: {[p.name for p in MODEL_DIR.iterdir()]}")

# --- Load the model robustly ---
model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.warning(f"Model not found at: {MODEL_PATH}")

# ---------------- App config ----------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection")
st.caption("Upload a file, choose a model & threshold — it will score every transaction and flag likely fraud. Get the fraud probability, a Fraud/Legit label, and a downloadable CSV.")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ---------------- Helpers ----------------
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

def read_any_table(upload) -> pd.DataFrame | None:
    """Read CSV/XLSX/XLS; prefer CSV for reliability."""
    if upload is None:
        return None
    name = (upload.name or "").lower()
    try:
        upload.seek(0)
        if name.endswith(".csv"):
            return pd.read_csv(upload)
        elif name.endswith(".xlsx"):
            return pd.read_excel(upload, engine="openpyxl")
        elif name.endswith(".xls"):
            try:
                import xlrd  # requires xlrd==1.2.0 for true .xls
            except Exception:
                raise ValueError("Reading .xls needs xlrd==1.2.0; or resave as CSV/.xlsx.")
            return pd.read_excel(upload, engine="xlrd")
        else:
            # Fallback: try CSV then Excel
            try:
                upload.seek(0);  return pd.read_csv(upload)
            except Exception:
                upload.seek(0);  return pd.read_excel(upload)
    except ValueError as e:
        st.error(f"Could not read the file: {e}")
        st.info("Tip: Save as CSV and upload again to avoid Excel engine issues.")
        return None
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return None

def get_expected_columns_from_model(model):
    # Try to infer feature names from model/pipeline
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    schema_file = MODELS_DIR / "expected_columns.joblib"
    if schema_file.exists():
        try:
            cols = joblib.load(schema_file)
            if isinstance(cols, (list, tuple)):
                return list(cols)
        except Exception:
            pass
    return None  # unknown

def align_to_expected(df: pd.DataFrame, expected_cols):
    if expected_cols is None:
        return df
    missing = [c for c in expected_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected_cols]
    if missing:
        st.warning(f"Missing expected columns (model will use those available): {missing}")
    if extra:
        st.info(f"Extra columns (ignored by the model): {extra[:12]}")
    return df[[c for c in expected_cols if c in df.columns]]

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        st.warning("Some values were not numeric and became NaN — filling with 0.")
        out = out.fillna(0)
    return out

def predict_with_threshold(model, X: pd.DataFrame, threshold: float):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        smin, smax = float(scores.min()), float(scores.max())
        proba = (scores - smin) / (smax - smin + 1e-9)
    else:
        proba = model.predict(X).astype(float)
    preds = (proba >= threshold).astype(int)
    return preds, proba

def smart_pct(p: float) -> str:
    return f"{p:.2%}" if p >= 0.0001 else "<0.01%"

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")

available_models = [p.name for p in MODELS_DIR.glob("*.pkl")]
if not available_models:
    st.sidebar.warning("No model files found in ./models. Put rf_best_model.pkl there.")
model_name = st.sidebar.selectbox("Choose model", options=available_models or ["(no models found)"])

threshold = st.sidebar.slider(
    "Alert threshold (fraud probability)",
    min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Lower → more alerts; Higher → fewer alerts."
)

uploaded = st.sidebar.file_uploader(
    "Upload CSV/XLS/XLSX (same features as training or raw if model is a Pipeline).",
    type=["csv", "xlsx", "xls"]
)

# IMPORTANT: give the checkbox a key; we’ll read from session_state
st.sidebar.checkbox("Show only flagged transactions", value=False, key="only_flags")

preview_limit = st.sidebar.number_input(
    "Preview rows (table)", min_value=10, max_value=2000, value=200, step=10
)

# ---------------- Main (batch only) ----------------
if not available_models:
    st.info("Waiting for a model… Add a .pkl in ./models, select it above.")
else:
    try:
        model = load_model(MODELS_DIR / model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

    if model is not None and uploaded is not None:
        df_raw = read_any_table(uploaded)
        if df_raw is not None and len(df_raw) > 0:

            expected_cols = get_expected_columns_from_model(model)
            X = align_to_expected(df_raw.copy(), expected_cols)
            X = coerce_numeric(X)

            try:
                preds, proba = predict_with_threshold(model, X, threshold=threshold)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            else:
                # Build results on original df for user context
                result = df_raw.copy()
                result["P(fraud)"]   = proba
                result["is_fraud"]   = (preds == 1)
                result["Prediction"] = np.where(result["is_fraud"], "Fraud", "Legit")
                result["Confidence"] = np.where(
                    result["is_fraud"],
                    [f"Fraud {smart_pct(p)}" for p in proba],
                    [f"Legit {smart_pct(1 - p)}" for p in proba]
                )

                # KPIs
                total = len(result)
                flagged = int(result["is_fraud"].sum())
                rate = flagged / max(total, 1)
                k1, k2, k3 = st.columns(3)
                k1.metric("Total transactions scored", f"{total:,}")
                k2.metric("Flagged as suspicious", f"{flagged:,}")
                k3.metric("Flag rate", f"{rate:.2%}")

                # Subset once based on checkbox state
                only_flags = st.session_state.get("only_flags", False)
                subset = result[result["is_fraud"]] if only_flags else result

                # Preview
                st.markdown("**Preview**")
                if len(subset) == 0:
                    st.info("No transactions were flagged at the current threshold.")
                else:
                    st.caption(
                        f"Showing {min(preview_limit, len(subset))} of {len(subset):,} rows "
                        f"({'flagged only' if only_flags else 'all rows'})."
                    )
                    st.dataframe(subset.head(int(preview_limit)), use_container_width=True)

                # Download: user chooses all vs only flagged
                download_choice = st.radio(
                    "Download which rows?",
                    options=["All rows", "Only flagged"],
                    horizontal=True,
                )
                to_download = subset if download_choice == "Only flagged" else result
                st.download_button(
                    "Download scored CSV",
                    data=to_download.to_csv(index=False).encode(),
                    file_name=(
                        "fraud_scored_flagged.csv"
                        if download_choice == "Only flagged"
                        else "fraud_scored.csv"
                    ),
                    mime="text/csv",
                )
        else:
            st.info("Uploaded file is empty or could not be parsed.")
