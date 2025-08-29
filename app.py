import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pickle  #fallback if joblib fails

#Paths & App config
APP_DIR    = Path(__file__).parent.resolve()
MODELS_DIR = APP_DIR / "models"
DATA_DIR   = APP_DIR / "data"

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")

##Helpers
@st.cache_resource
def load_model(path: Path):
    """Load a scikit-learn model saved with joblib or pickle."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

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
            import xlrd  # requires xlrd==1.2.0 if enabling it in requirements
            return pd.read_excel(upload, engine="xlrd")
        else:
            upload.seek(0);  return pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return None

def get_expected_columns_from_model(model):
    """Infer expected feature names from pipeline/model or a saved schema file."""
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
    return None

def align_to_expected(df: pd.DataFrame, expected_cols):
    """Reorder/subset columns to match training schema; warn on diffs."""
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
    """Coerce to numeric; fill NaNs with 0."""
    out = df.apply(pd.to_numeric, errors="coerce")
    return out.fillna(0)

def predict_with_threshold(model, X: pd.DataFrame, threshold: float):
    """Return (preds, proba) using predict_proba/decision_function/predict."""
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

@st.cache_data
def load_default_dataframe() -> pd.DataFrame:
    """Load default dataset: ./data/X_test.csv → first CSV in ./data → synthetic demo."""
    preferred = DATA_DIR / "X_test.csv"
    if preferred.exists():
        return pd.read_csv(preferred)
    if DATA_DIR.exists():
        csvs = sorted(DATA_DIR.glob("*.csv"))
        if csvs:
            return pd.read_csv(csvs[0])
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame({
        "V1": rng.normal(size=n),
        "V2": rng.normal(size=n),
        "Amount": rng.lognormal(mean=3, sigma=1, size=n),
    })

def score_and_render(df_raw: pd.DataFrame, model, threshold: float, source_label: str, preview_limit: int):
    """Common UI for scoring + KPIs + preview + download."""
    expected_cols = get_expected_columns_from_model(model)
    X = align_to_expected(df_raw.copy(), expected_cols)
    X = coerce_numeric(X)

    try:
        preds, proba = predict_with_threshold(model, X, threshold)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    result = df_raw.copy()
    result["P(fraud)"]   = proba
    result["is_fraud"]   = (preds == 1)
    result["Prediction"] = np.where(result["is_fraud"], "Fraud", "Legit")
    result["Confidence"] = np.where(
        result["is_fraud"],
        [f"Fraud {smart_pct(p)}" for p in proba],
        [f"Legit {smart_pct(1 - p)}" for p in proba]
    )

    st.subheader(f"Scoring — {source_label}")
    total = len(result)
    flagged = int(result["is_fraud"].sum())
    rate = flagged / max(total, 1)
    k1, k2, k3 = st.columns(3)
    k1.metric("Total transactions scored", f"{total:,}")
    k2.metric("Flagged as suspicious", f"{flagged:,}")
    k3.metric("Flag rate", f"{rate:.2%}")

    only_flags = st.session_state.get("only_flags", False)
    subset = result[result["is_fraud"]] if only_flags else result

    st.markdown("**Preview**")
    if len(subset) == 0:
        st.info("No transactions were flagged at the current threshold.")
    else:
        st.caption(
            f"Showing {min(preview_limit, len(subset))} of {len(subset):,} rows "
            f"({'flagged only' if only_flags else 'all rows'})."
        )
        st.dataframe(subset.head(int(preview_limit)), use_container_width=True)

    download_choice = st.radio(
        "Download which rows?",
        options=["All rows", "Only flagged"],
        horizontal=True,
    )
    to_download = subset if download_choice == "Only flagged" else result
    st.download_button(
        "Download scored CSV",
        data=to_download.to_csv(index=False).encode(),
        file_name=("fraud_scored_flagged.csv" if download_choice == "Only flagged" else "fraud_scored.csv"),
        mime="text/csv",
    )

#Sidebar: tabs (via radio) + controls
with st.sidebar:
    st.header("Navigation")
    nav = st.radio("Sections", ["Welcome", "Built-in data", "Upload my data"], index=0)

    st.header("Model & Threshold")
    available_models = [p.name for p in MODELS_DIR.glob("*.pkl")]
    if not available_models:
        st.warning("No model files found in ./models")

    model_name = st.selectbox(
        "Choose model",
        options=available_models or ["(no models found)"],
        index=0 if available_models else 0,
    )

    threshold = st.slider(
        "Alert threshold (fraud probability)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Lower → more alerts; Higher → fewer alerts."
    )

    st.header("Display")
    st.checkbox("Show only flagged transactions", value=False, key="only_flags")
    preview_limit = st.number_input("Preview rows", min_value=10, max_value=2000, value=200, step=10)

    #Uploader lives in its tab, but we define the widget here conditionally for layout
    uploaded = None
    if nav == "Upload my data":
        uploaded = st.file_uploader("Upload CSV/XLS/XLSX", type=["csv", "xlsx", "xls"])

#Main content
if not available_models:
    st.info("Waiting for a model… Add a .pkl in ./models, then refresh.")
else:
    try:
        model = load_model(MODELS_DIR / model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

    if model is not None:
        if nav == "Welcome":
            st.markdown("""
### Welcome
This dashboard helps to detect potential **credit card fraud transactions** with a trained ML model.

**How to use:**
1. Go to **Built-in data** to try the dashboard with a sample dataset that is included for demonstration.
2. Or choose **Upload my data** and provide a CSV/XLSX/XLS with the same feature columns used for training.
3. Adjust the **alert threshold** in the sidebar to tune sensitivity.
4. See **KPIs**, preview results, and **download** the scored file.

> Tip: Keep model files under 100 MB. For larger models, load from cloud storage at runtime.
""")
        elif nav == "Built-in data":
            df_default = load_default_dataframe()
            if df_default is None or len(df_default) == 0:
                st.info("Built-in dataset is empty or missing.")
            else:
                score_and_render(df_default, model, threshold, "Built-in data", preview_limit)
        elif nav == "Upload my data":
            if uploaded is None:
                st.info("Upload a CSV/XLSX/XLS using the sidebar to score data.")
            else:
                df_user = read_any_table(uploaded)
                if df_user is None or len(df_user) == 0:
                    st.info("Uploaded file is empty or could not be parsed.")
                else:
                    score_and_render(df_user, model, threshold, "Uploaded file", preview_limit)
