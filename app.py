"""Interactive inference and evaluation. Does not retrain any model."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

from metrics_utils import score_predictions

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Digit Classification | ML Assignment 2",
                   page_icon="🔢", layout="wide")


@st.cache_resource(show_spinner="Loading trained classifiers …")
def load_models():
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {name: joblib.load(MODEL_DIR / filename)
              for name, filename in metadata["model_files"].items()}
    return models, metadata


def validate_test_csv(data: pd.DataFrame, feature_names: list[str], target: str):
    mandatory = feature_names + [target]
    missing = [c for c in mandatory if c not in data.columns]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    if len(data) == 0:
        raise ValueError("Uploaded CSV has zero rows.")
    if data[mandatory].isna().any().any():
        raise ValueError("Test CSV contains blank feature or target cells.")
    X = data.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(data[target], errors="raise").astype(int)
    if not set(y.unique()).issubset(set(range(10))):
        raise ValueError("The target must contain digits 0 to 9.")
    return X, y


st.title("🔢 Handwritten Digit Classification — ML Model Comparison")
st.caption("Five classifiers · 64 pixel features · ten classes (0–9) · six evaluation metrics")
st.markdown("""**How to use:** upload a labeled test CSV containing `pixel_0_0` to
`pixel_7_7` and the true `target` column, or use the included held-out example.
Only saved training pipelines are used; the uploaded rows are never used for model fitting.""")

try:
    models, info = load_models()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Trained models are missing or invalid: {exc}")
    st.info("Run `python train_models.py`, and commit the model/ files to GitHub.")
    st.stop()

upload = st.file_uploader("Upload your labeled test_data.csv", type=["csv"])
if upload is None:
    data = pd.read_csv(ROOT / "test_data.csv")
    st.info("Using included held-out test_data.csv. Upload another labeled CSV to change it.")
else:
    try:
        data = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Cannot parse uploaded CSV: {exc}")
        st.stop()

try:
    X, y = validate_test_csv(data, info["features"], info["target_column"])
except (ValueError, TypeError) as exc:
    st.error(f"Invalid test CSV: {exc}")
    st.stop()

st.subheader("Test data overview")
a, b, c = st.columns(3)
a.metric("Test records", f"{len(data):,}")
b.metric("Input features", len(info["features"]))
c.metric("Classes present", len(y.unique()))
with st.expander("Preview uploaded data and class counts"):
    st.dataframe(data.head(10), use_container_width=True)
    st.bar_chart(y.value_counts().sort_index())

rows = []
for name, model in models.items():
    pred = model.predict(X)
    prob = model.predict_proba(X)
    metrics = score_predictions(y, pred, prob, classes=model.classes_)
    rows.append({"Model": name, **metrics})

st.subheader("Model comparison — same test records")
summary = pd.DataFrame(rows).set_index("Model")
st.dataframe(summary.style.format("{:.4f}", na_rep="N/A"), use_container_width=True)
st.caption("Precision, Recall and F1 are macro-averaged. AUC is macro one-vs-rest. "
           "AUC is N/A if the uploaded CSV lacks a class required for computation.")

selection = st.selectbox("Select a model for detailed evaluation", list(models.keys()))
selected_model = models[selection]
selected_pred = selected_model.predict(X)
selected_prob = selected_model.predict_proba(X)
selected_metrics = score_predictions(y, selected_pred, selected_prob,
                                      classes=selected_model.classes_)

st.subheader(f"Detailed evaluation: {selection}")
metric_columns = st.columns(3)
for idx, (metric, val) in enumerate(selected_metrics.items()):
    metric_columns[idx % 3].metric(metric, "N/A" if np.isnan(val) else f"{val:.4f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, selected_pred, labels=info["classes"], ax=ax,
        cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(selection)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, selected_pred, labels=info["classes"],
                                   zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with st.expander("Visualize one 8×8 pixel example"):
    index = st.number_input("Select row number", min_value=0,
                            max_value=len(X)-1, value=0, step=1)
    row = X.iloc[int(index)].to_numpy(dtype=float).reshape(8, 8)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(row, cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"True: {int(y.iloc[int(index)])} | Predicted: {int(selected_pred[int(index)])}")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

st.caption("Dataset source: UCI Optical Recognition of Handwritten Digits "
           "(the 1,797-row copy supplied with scikit-learn). "
           "For learning and demonstration, not production decision-making.")
