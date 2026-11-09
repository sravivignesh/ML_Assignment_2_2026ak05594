"""Reproducible training of five classifiers on one UCI-derived dataset.

Run: python train_models.py
Output: joblib pipelines, root test_data.csv, comparison metrics, metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics_utils import score_predictions

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "raw_data" / "digits.csv"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
TARGET = "target"

# Each estimator gets the same training and test examples.
MODEL_CONFIGS = {
    "Logistic Regression": ("logistic_regression.joblib",
                            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    "Decision Tree": ("decision_tree.joblib",
                      DecisionTreeClassifier(max_depth=12, min_samples_leaf=2,
                                             random_state=RANDOM_STATE)),
    "KNN": ("knn.joblib", KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": ("naive_bayes.joblib", GaussianNB(var_smoothing=1e-9)),
    "Support Vector Machine": ("svm.joblib",
                               SVC(kernel="rbf", C=3.0, probability=True,
                                   random_state=RANDOM_STATE)),
}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(RAW_CSV)
    assert raw.shape == (1797, 65), f"Unexpected raw data size: {raw.shape}"
    assert not raw.isna().any().any(), "Unexpected missing values in raw data"
    assert set(raw[TARGET]) == set(range(10)), "Expected digits 0 to 9"

    X = raw.drop(columns=[TARGET])
    y = raw[TARGET].astype(int)
    assert len(X) >= 500 and X.shape[1] >= 12

    # Test data is held out BEFORE any preprocessing is fitted.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    comparison = []
    for name, (filename, estimator) in MODEL_CONFIGS.items():
        # The complete pipeline is saved. Scaling is fitted on training data only.
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        metrics = score_predictions(y_test, predictions, probabilities,
                                    classes=pipeline.classes_)
        joblib.dump(pipeline, MODEL_DIR / filename, compress=3)
        comparison.append({"Model": name, **metrics})
        print(f"{name:<23} Accuracy={metrics['Accuracy']:.4f} "
              f"F1={metrics['F1 (macro)']:.4f} MCC={metrics['MCC']:.4f}")

    scores = pd.DataFrame(comparison)
    scores.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    schema = {
        "dataset": "scikit-learn copy of UCI Optical Recognition of Handwritten Digits",
        "target_column": TARGET,
        "features": X.columns.tolist(),
        "classes": sorted(y.unique().tolist()),
        "n_raw_rows": int(len(raw)),
        "n_features": int(X.shape[1]),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "split": "stratified 80/20, random_state=42",
        "scikit_learn_version": sklearn.__version__,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_files": {name: filename for name,(filename,_) in MODEL_CONFIGS.items()},
        "metric_average": "macro (for precision, recall, F1, ROC-AUC one-vs-rest)",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print("\nFull model comparison on the held-out test dataset:")
    print(scores.round(4).to_string(index=False))
    print("\nSaved: raw_data/digits.csv, test_data.csv, 5 joblib models, "
          "results/model_comparison.csv, model/metadata.json")


if __name__ == "__main__":
    main()
