# Handwritten Digit Classification — Machine Learning Assignment 2

**Classification of handwritten digits 0–9 using five machine learning algorithms.**

This is an educational starter implementation. Read and understand it, perform
and document your own experiments, and customize the Streamlit interface and
write-up before using it for an academic submission.

## A. Problem statement

Given an 8×8 grayscale pixel intensity representation of a handwritten digit,
predict which of ten digit classes (0–9) the image represents.

## B. Dataset description

- **Dataset:** Optical Recognition of Handwritten Digits (scikit-learn copy of
  the UCI dataset's test set).
- **Raw file:** `raw_data/digits.csv` (included in this repository).
- **Instances:** 1,797.
- **Input features:** 64 numerical pixel values from 0–16.
- **Output:** `target` containing integers 0–9 (ten-class classification).
- **Source:** https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- **Documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- **Train/test split:** stratified 80% / 20%, `random_state=42`.
- **Preprocessing:** StandardScaler within each training pipeline. All models
  share the same train and untouched test sets.

## C. GitHub repository link

Replace this with **your own** repository: `https://github.com/YOUR_USERNAME/YOUR_REPO`

## D. Models used and six evaluation metrics

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF SVM)

For a multi-class target, **Precision, Recall and F1 are macro averages**;
**AUC is macro one-vs-rest** (on class probabilities), and MCC is multi-class MCC.

### Comparison table — held-out test data (360 records)

<!-- METRICS_START -->
Run `python train_models.py` to generate `results/model_comparison.csv`.
<!-- METRICS_END -->

### Model performance observations (write these in your own words)

| Model | Interpretation to complete from your experiment |
|---|---|
| Logistic Regression | Describe precision, recall and any key confusion pairs. |
| Decision Tree | Discuss possible overfitting and its test score. |
| KNN | Discuss the impact of distance/scaling and your chosen k. |
| Naive Bayes | Explain the independence assumption and its observed effect. |
| Support Vector Machine | Discuss its test AUC, F1, and performance against others. |
| **Overall winner** | Choose based on evidence, not just accuracy. |

## E. Streamlit application

Live URL (replace after deploying): `https://YOUR_APP.streamlit.app`

The app provides test CSV upload (or bundled test dataset), model selection,
all six metrics, a ten-class confusion matrix, classification report,
and a visualization of the selected example's 8×8 pixels.

**It never retrains using uploaded test data.** It loads saved trained pipelines.

## F. How to reproduce

```bash
python -m pip install -r requirements.txt
python train_models.py
python run_checks.py
python -m streamlit run app.py
```

Read `HOW_TO_RUN.md` for step-by-step GitHub and deployment instructions.

## G. Repository contents

```text
ML_Assignment_2_GitHub_Ready_Digits/
├── raw_data/
│   └── digits.csv               # Included UCI-derived raw data
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metadata.json
├── results/
│   └── model_comparison.csv
├── train_models.py
├── metrics_utils.py
├── app.py
├── run_checks.py
├── test_data.csv               # Labeled hold-out data for Streamlit
├── requirements.txt
├── HOW_TO_RUN.md
├── LICENSE_DATA.md
├── .gitignore
└── README.md
```

## H. Academic integrity

The original assignment explicitly states that AI tools may be used for learning
support, not direct copy-paste submissions. Customize the model settings, run
and explain your own experiments, and document your work honestly.
