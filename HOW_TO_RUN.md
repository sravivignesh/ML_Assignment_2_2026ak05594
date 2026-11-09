# Run this project step by step

## 1. Requirements

- Python 3.11 or newer (Python 3.13 and scikit-learn 1.8 were used to build these artifacts).
- You do not need to download the data. `raw_data/digits.csv` is already included.
- If you retrain with a different scikit-learn release, update deployment requirements to match.

## 2. Open the project folder

```bash
cd ML_Assignment_2_GitHub_Ready_Digits
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip it and run `.venv\Scripts\python.exe` instead.

## 3. Confirm the public source CSV exists

```bash
python -c "import pandas as pd; d=pd.read_csv('raw_data/digits.csv'); print(d.shape); print(d.target.value_counts().sort_index())"
```

Expected shape is `(1797, 65)` — 64 inputs and one target column.

## 4. Train all five classifiers from the same split

```bash
python train_models.py
```

This creates/replaces the five pipeline files in `model/`, `model/metadata.json`,
`test_data.csv`, and `results/model_comparison.csv`.

## 5. Verify the output

```bash
python run_checks.py
```

Every model should print PASS. The test checks model predictions and all six
metrics against the saved comparison CSV.

## 6. Start the local Streamlit app

```bash
python -m streamlit run app.py
```

Use the bundled held-out test set or upload `test_data.csv`. Choose all five
models, check the comparison table, confusion matrix and classification report.
The app never calls `.fit()` or retrains on uploads.

## 7. Push to your own GitHub repository

Create an **empty** GitHub repo and run:

```bash
git init
git add .
git commit -m "Initial classification experiments and Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Alternatively upload the project files using GitHub's Add file > Upload files.
Verify that the CSV, models, app.py, README and requirements.txt were committed.
Do not upload `.venv` or secrets.

## 8. Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io and sign in with GitHub.
2. Click Create app > Deploy a public app from GitHub.
3. Choose your repository, branch `main`, entrypoint `app.py`.
4. Choose a compatible Python version (the same major/minor version used for training, if available).
5. Deploy, then verify that it opens without errors and works on `test_data.csv`.

## 9. Customize the README and submission PDF

- Put **your real** GitHub and Streamlit links in README.md.
- Interpret the model comparison with your own observations, including why
  accuracy alone is insufficient.
- The assignment also asks you to paste complete README text into the final PDF.
- Add your own BITS Virtual Lab execution screenshot to that final PDF; lab steps
  are intentionally excluded from this project.
- Make meaningful changes before submitting: the brief says AI tools are for
  learning support and warns about direct copy-paste submissions.

## Debugging

- Model files missing: run `python train_models.py` and commit `model/*.joblib`.
- Sklearn version mismatch: install the exact version in requirements.txt or retrain.
- Test CSV missing target: it must include a `target` column with digit labels.
- Wrong column names: original inputs are `pixel_0_0` through `pixel_7_7`.
- AUC is N/A: uploaded data must include all classes to compute ten-class macro AUC.
