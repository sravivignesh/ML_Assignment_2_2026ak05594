"""Smoke test: load saved models, predict, and reproduce the six metrics."""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from metrics_utils import score_predictions

ROOT=Path(__file__).resolve().parent
metadata=json.loads((ROOT/'model'/'metadata.json').read_text())
data=pd.read_csv(ROOT/'test_data.csv')
X=data[metadata['features']]
y=data[metadata['target_column']]
expected=pd.read_csv(ROOT/'results'/'model_comparison.csv').set_index('Model')
for name, fname in metadata['model_files'].items():
    pipeline=joblib.load(ROOT/'model'/fname)
    pred=pipeline.predict(X)
    proba=pipeline.predict_proba(X)
    result=score_predictions(y,pred,proba,pipeline.classes_)
    for metric, val in result.items():
        assert np.isclose(val, expected.loc[name,metric],equal_nan=True), (name,metric)
    print(f'PASS {name}: {len(pred)} predictions; all six metrics verified')
print('SUCCESS: All five models and CSV artifacts are ready.')
