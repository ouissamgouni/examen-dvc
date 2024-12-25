import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score

X_test_scaled = np.load('data/scaled_data/X_test_scaled.npy')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


model = load("models/trained_model.joblib")
predictions = model.predict(X_test_scaled)
mean_squared_error = mean_squared_error(y_test, predictions)
r2_score = r2_score(y_test, predictions)
metrics = {"scores": { "mean_squared_error": mean_squared_error, "r2_score": r2_score}}
scores_path = Path("metrics/scores.json")
scores_path.write_text(json.dumps(metrics))