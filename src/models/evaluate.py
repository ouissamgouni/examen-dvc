import pandas as pd 
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score

X_test_scaled = np.load('data/processed_data/X_test_scaled.npy')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


model = pickle.load(Path("models/trained_model.pkl").open("rb"))
predictions = model.predict(X_test_scaled)
mean_squared_error = mean_squared_error(y_test, predictions)
r2_score = r2_score(y_test, predictions)
metrics = {"scores": { "mean_squared_error": mean_squared_error, "r2_score": r2_score}}

output_dir = Path("metrics")
output_dir.mkdir(parents=True, exist_ok=True)

scores_path = output_dir.joinpath("scores.json")
scores_path.write_text(json.dumps(metrics))

pred_output_dir = Path("data/predictions")
pred_output_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(predictions).to_csv(pred_output_dir.joinpath("predictions.csv"), index=False)