from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import yaml

GBR = GradientBoostingRegressor()


with open('src/models/params.yaml', 'r') as f:
    parameters = yaml.load(f, Loader=yaml.SafeLoader)


grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 2, n_jobs=-1,verbose=2)

X_train_scaled = np.load("data/scaled_data/X_train_scaled.npy")
y_train = pd.read_csv("data/processed_data/y_train.csv")
y_train = np.ravel(y_train)

grid_GBR.fit(X_train_scaled, y_train)

output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

pickle.dump(grid_GBR.best_params_, output_dir.joinpath("best_params.pkl").open("wb"))
print("Model best params saved successfully.")