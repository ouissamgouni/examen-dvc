import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle

GBR = GradientBoostingRegressor()

parameters = {'learning_rate': [0.01,0.02,0.03],
                  'subsample'    : [0.9, 0.5, 0.2],
                  'n_estimators' : [100,500],
                  'max_depth'    : [4,6,8]
                 }

grid_GBR = GridSearchCV(estimator=GBR, param_grid = parameters, cv = 2, n_jobs=-1,verbose=2)

X_train_scaled = np.load("data/scaled_data/X_train_scaled.npy")
y_train = pd.read_csv("data/processed_data/y_train.csv")
y_train = np.ravel(y_train)

grid_GBR.fit(X_train_scaled, y_train)

pickle.dump(grid_GBR.best_params_, open("models/best_params.pkl", "wb"))
print("Model best params saved successfully.")