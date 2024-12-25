import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle


X_train_scaled = np.load('data/scaled_data/X_train_scaled.npy')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_train = np.ravel(y_train)

best_params=pickle.load(open("models/best_params.pkl", "rb"))
GBR = GradientBoostingRegressor(**best_params)

#--Train the model
GBR.fit(X_train_scaled, y_train)

#--Save the trained model to a file
model_filename = 'models/trained_model.joblib'
joblib.dump(GBR, model_filename)
print("Model trained and saved successfully.")

