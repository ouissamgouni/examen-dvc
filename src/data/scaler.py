from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import preprocessing

X_train = pd.read_csv("data/processed_data/X_train.csv")
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(pd.read_csv("data/processed_data/X_test.csv"))

output_dir = Path("data/scaled_data")
output_dir.mkdir(parents=True, exist_ok=True)

for data, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
    output_filepath = output_dir.joinpath(f'{filename}.npy')
    np.save(output_filepath, data)