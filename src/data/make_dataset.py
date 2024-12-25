import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/raw_data/raw.csv")

target = df['silica_concentrate']
feats = df.drop(['silica_concentrate','date'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)

for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
    output_filepath = os.path.join("data/processed_data", f'{filename}.csv')
    file.to_csv(output_filepath, index=False)