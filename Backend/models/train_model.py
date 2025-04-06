import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from config.config import DATASET_PATH, MODEL_PATH

def load_and_prepare_data():
    # Load and preprocess dataset 
    df = pd.read_csv(DATASET_PATH)
    if df['Crop_Type'].dtype == 'object':
        # Updated: Do not drop the first dummy to ensure both crop types (e.g., Tomato and Capsicum) are preserved
        df = pd.get_dummies(df, columns=['Crop_Type'], drop_first=False)
    return df

def train_model(df):
    # Train and evaluate the machine learning model 
    crop_columns = [col for col in df.columns if col.startswith('Crop_Type_')]
    feature_columns = ['Ph', 'EC', 'P', 'K'] + crop_columns
    target_columns = ['Urea', 'TSP', 'MOP', 'Compost']

    X = df[feature_columns]
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    model = MultiOutputRegressor(rf)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    for i, target in enumerate(target_columns):
        mse = mean_squared_error(y_test[target], y_pred[:, i])
        r2 = r2_score(y_test[target], y_pred[:, i])
        print(f"Target: {target}, MSE: {mse:.3f}, R2 Score: {r2:.3f}")

    return model

def save_model(model):
    # Save the trained model 
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved at {MODEL_PATH}")

if __name__ == '__main__':
    df = load_and_prepare_data()
    model = train_model(df)
    save_model(model)
