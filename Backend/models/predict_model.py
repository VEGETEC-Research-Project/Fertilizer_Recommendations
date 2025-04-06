import pickle
import pandas as pd
import os
from config.config import MODEL_PATH, DATASET_PATH

# Global model variable to load the model only once
model = None

def load_model():
    global model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

def get_expected_feature_columns():
    # Load the dataset to derive expected feature columns
    df = pd.read_csv(DATASET_PATH)
    if df['Crop_Type'].dtype == 'object':
        # Create dummy variables for all crop types (do not drop any)
        df = pd.get_dummies(df, columns=['Crop_Type'], drop_first=False)
    # Get all columns that start with "Crop_Type_"
    crop_columns = [col for col in df.columns if col.startswith('Crop_Type_')]
    return ['Ph', 'EC', 'P', 'K'] + crop_columns

def predict_fertilizer(input_data):
    global model
    if model is None:
        load_model()
    
    # Get expected columns (including crop type dummies)
    expected_columns = get_expected_feature_columns()

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode 'Crop_Type' without dropping any column
    input_df = pd.get_dummies(input_df, columns=['Crop_Type'], drop_first=False)
    
    # Reindex DataFrame to match expected columns (fill missing columns with 0)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Debug print: verify the one-hot encoded data
    # print(f"Input Data after One-Hot Encoding:\n{input_df}")

    # Make prediction using the model
    prediction = model.predict(input_df)

    # Return the predictions rounded to 2 decimal places
    return {
        'Urea': round(prediction[0][0], 2),
        'TSP': round(prediction[0][1], 2),
        'MOP': round(prediction[0][2], 2),
        'Compost': round(prediction[0][3], 2)
    }
