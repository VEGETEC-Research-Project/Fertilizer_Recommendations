import pickle
import pandas as pd
import os
from config.config import MODEL_PATH, DATASET_PATH

def load_model():
    # Load trained model from file 
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def get_expected_feature_columns():
    # Get expected feature columns from dataset 
    df = pd.read_csv(DATASET_PATH)
    if df['Crop_Type'].dtype == 'object':
        df = pd.get_dummies(df, columns=['Crop_Type'], drop_first=True)
    crop_columns = [col for col in df.columns if col.startswith('Crop_Type_')]
    return ['Ph', 'EC', 'P', 'K'] + crop_columns

def predict_fertilizer(input_data):
    # Make predictions using trained model 
    model = load_model()
    expected_columns = get_expected_feature_columns()
    
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['Crop_Type'], drop_first=True)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    prediction = model.predict(input_df)
    
    return {
        'Urea': round(prediction[0][0], 2),
        'TSP': round(prediction[0][1], 2),
        'MOP': round(prediction[0][2], 2),
        'Compost': round(prediction[0][3], 2)
    }
