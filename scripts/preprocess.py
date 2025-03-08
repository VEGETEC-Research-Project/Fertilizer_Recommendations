import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
          
    # Encode Crop_Type
    label_encoder = LabelEncoder()
    data['Crop_Type'] = label_encoder.fit_transform(data['Crop_Type'])
    
    # Split data
    X = data[['Ph', 'EC', 'P', 'K', 'Crop_Type']]
    y = data[['Urea', 'TSP', 'MOP', 'Compost']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder
