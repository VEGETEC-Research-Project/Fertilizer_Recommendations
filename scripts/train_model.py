import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor
from preprocess import preprocess_data


def train_model():
    # Preprocess the data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data('./data/Cleaned_Fertilizer_Rec.csv')
    
    # Train a RandomForest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

     # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    accuracy = r2 * 100
    print(f"Model Accuracy: {accuracy:.2f}%")
    
    # Save the model and label encoder
    joblib.dump(model, './models/fertilizer_model.pkl')
    joblib.dump(label_encoder, './models/label_encoder.pkl')
    
    print("Model trained and saved!")

if __name__ == '__main__':
    train_model()
