from flask import Blueprint, request, jsonify
from models.predict_model import predict_fertilizer
import json

# Create Blueprint for API routes
api = Blueprint('api', __name__)

# /predict route to perform fertilizer prediction based on incoming data
@api.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data sent from ESP32 or client
        data = request.get_json()

        # Required fields for prediction
        required_columns = ['Ph', 'EC', 'P', 'K', 'Crop_Type']

        # Check if all required fields are present in the data
        if not all(col in data for col in required_columns):
            return jsonify({'error': 'Missing required fields'}), 400

        # Call the ML Model for prediction
        prediction = predict_fertilizer(data)

        # Return the prediction as a response
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New route to receive sensor data
@api.route('/sensor_data', methods=['POST'])
def sensor_data():
    try:
        data = request.get_json()  # Get the JSON data from ESP32

        # Ensure required fields are present in the request data
        required_columns = ['Ph', 'EC', 'P', 'K', 'Crop_Type']
        for col in required_columns:
            if col not in data:
                return jsonify({'error': f'Missing required field: {col}'}), 400

        # Call the prediction function with the sensor data
        prediction = predict_fertilizer(data)  # Assuming predict_fertilizer is your function for fertilizer prediction
        
        # Return the prediction results as a response
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
