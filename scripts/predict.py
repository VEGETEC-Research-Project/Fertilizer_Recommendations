import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load('./models/fertilizer_model.pkl')
label_encoder = joblib.load('./models/label_encoder.pkl')

def predict_fertilizers(ph, ec, p, k, crop_type):
    # Prepare the input data as a pandas DataFrame
    input_data = pd.DataFrame({
        'Ph': [ph],
        'EC': [ec],
        'P': [p],
        'K': [k],
        'Crop_Type': [crop_type]
    })

    # Encode Crop_Type using the label encoder
    input_data['Crop_Type'] = label_encoder.transform(input_data['Crop_Type'])

    # Make prediction using the trained model
    predictions = model.predict(input_data)

    # Map the prediction values to their corresponding fertilizer categories
    fertilizer_categories = ['Urea', 'TSP', 'MOP', 'Compost']
    fertilizer_recommendations = {category: value for category, value in zip(fertilizer_categories, predictions[0])}

    # Print the recommendations in the desired format
    print(f"Fertilizer Recommendations for {crop_type}:")
    for category, value in fertilizer_recommendations.items():
        print(f"{category}: {value:.2f} kg/Ac")

def main():
    # Ask how many times 
    num_loops = int(input("How many circles do you need? "))

    # Loop through the user input for the specified number of times
    for _ in range(num_loops):
        print("\nPlease enter the details for the fertilizer recommendation:")

        # Get input values from the user
        ph = float(input("Enter pH: "))
        ec = float(input("Enter EC (Electrical Conductivity): "))
        p = float(input("Enter Phosphorus (P) value: "))
        k = float(input("Enter Potassium (K) value: "))
        crop_type = input("Enter Crop Type (e.g., Tomato, Capsicum): ")

        # Make prediction based on the inputs
        predict_fertilizers(ph, ec, p, k, crop_type)

if __name__ == '__main__':
    main()
