import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import os

# Debugging: Print current working directory and its contents
print("Current Directory:", os.getcwd())
print("Directory Contents:", os.listdir())

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder="templates")


# Define model path
model_path = r"C:\Users\Siribaddana\OneDrive\Desktop\DSGP\Football-Injury-Prediction\Tharushi1\random_forest_model (1).pkl"
# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model with error handling
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e  # Stop execution if model fails to load

@app.route("/")
def home():
    return render_template("index.html")  # Ensure 'templates/index.html' exists

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get request data
        data = request.json
        print("Received Data:", data)  # Debugging

        # Check if 'features' key exists
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        # Convert features into NumPy array
        features = np.array(data["features"]).reshape(1, -1)
        print("Feature Shape:", features.shape)  # Debugging

        # Ensure correct feature length (should match model input size)
        expected_features = 9  # Change this if needed
        if features.shape[1] != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, but got {features.shape[1]}"}), 400

        # Make prediction
        prediction = model.predict(features)
        print("Prediction Result:", prediction)  # Debugging

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("Prediction Error:", str(e))  # Print error in console
        return jsonify({"error": str(e)}), 500  # Return error response

if __name__ == "__main__":
    app.run(debug=True)
