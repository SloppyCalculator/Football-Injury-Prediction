import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Allow requests from frontend

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define feature order based on training dataset
FEATURES = ["speed_w", "age", "Height", "Weight", "YrsRunning", "Gender_Male", "DominantLeg_Right", "Level_Recreational"]

@app.route("/")
def home():
    return "Injury Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input into a format compatible with model
        input_data = pd.DataFrame([[
            float(data["walkingSpeed"]),
            float(data["age"]),
            float(data["height"]),
            float(data["weight"]),
            float(data["yearsRunning"]),
            bool(data["Gender"]),
            bool(data["DominantLeg_Right"]),
            bool(data["Level_Recreational"])
        ]], columns=FEATURES)

        # Get probability of injury risk
        probability = model.predict_proba(input_data)[0][1]  # Get probability of injury
        injury_percentage = round(probability * 100, 2)  # Convert to percentage

        # Convert injury percentage to health percentage
        health_percentage = 100 - injury_percentage  # Higher health = Lower injury risk

        # Improved response messages based on health percentage
        if health_percentage > 70:
            message = "✅ The player is in excellent condition and can play the next match! ⚽💪"
        elif health_percentage > 40:
            message = "⚠ The player is moderately fit. Consider some rest before the next match. 🏋️‍♂️"
        else:
            message = "❌ The player is not fit for the next match. Rest is advised. 🏥❌"

        # Updated response includes health percentage and status
        return jsonify({
            "health_status": message,
            "health_percentage": health_percentage
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True,use_reloader=True)