




import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your model (ensure the file name is correct)
with open("hashan.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "speed_w", "age", "Height", "Weight", "YrsRunning",
    "Gender_Male", "DominantLeg_Right", "Level_Recreational"
]

@app.route("/")
def home():
    return "Injury Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    # 1) Get JSON data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    # 2) Try to parse and convert all fields
    try:
        # Debug print to see raw data
        print("Raw data from frontend:", data)

        # Convert to correct types
        speed_w = float(data["walkingSpeed"])
        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])
        yrs_running = float(data["yearsRunning"])

        # Convert booleans to 1/0
        gender_male = 1 if data["Gender"] else 0
        dominant_right = 1 if data["DominantLeg_Right"] else 0
        level_recreational = 1 if data["Level_Recreational"] else 0

    except KeyError as e:
        # Missing a key in JSON
        print("Missing key:", e)
        return jsonify({"error": f"Missing key: {e}"}), 400
    except ValueError as e:
        # Could not convert string to float/int
        print("Value conversion error:", e)
        return jsonify({"error": f"Value conversion error: {e}"}), 400
    except Exception as e:
        # Catch-all for anything else
        print("Unexpected error:", e)
        return jsonify({"error": str(e)}), 400

    # 3) Now that all variables are defined, construct DataFrame
    input_data = pd.DataFrame([[
        speed_w, age, height, weight, yrs_running,
        gender_male, dominant_right, level_recreational
    ]], columns=FEATURES)

    # Debug: Inspect DataFrame
    print("Constructed input DataFrame:")
    print(input_data)
    print("DataFrame dtypes:")
    print(input_data.dtypes)

    # 4) Predict with your model
    probability = model.predict_proba(input_data)[0][1]
    injury_percentage = round(probability * 100, 2)
    health_percentage = 100 - injury_percentage

    # 5) Build a response message
    if health_percentage > 70:
        message = "✅ The player is in excellent condition and can play the next match! ⚽💪"
    elif health_percentage > 40:
        message = "⚠ The player is moderately fit. Consider some rest before the next match. 🏋️‍♂️"
    else:
        message = "❌ The player is not fit for the next match. Rest is advised. 🏥❌"

    # 6) Return final JSON response
    response = {
        "health_status": message,
        "health_percentage": health_percentage
    }
    print("Response:", response)  # Debug
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
