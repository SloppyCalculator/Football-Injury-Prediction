from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load pre-trained models
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")
pca_model = joblib.load("pca_model (1).pkl")
load_model = joblib.load("SVR_best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # Ensure this is a dict
feature_order = joblib.load("feature_order.pkl")  # Ensure correct feature order
standard_scaler_match = joblib.load("standard_scaler_ml.pkl")
match_model = joblib.load("trained_model.pkl")
feature_order_ml = joblib.load("feature_order_ml.pkl")
updated_pca_model = joblib.load("updated_pca_model.pkl")
model = joblib.load("injury_model.pkl")
FEATURES_INJURY = ["speed_w", "age", "Height", "Weight", "YrsRunning", "Gender_Male", "DominantLeg_Right", "Level_Recreational"]


# Debugging: Print loaded feature order
print("Loaded Feature Order:", feature_order)
print("Feature Count:", len(feature_order))

@app.route("/")
def home():
    print("Accessing the home page")
    if not os.path.exists("templates/home.html"):
        return "Error: home.html file not found in templates directory.", 500
    return render_template("home.html")

@app.route('/home')
def homepage():
    print("Accessing the home page")
    return render_template("index.html")

@app.route('/sign')  # Add this route
def sign():
    return render_template("sign.html")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form.get('fullName')
        password = request.form.get('password')
        # Process the form data (e.g., save to database)
        return "Form submitted successfully!"
    return render_template("index.html")  # Render the signup form

@app.route('/tl_form')
def tl_form():
    return render_template('tl_enter.html')

@app.route('/ml_form')
def ml_form():
    return render_template('ml_enter.html')

@app.route('/load')
def load_prediction():
    return render_template('load.html')

@app.before_request
def prevent_long_queries():
    """Prevent excessively long GET requests."""
    max_length = 1024  # Limit URL query length
    if request.method == "GET" and len(request.query_string) > max_length:
        abort(413, "Query string too long")  # Prevent excessive GET requests

@app.route('/injury')
def injury_pred():
    return render_template('injury.html')

@app.route('/injury_predict', methods=["POST"])
def predict_injury():
    print("are you ready?")
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
        ]], columns=FEATURES_INJURY)

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





@app.route('/predict_ml', methods=['POST'])
def predict_ml():
    print("Started")
    print("Raw Form Data:", request.form)
    try:
        data = {}
        # Check request type and extract data accordingly
        if request.content_type == "application/json":
            if not request.is_json:
                abort(400, "Invalid JSON request")
            data = request.get_json()
        
        elif request.content_type in ["application/x-www-form-urlencoded", "multipart/form-data"]:
            data = {key: request.form.get(key, None) for key in feature_order_ml}
        
        else:
            abort(400, "Unsupported content type. Use JSON or Form data.")

        # Debugging: Print raw received data
        print("\nReceived Data:", data)

        # Ensure all expected features exist, defaulting to None if missing
        data = {key: data.get(key, None) for key in feature_order_ml}
        df = pd.DataFrame([data])

        # Fill missing values with zeros
        df.fillna(0, inplace=True)

        # Debugging: Print processed DataFrame
        print("\nProcessed DataFrame before Scaling:\n", df)

        df_scaled = standard_scaler_match.transform(df)
        
        load_prediction = match_model.predict(df_scaled)

        # Check for NaN or invalid values in rescaled prediction
        if load_prediction is None or np.isnan(load_prediction[0]):
            return render_template("matchload.html", error="Invalid prediction value", prediction=None)

        # Debugging: Print final prediction
        prediction_value = float(load_prediction[0])
        print("\nFinal Prediction:", prediction_value)

        # Render the HTML template with the prediction
        return render_template("matchload.html", prediction=prediction_value, error=None)

    except Exception as e:
        # Log the error for better debugging
        print(f"\nError: {str(e)}")
        return render_template("matchload.html", error=str(e), prediction=None)
    

@app.route('/predict', methods=['POST'])
def predict():
    print("Started")
    try:
        data = {}

        # Check request type and extract data accordingly
        if request.content_type == "application/json":
            if not request.is_json:
                return render_template('matchload.html', error="Invalid JSON request")
            data = request.get_json()
        
        elif request.content_type in ["application/x-www-form-urlencoded", "multipart/form-data"]:
            data = {key: request.form.get(key, None) for key in feature_order}
        
        else:
            return render_template('matchload.html', error="Unsupported content type. Use JSON or Form data.")

        # Debugging: Print raw received data
        print("\nReceived Data:", data)

        # Ensure all expected features exist, defaulting to None if missing
        data = {key: data.get(key, None) for key in feature_order}
        df = pd.DataFrame([data])

        # Fill missing values with zeros
        df.fillna(0, inplace=True)

        # Encode categorical features safely
        categorical_features = ["Micro-cycle", "Position", "MatchDay"]
        for col in categorical_features:
            if col in df.columns:
                encoder = label_encoders.get(col)
                if encoder:
                    df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                else:
                    df[col] = -1  # Assign -1 for unknown categories

        # Convert numeric columns to float
        numeric_columns = [
            "Sleep", "Stress", "Fatigue", "Pain", "Wellness", "RPE", "Duration", 
            "Acute", "Chronic", "ACWR", "Total Duration", "TD/min", 
            "Dist MP 20-35W", "Dist MP 35-55W", "Dist MP>55 W", 
            "Distance 14,4-19,8 km/h / min", "Distance 19,8-25 km/h / min", 
            "Distance > 25 km/h / min", "Dist Acc>3 / min", "Dist Dec <-3 / min",  
            "Dist Acc 2-3 / min", "Dist Dec 2-3 / min", 
            "Dist MP 20-35W / min", "Dist MP 35-55W / min", "Dist MP>55 W / min"
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        # Ensure correct feature order
        df = df[feature_order]  

        # Scale input features
        df_scaled = scaler_x.transform(df)

        # Apply PCA transformation
        df_pca = pca_model.transform(df_scaled)

        # Check for NaN values in PCA data
        if np.isnan(df_pca).any():
            df_pca = np.nan_to_num(df_pca)  # Replace NaN with zeros

        # Make prediction using Random Forest
        updated_df_pca = updated_pca_model.transform(df_pca)
        prediction = load_model.predict(updated_df_pca)

        # Validate prediction result
        if prediction is None or len(prediction) == 0:
            return render_template('trainingload.html', error="Prediction failed, empty result")

        # Rescale prediction back to original range
        # prediction_original = scaler_y.inverse_transform(np.array(prediction).reshape(-1, 1))

        # Check for NaN or invalid values in rescaled prediction
        #if prediction_original is None or np.isnan(prediction_original[0]):
            #return render_template('matchload.html', error="Rescaling failed, invalid prediction value")

        final_prediction = float(prediction[0])

        # Render the HTML template with the prediction result
        return render_template('trainingload.html', prediction=final_prediction)

    except Exception as e:
        print(f"\nError: {str(e)}")
        return render_template('trainingload.html', error=str(e))
if __name__ == "__main__":
    app.run(debug=True)
