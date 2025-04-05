from flask import Flask, request, jsonify, render_template, abort, session, redirect, url_for, flash
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import joblib
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone
import bcrypt
import json
import random


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = "https://mmwqewlddsjqydfbekdk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1td3Fld2xkZHNqcXlkZmJla2RrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMzMTE5ODcsImV4cCI6MjA1ODg4Nzk4N30.ff_CuRIRJhPIlUIpxFUY1JdAbMAYR9aXBIq5EP4dGZQ"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests



app.secret_key = os.urandom(24)  # Random secret key for session encryption

# Load pre-trained models
scaler_x = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
pca_model = joblib.load('pca_model (1).pkl')
load_model = joblib.load('SVR_best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')  # Ensure this is a dict
feature_order = joblib.load('feature_order.pkl')  # Ensure correct feature order
standard_scaler_match = joblib.load('standard_scaler_ml.pkl')
match_model = joblib.load('trained_model.pkl')
feature_order_ml = joblib.load('feature_order_ml.pkl')
updated_pca_model = joblib.load('updated_pca_model.pkl')
model = joblib.load('hashan.pkl')
FEATURES_INJURY = ["speed_w", "age", "Height", "Weight", "YrsRunning", "Gender_Male", "DominantLeg_Right", "Level_Recreational"]
performance_model = joblib.load('random_forest_model-1.pkl')
performance_scaler = joblib.load('scaler.pkl')


# Debugging: Print loaded feature order
print("Loaded Feature Order:", feature_order)
print("Feature Count:", len(feature_order))



@app.route('/')
def home():
    print("Accessing the home page")
    if 'user_id' in session:  # Check if user is logged in
        return redirect("/home")  # Redirect to /home if user is logged in
    if not os.path.exists('templates\home.html'):
        return "Error: home.html file not found in templates directory.", 500
    return render_template("home.html")

@app.route('/home')
def homepage():
    if 'user_id' not in session:
        return redirect('/')
    print("Accessing the home page")
    print(session['username'])
    return render_template("index.html", username=session['username'])

@app.route('/logout')
def logout():
    # Remove username from session if it's there
    session.pop('username', None)
    session.pop('user_id', None)
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/sign')  # Add this route
def sign():
    return render_template("sign.html")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    try:
        if request.content_type == "application/json":
            data = request.get_json()
        else:
            data = request.form  # Accept form-encoded data

        username = data.get("fullName")  # Match the form field name
        password = data.get("password")

        # Validate input
        if not username or not password:
            return "<script>alert('Please fill in both Full Name and Password fields.'); window.location.href='/sign';</script>", 400

        # Check if username exists
        existing_user = supabase.table("users").select("*").eq("username", username).execute()
        if existing_user.data:
            return "<script>alert('This username is already taken. Please choose a different one.'); window.location.href='/sign';</script>", 409

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Insert into Supabase
        response = supabase.table("users").insert({
            "username": username,
            "password": hashed_password
        }).execute()

        # Store user info in session
        session['user_id'] = response.data[0]['id']
        session['username'] = username

        return "<script>alert('Registration successful! Welcome aboard!'); window.location.href='/home';</script>", 201

    except Exception as e:
        return "<script>alert('Oops! Something went wrong during registration. Please try again.'); window.location.href='/sign';</script>", 500

@app.route('/loginsubmit', methods=['GET', 'POST'])
def loginsubmit():
    try:
        if request.content_type == "application/json":
            data = request.get_json()
        else:
            data = request.form  # Accept form-encoded data

        username = data.get("username")  
        password = data.get("password")

        # Validate input
        if not username or not password:
            return "<script>alert('Please enter both your username and password.'); window.location.href='/login';</script>", 400

        # Check if username exists
        existing_user = supabase.table("users").select("*").eq("username", username).execute()
        if not existing_user.data:
            return "<script>alert('Username not found. Please check your spelling or register first.'); window.location.href='/login';</script>", 404

        # Verify password
        hashed_password = existing_user.data[0]["password"]
        if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            return "<script>alert('Incorrect password. Please try again.'); window.location.href='/login';</script>", 401
        
        # Store user info in session
        session['user_id'] = existing_user.data[0]['id']
        session['username'] = username

        # User successfully logged in
        return "<script>alert('Login successful! Welcome back!'); window.location.href='/home';</script>", 200

    except Exception as e:
        return "<script>alert('We encountered an error processing your login. Please try again.'); window.location.href='/login';</script>", 500

@app.route('/tl_form')
def tl_form():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('tl_enter.html', username=session['username'])

@app.route('/ml_form')
def ml_form():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('ml_enter.html', username=session['username'])

@app.route('/load')
def load_prediction():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('load.html', username=session['username'])

@app.before_request
def prevent_long_queries():
    """Prevent excessively long GET requests."""
    max_length = 1024  # Limit URL query length
    if request.method == "GET" and len(request.query_string) > max_length:
        abort(413, "Query string too long")  # Prevent excessive GET requests

@app.route('/injury')
def injury_pred():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('injury.html', username=session['username'])

@app.route('/injury_predict', methods=["POST"])
def predict_injury():
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
    ]], columns=FEATURES_INJURY)

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
        user_id = session['user_id']
        created_at = datetime.now(timezone.utc).isoformat()
        datum = {
            "user_id": user_id,
            "match_load": prediction_value,
            "created_at": created_at
        }
        
        response = supabase.table("match_load").insert(datum).execute()
        if not response or not getattr(response, "data", None):
            print("Error inserting data:", response)
            return jsonify({"error": "Database insertion failed"}), 500
        

        # Render the HTML template with the prediction
        return render_template("matchload.html", prediction=prediction_value, error=None, username=session['username'])

    except Exception as e:
        # Log the error for better debugging
        print(f"\nError: {str(e)}")
        return render_template("matchload.html", error=str(e), prediction=None, username=session['username'])
    

@app.route('/predict', methods=['POST'])
def predict():
    print("Started prediction process")
    try:
        data = {}

        # Check request type and extract data accordingly
        if request.content_type == "application/json":
            if not request.is_json:
                return jsonify({"error": "Invalid JSON request"}), 400
            data = request.get_json()
        elif request.content_type in ["application/x-www-form-urlencoded", "multipart/form-data"]:
            data = {key: request.form.get(key, None) for key in feature_order}
        else:
            return jsonify({"error": "Unsupported content type. Use JSON or Form data."}), 400

        print("Received Data:", data)
        
        # Ensure all expected features exist
        data = {key: data.get(key, None) for key in feature_order}
        df = pd.DataFrame([data])
        df.fillna(0, inplace=True)  # Fill missing values with zeros

        # Encode categorical features
        categorical_features = ["Micro-cycle", "Position", "MatchDay"]
        for col in categorical_features:
            if col in df.columns:
                encoder = label_encoders.get(col)
                if encoder:
                    df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                else:
                    df[col] = -1  # Unknown categories

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
        df_pca = np.nan_to_num(df_pca)  # Replace NaN with zeros

        # Make prediction
        updated_df_pca = updated_pca_model.transform(df_pca)
        prediction = load_model.predict(updated_df_pca)

        if prediction is None or len(prediction) == 0:
            return jsonify({"error": "Prediction failed, empty result"}), 500

        final_prediction = float(prediction[0])  # Ensure JSON serializable
        sent_prediction = float(prediction[0])
        # Store in database
        user_id = session['user_id']
        created_at = datetime.now(timezone.utc).isoformat()
        datum = {
            "user_id": user_id,
            "training_load": final_prediction,
            "created_at": created_at
        }
        
        response = supabase.table("training_load").insert(datum).execute()
        if not response or not getattr(response, "data", None):
            print("Error inserting data:", response)
            return jsonify({"error": "Database insertion failed"}), 500
        
        #return jsonify({"prediction": final_prediction})
        return render_template('trainingload.html', prediction=sent_prediction, username=session['username'])
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    
@app.route('/performance')
def performance():
    if 'user_id' not in session:
        return redirect('/')
    return render_template("performance.html", username=session['username'])

@app.route("/predict_performance", methods=["POST"])
def predict_performance():
    if performance_model is None or performance_scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Check the files."})

    try:
        # Get user input
        features = [
            float(request.form.get("height_cm", 0)),
            float(request.form.get("weight_kg", 0)),
            float(request.form.get("potential", 0)),
            float(request.form.get("dribbling", 0)),
            float(request.form.get("pas", 0)),
            float(request.form.get("sho", 0)),
            float(request.form.get("acceleration", 0)),
            float(request.form.get("stamina", 0))
        ]

        # Scale input features
        features_scaled = performance_scaler.transform([features])

        # Make prediction
        prediction = performance_model.predict(features_scaled)[0]
        predicted_category = ["Low", "Medium", "High"][prediction]

        # Render index.html with prediction and user inputs
        return render_template("performance.html", 
                               prediction=predicted_category, 
                               user_input=request.form)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/report", methods=["POST"])
def report():
    try:
        # Extract user input features from the form
        features = [
            float(request.form.get("height_cm", 0)),
            float(request.form.get("weight_kg", 0)),
            float(request.form.get("potential", 0)),
            float(request.form.get("dribbling", 0)),
            float(request.form.get("pas", 0)),
            float(request.form.get("sho", 0)),
            float(request.form.get("acceleration", 0)),
            float(request.form.get("stamina", 0))
        ]

        # Get the predicted category properly
        prediction = request.form["prediction"]

        return render_template("report.html", 
                               prediction=prediction, 
                               features=features)
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/get_training_load_data')
def get_training_load_data():
    # Get the user_id from session (or dynamically pass it here)
    user_id = session.get('user_id', 1)  # Replace with dynamic value or session variable

    # Fetch the last 5 training load records for the given user
    training_loads = supabase.table('training_load') \
        .select('training_load') \
        .eq('user_id', user_id) \
        .order('created_at', desc=True) \
        .limit(5) \
        .execute()

    # Check if we received data
    if training_loads.data:  # Access the 'data' attribute directly
        # Extract the training load values from the response
        data = [record['training_load'] for record in training_loads.data]
        return jsonify(data)
    else:
        # Return an empty array if no data is found
        return jsonify([])


# Route to get match load data (same as above)
@app.route('/get_match_load_data')
def get_match_load_data():
     # Get the user_id from session (or dynamically pass it here)
    user_id = session.get('user_id', 1)  # Replace with dynamic value or session variable

    # Fetch the last 5 training load records for the given user
    match_loads = supabase.table('match_load') \
        .select('match_load') \
        .eq('user_id', user_id) \
        .order('created_at', desc=True) \
        .limit(5) \
        .execute()

    # Check if we received data
    if match_loads.data:  # Access the 'data' attribute directly
        # Extract the training load values from the response
        data = [record['match_load'] for record in match_loads.data]
        return jsonify(data)
    else:
        # Return an empty array if no data is found
        return jsonify([])
    

@app.route('/chat')
def chatbot_interface():
    return render_template('Chatbot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_proxy():
    try:
        response = requests.post(
            "http://localhost:5005/webhooks/rest/webhook",
            json=request.json,
            headers={'Content-Type': 'application/json'}
        )
        return jsonify(response.json())
    except requests.exceptions.RequestException:
        return jsonify({'error': 'Rasa server not responding'}), 500


if __name__ == '__main__':
    app.run(port=4000, debug=True)

