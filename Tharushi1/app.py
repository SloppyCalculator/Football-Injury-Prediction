from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load the trained model and scaler safely
# Ensure this path points to your saved model
model_path = r"C:\Users\Siribaddana\Downloads\random_forest_model (8).pkl"
scaler_path = r"C:\Users\Siribaddana\Downloads\scaler.pkl"
# Load the model and scaler
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
else:
    print("Error: Model or scaler file not found!")
    model = None
    scaler = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Check the files."})

    try:
        # Get user input from the form
        feature1 = float(request.form.get("height_cm", 0))
        feature2 = float(request.form.get("weight_kg", 0))
        feature3 = float(request.form.get("potential", 0))
        feature4 = float(request.form.get("dribbling", 0))
        feature5 = float(request.form.get("pas", 0))
        feature6 = float(request.form.get("sho", 0))
        feature7 = float(request.form.get("acceleration", 0))
        feature8 = float(request.form.get("stamina", 0))

        # Convert the input features to a NumPy array (reshaping to match the model input)
        features = np.array(
            [[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])

        # Scale the input features to match the training preprocessing
        features_scaled = scaler.transform(features)

        # Make the prediction
        prediction = model.predict(features_scaled)[0]
        predicted_category = ["Low", "Medium", "High"][prediction]

        # Return the result to the HTML template
        return render_template("index.html", prediction=predicted_category)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
