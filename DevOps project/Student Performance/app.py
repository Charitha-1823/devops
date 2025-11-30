from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder="templates")
app.secret_key = "super-secret"

# ------------------- MODEL PATH -------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Student_performance_classifier.pkl"
MODEL_PATH = os.path.join(APP_ROOT, MODEL_NAME)

print(f"[INFO] Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")

model_data = joblib.load(MODEL_PATH)

# If model saved as dict
if isinstance(model_data, dict):
    model = model_data.get("model") or model_data.get("classifier")
    scaler = model_data.get("scaler")
    features = model_data.get("features")
else:
    model = model_data
    scaler = None
    features = ['G1','G2','G3','studytime','failures','absences','traveltime','freetime']

print("[INFO] Model loaded successfully!")


# ------------------- DEFAULT EXPLANATIONS -------------------

def generate_explanation(pred, G1, G2, G3, studytime):
    improvements = []
    weaknesses = []
    suggestions = []

    # Performance trend
    if G1 == G2:
        improvements.append("No change from Initial → Midterm")
    else:
        improvements.append("Marks improved from Initial → Midterm" if G2 > G1 else "Drop from Initial → Midterm")

    if G2 == G3:
        improvements.append("No change from Midterm → Final")
    else:
        improvements.append("Marks improved from Midterm → Final" if G3 > G2 else "Drop from Midterm → Final")

    if G1 == G3:
        improvements.append("Overall change from Initial → Final : No change")
    else:
        improvements.append("Overall changed from Initial → Final : Improved" if G3 > G1 else "Overall drop from Initial → Final")

    # Weakness based on model
    if studytime <= 2:
        weaknesses.append("Low studytime")
        suggestions.append("Increase daily study time to 2–3 hours.")

    if G1 < 10:
        weaknesses.append("Weak initial performance.")
        suggestions.append("Practice chapter tests weekly.")

    weaknesses.append("Inconsistency in academic performance.")
    suggestions.append("Follow a timetable regularly.")

    return improvements, weaknesses, suggestions


# ------------------- HOME ROUTE -------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, input_vals=None, features=features)


# ------------------- PREDICT ROUTE (POST ONLY) -------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_vals = {}

        # extract form values
        for f in features:
            raw = request.form.get(f, "0")
            try:
                input_vals[f] = float(raw)
            except:
                input_vals[f] = 0.0

        # Create DataFrame
        row = pd.DataFrame([[input_vals[f] for f in features]], columns=features)

        # scale if scaler exists
        if scaler:
            X = scaler.transform(row)
        else:
            X = row.values

        # prediction
        pred = model.predict(X)[0]
        pred = str(pred)

        # explanation engine
        improvements, weaknesses, suggestions = generate_explanation(
            pred,
            input_vals['G1'],
            input_vals['G2'],
            input_vals['G3'],
            input_vals['studytime']
        )

        result_data = {
            "prediction": pred,
            "improvements": improvements,
            "weaknesses": weaknesses,
            "suggestions": suggestions
        }

        return render_template(
            "index.html",
            result=result_data,
            input_vals=input_vals,
            features=features
        )

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 400


# ------------------- API ENDPOINT (IF REQUIRED) -------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        row = [data.get(f, 0) for f in features]

        df = pd.DataFrame([row], columns=features)
        X = scaler.transform(df) if scaler else df.values
        pred = str(model.predict(X)[0])

        return jsonify({"prediction": pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ------------------- RUN SERVER -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
