from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Get the directory where app.py is located
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to your pickle file
PKL_PATH = os.path.join(APP_ROOT, 'Sstudent_performance_classifier.pkl')

FEATURE_ORDER_FALLBACK = ['G1','G2','G3','studytime','failures','absences','traveltime','freetime']

FORM_TO_MODEL = {
    'MID-I Marks(0-15)': 'G1',
    'MID-II Marks(0-15)': 'G2',
    'Semister Marks(0-15)': 'G3',
    'Studytime(1-5)': 'studytime',
    'Failed(out of 3 exams)': 'failures',
    'Absents(0-40)': 'absences',
    'Traveltime(1-5)': 'traveltime',
    'Freetime(1-5)': 'freetime'
}

FORM_FIELDS = list(FORM_TO_MODEL.keys())

# ----------- Helper Text Functions (Preserved from original) -----------
def improvement_text(g1, g2, g3):
    lines = []
    if g2 > g1:
        lines.append("Improved from Initial → Midterm")
    elif g2 < g1:
        lines.append("Declined from Initial → Midterm")
    else:
        lines.append("No change from Initial → Midterm")

    if g3 > g2:
        lines.append("Improved from Midterm → Final")
    elif g3 < g2:
        lines.append("Declined from Midterm → Final")
    else:
        lines.append("No change from Midterm → Final")

    if g1>g3:
        lines.append("Overall change from Initial → Final : Improved")
    else:
        lines.append("Overall change from Initial → Final : Not Improved")

    return lines

def detect_weaknesses(g1, g2, g3, studytime, failures, absences, traveltime, freetime):
    weaknesses = []
    if studytime < 2:
        weaknesses.append("Low studytime")
    if absences > 5:
        weaknesses.append("High absences")
    if failures > 0:
        weaknesses.append("Has previous failures")
    if g3 < g2:
        weaknesses.append("Dropped performance from Midterm → Final")
    if freetime > 3:
        weaknesses.append("Too much freetime")
    if traveltime > 2:
        weaknesses.append("Long travel time")
    if g1 <= 5:
        weaknesses.append("Weak initial performance")
    if not weaknesses:
        weaknesses.append("No major weaknesses identified")
    return weaknesses

def generate_suggestions(g1, g2, g3, studytime, failures, absences, traveltime, freetime):
    suggestions = []
    if studytime < 2:
        suggestions.append("Increase daily study to 2–3 hrs.")
    if absences > 5:
        suggestions.append("Improve attendance.")
    if failures > 0:
        suggestions.append("Revise weak topics weekly.")
    if freetime > 3:
        suggestions.append("Reduce unproductive time.")
    if traveltime > 2:
        suggestions.append("Use commute time for microlearning.")
    if g3 < 10:
        suggestions.append("Practice chapter tests regularly.")
    if g2 > g1 and g3 >= g2:
        suggestions.append("Great improvement — keep going!")
    if not suggestions:
        suggestions.append("Excellent — continue same habits.")
    return suggestions

# ----------- Load Model -----------
if not os.path.exists(PKL_PATH):
    print(f"ERROR: Model file not found at {PKL_PATH}")
    model = None
    scaler = None
    features_model = FEATURE_ORDER_FALLBACK
else:
    raw = joblib.load(PKL_PATH)
    if isinstance(raw, dict):
        model = raw.get('model') or raw.get('clf') or raw.get('classifier')
        scaler = raw.get('scaler')
        features_model = raw.get('features') or FEATURE_ORDER_FALLBACK
    else:
        model = raw
        scaler = None
        features_model = FEATURE_ORDER_FALLBACK

features_model = list(features_model)

# Initialize Flask
# IMPORTANT: template_folder='templates' matches your folder structure
app = Flask(__name__, template_folder='templates')
app.secret_key = "secret-key"

# ----------- Routes -----------

@app.route('/')
def index():
    # Looks for 'index.html' inside the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model file not loaded properly."})

    vals_model = {feat: 0.0 for feat in features_model}

    for form_name, model_key in FORM_TO_MODEL.items():
        raw_val = request.form.get(form_name) or "0"
        vals_model[model_key] = float(raw_val)

    row_df = pd.DataFrame([[vals_model[f] for f in features_model]], columns=features_model)

    if scaler:
        X_in = scaler.transform(row_df)
    else:
        X_in = row_df.values.astype(float)

    pred = model.predict(X_in)[0]

    # Return result as JSON
    return jsonify({
        "prediction": str(pred)
    })

# ---------- Run App ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
