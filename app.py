from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-a-secure-key'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"[WARN] model file not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load models (if present)
diabetes_model = load_model('diabetes.pkl')       # expects 8 features
heart_model = load_model('heart.pkl')             # expects 13 features
parkinsons_model = load_model('parkinsons.pkl')   # expects 22 features

PRESCRIPTIONS = {
    'diabetes': "Diet, exercise, monitor glucose, consult clinician.",
    'heart': "Seek cardiology advice, ECG/lipid tests, heart-healthy lifestyle.",
    'parkinsons': "See neurologist, consider physiotherapy and specialist care."
}

def parse_numeric_fields(form, count):
    vals = []
    for i in range(1, count + 1):
        key = f'field{i}'
        if key not in form:
            raise ValueError(f"Missing field: {key}")
        raw = form.get(key, '').strip()
        if raw == '':
            raise ValueError(f"Empty value for {key}")
        try:
            vals.append(float(raw))
        except ValueError:
            raise ValueError(f"Non-numeric value for {key}: {raw}")
    return np.array([vals], dtype=float)

def make_prediction(model, X):
    pred = model.predict(X)
    return bool(int(pred[0]))

# ------------------ Routes ------------------

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

@app.route('/heartdisease')
def heart_page():
    return render_template('heartdisease.html')

@app.route('/parkinsons')
def parkinsons_page():
    return render_template('parkinsons.html')

# IMPORTANT: endpoint name is 'chatbot' so url_for('chatbot') works in templates
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# ---------- API endpoints for ajax forms ----------
@app.route('/api/predict/diabetes', methods=['POST'])
def api_predict_diabetes():
    if diabetes_model is None:
        return jsonify(error="Diabetes model not loaded."), 500
    try:
        X = parse_numeric_fields(request.form, 8)
        pos = make_prediction(diabetes_model, X)
        return jsonify(positive=pos,
                       message="This person has Diabetes" if pos else "No Diabetes detected",
                       prescription=PRESCRIPTIONS['diabetes'])
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error="Prediction failed: " + str(e)), 500

@app.route('/api/predict/heart', methods=['POST'])
def api_predict_heart():
    if heart_model is None:
        return jsonify(error="Heart model not loaded."), 500
    try:
        X = parse_numeric_fields(request.form, 13)
        pos = make_prediction(heart_model, X)
        return jsonify(positive=pos,
                       message="This person has Heart Disease" if pos else "No Heart Disease detected",
                       prescription=PRESCRIPTIONS['heart'])
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error="Prediction failed: " + str(e)), 500

@app.route('/api/predict/parkinsons', methods=['POST'])
def api_predict_parkinsons():
    if parkinsons_model is None:
        return jsonify(error="Parkinsons model not loaded."), 500
    try:
        X = parse_numeric_fields(request.form, 22)
        pos = make_prediction(parkinsons_model, X)
        return jsonify(positive=pos,
                       message="This person has Parkinson's" if pos else "No Parkinson's detected",
                       prescription=PRESCRIPTIONS['parkinsons'])
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error="Prediction failed: " + str(e)), 500

# Optional: classic form POST handlers (render precaution pages)
@app.route('/predictdiabetes', methods=['POST'])
def predictdiabetes_form():
    if diabetes_model is None:
        return render_template('diabetesprecaution.html', output_text="Model not available.")
    try:
        X = parse_numeric_fields(request.form, 8)
        pos = make_prediction(diabetes_model, X)
        msg = "This person has Diabetes" if pos else "This person does not have Diabetes"
        return render_template('diabetesprecaution.html', output_text=msg)
    except Exception as e:
        return render_template('diabetesprecaution.html', output_text=f"Error: {e}")

@app.route('/predictheartdisease', methods=['POST'])
def predictheart_form():
    if heart_model is None:
        return render_template('heartPrecuations.html', output_text="Model not available.")
    try:
        X = parse_numeric_fields(request.form, 13)
        pos = make_prediction(heart_model, X)
        msg = "This person has Heart Disease" if pos else "This person does not have Heart Disease"
        return render_template('heartPrecuations.html', output_text=msg)
    except Exception as e:
        return render_template('heartPrecuations.html', output_text=f"Error: {e}")

@app.route('/predictparkinsons', methods=['POST'])
def predictparkinsons_form():
    if parkinsons_model is None:
        return render_template('parikson_precaution.html', output_text="Model not available.")
    try:
        X = parse_numeric_fields(request.form, 22)
        pos = make_prediction(parkinsons_model, X)
        msg = "This person has Parkinson's" if pos else "This person does not have Parkinson's"
        return render_template('parikson_precaution.html', output_text=msg)
    except Exception as e:
        return render_template('parikson_precaution.html', output_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
