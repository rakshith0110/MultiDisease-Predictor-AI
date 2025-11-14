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
# Load models
diabetes_model = load_model('diabetes.pkl')
heart_model = load_model('heart.pkl')
parkinsons_model = load_model('parkinsons.pkl')

# === FIX: Create model aliases so prediction routes work ===
diabetes_predict = diabetes_model
heart_predict = heart_model
parkinsons_predict = parkinsons_model


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
# ---------- PREDICTION ROUTES (paste after model loads) ----------
from flask import flash  # optional

@app.route('/predictdiabetes', methods=['POST'])
def predictdiabetes():
    try:
        # expected field names from template
        keys = ['pregnancies','glucose','bloodpressure','skinthickness','insulin','bmi','dpf','age']
        data = []
        for k in keys:
            v = request.form.get(k, None)
            if v is None or v == '':
                raise ValueError(f"Missing value for {k}")
            data.append(float(v))
        arr = np.array([data])               # shape (1,8)
        pred = diabetes_predict.predict(arr)
        if int(pred[0]) == 1:
            result = 'Positive'
            message = 'Model indicates presence of Diabetes. Suggest consult a doctor; consider diet control, medication and regular monitoring.'
        else:
            result = 'Negative'
            message = "No Diabetes detected by model. Maintain healthy diet and follow-up if symptoms appear."
        # render predict page and display inline result
        return render_template('predict.html', diabetes_result=result, diabetes_message=message)
    except Exception as e:
        # show the error on the same page (helpful during dev)
        return render_template('predict.html', diabetes_result='Error', diabetes_message=str(e))


@app.route('/predictheartdisease', methods=['POST'])
def predictheartdisease():
    try:
        # 13 features as used in the form
        keys = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        data = []
        for k in keys:
            v = request.form.get(k, None)
            if v is None or v == '':
                raise ValueError(f"Missing value for {k}")
            data.append(float(v))
        arr = np.array([data])   # shape (1,13)
        pred = heart_predict.predict(arr)
        if int(pred[0]) == 1:
            result = 'Positive'
            message = 'Model indicates possible heart disease. Seek cardiology consultation and run recommended tests (ECG, stress-test).'
        else:
            result = 'Negative'
            message = "Low heart disease risk predicted. Keep healthy lifestyle and periodic checkups."
        return render_template('predict.html', heart_result=result, heart_message=message)
    except Exception as e:
        return render_template('predict.html', heart_result='Error', heart_message=str(e))


@app.route('/predictparkinsons', methods=['POST'])
def predictparkinsons():
    try:
        # fields f1..f22 used by template
        keys = [f'f{i}' for i in range(1, 23)]
        data = []
        for k in keys:
            v = request.form.get(k, None)
            if v is None or v == '':
                raise ValueError(f"Missing value for {k}")
            data.append(float(v))
        arr = np.array([data])   # shape (1,22)
        pred = parkinsons_predict.predict(arr)
        if int(pred[0]) == 1:
            result = 'Positive'
            message = "Model indicates Parkinson's-like pattern. Recommend neurology referral for clinical confirmation."
        else:
            result = 'Negative'
            message = "No Parkinson's pattern detected by model. Monitor symptoms and consult if concerns arise."
        return render_template('predict.html', parkinsons_result=result, parkinsons_message=message)
    except Exception as e:
        return render_template('predict.html', parkinsons_result='Error', parkinsons_message=str(e))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
