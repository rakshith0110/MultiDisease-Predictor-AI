from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-secure-key'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------- MODEL LOADING ----------------------
def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"[WARNING] Model not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

diabetes_model = load_model("diabetes.pkl")
heart_model = load_model("heart.pkl")
parkinsons_model = load_model("parkinsons.pkl")

# ---------------------- MAIN ROUTES ----------------------

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



# ---------------------- PREDICTION PAGES ----------------------

@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

@app.route('/heartdisease')
def heart_page():
    return render_template('heartdisease.html')

@app.route('/parkinsons')
def parkinsons_page():
    return render_template('parkinsons.html')


# ---------------------- PRECAUTION PAGES ----------------------

@app.route('/diabetesprecaution')
def diabetes_precaution():
    return render_template('diabetesprecaution.html')

@app.route('/heartPrecuations')
def heart_precautions():
    return render_template('heartPrecuations.html')

@app.route('/parikson_precaution')
def parkinson_precaution():
    return render_template('parikson_precaution.html')



# ---------------------- PREDICTION LOGIC ----------------------

@app.route('/predictdiabetes', methods=['POST'])
def predictdiabetes():
    try:
        keys = [
            'pregnancies','glucose','bloodpressure','skinthickness',
            'insulin','bmi','dpf','age'
        ]
        data = [float(request.form[k]) for k in keys]
        pred = diabetes_model.predict([data])[0]

        result = "Positive" if pred == 1 else "Negative"
        message = (
            "Model indicates possible Diabetes. Consult a doctor."
            if pred == 1 else
            "No Diabetes detected. Maintain healthy habits."
        )

        return render_template("diabetes.html",
                               diabetes_result=result,
                               diabetes_message=message)

    except Exception as e:
        return render_template("diabetes.html",
                               diabetes_result="Error",
                               diabetes_message=str(e))


@app.route('/predictheartdisease', methods=['POST'])
def predictheartdisease():
    try:
        keys = [
            'age','sex','cp','trestbps','chol','fbs','restecg','thalach',
            'exang','oldpeak','slope','ca','thal'
        ]
        data = [float(request.form[k]) for k in keys]
        pred = heart_model.predict([data])[0]

        result = "Positive" if pred == 1 else "Negative"
        message = (
            "Potential Heart Disease detected. Cardiology tests recommended."
            if pred == 1 else
            "Low heart disease risk detected."
        )

        return render_template("heartdisease.html",
                               heart_result=result,
                               heart_message=message)

    except Exception as e:
        return render_template("heartdisease.html",
                               heart_result="Error",
                               heart_message=str(e))


@app.route('/predictparkinsons', methods=['POST'])
def predictparkinsons():
    try:
        data = [float(request.form[f'f{i}']) for i in range(1, 23)]
        pred = parkinsons_model.predict([data])[0]

        result = "Positive" if pred == 1 else "Negative"
        message = (
            "Parkinson's-like symptoms detected. Please visit a neurologist."
            if pred == 1 else
            "No Parkinson's pattern detected."
        )

        return render_template("parkinsons.html",
                               parkinsons_result=result,
                               parkinsons_message=message)

    except Exception as e:
        return render_template("parkinsons.html",
                               parkinsons_result="Error",
                               parkinsons_message=str(e))


# ---------------------- RUN SERVER ----------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
