import joblib
import numpy as np


def load_model():
    return joblib.load("models/heart_model.pkl")


def predict_heart_risk(data):

    model = load_model()

    features = np.array([[
        data["age"],
        data["sex"],
        data["cp"],
        data["trestbps"],
        data["chol"],
        data["fbs"],
        data["restecg"],
        data["thalach"],
        data["exang"],
        data["oldpeak"],
        data["slope"],
        data["ca"],
        data["thal"]
    ]])

    print("Features:", features)
    print("Predict:", model.predict(features))
    print("Probabilities:", model.predict_proba(features))

    prediction = 1 - model.predict_proba(features)[0][1]
    return round(float(prediction), 2)