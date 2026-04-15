from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = "Fraud Transaction" if prediction
    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
