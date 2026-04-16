from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model properly
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features = np.array([input_features])
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        result = "Fraud Transaction"
    else:
        result = "Normal Transaction"
        
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
