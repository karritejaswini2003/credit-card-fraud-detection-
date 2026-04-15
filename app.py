import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join("model", "model.pkl")
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Prediction will not work.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        features = [float(x) for x in request.form.values()]
        input_array = np.array(features).reshape(1, -1)

        if model is None:
            return render_template("index.html", prediction_text="Model not loaded. Please check server logs.")

        prediction = model.predict(input_array)[0]
        result = "FRAUD DETECTED 🚨" if prediction == 1 else "Transaction is LEGITIMATE ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction), "result": "fraud" if prediction == 1 else "legitimate"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
