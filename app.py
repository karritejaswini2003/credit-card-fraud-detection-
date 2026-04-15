"""
CardGuard — Credit Card Fraud Detection
Flask Backend API
Author  : [Your Name] (Roll No. 49, MCA — Aditya Degree & PG College)
Guide   : Sai Vardhan Sir
Dataset : ULB Credit Card Fraud Dataset (Kaggle)
Model   : XGBoost (AUC 0.991) with SMOTE + Stratified K-Fold
"""

import os
import pickle
import threading
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow requests from frontend

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "xgboost_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model  = None
scaler = None

def load_artifacts():
    global model, scaler
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[✓] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] Model not found at {MODEL_PATH}. Run model/train_model.py first.")

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"[✓] Scaler loaded from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"[!] Scaler not found at {SCALER_PATH}. Run model/train_model.py first.")

load_artifacts()

# ─────────────────────────────────────────────
# EMAIL ALERT (Gmail SMTP)
# ─────────────────────────────────────────────
# Set these in environment variables or fill directly for testing:
SENDER_EMAIL    = os.getenv("ALERT_EMAIL",    "your_gmail@gmail.com")
SENDER_PASSWORD = os.getenv("ALERT_PASSWORD", "your_app_password")   # Gmail App Password

def send_email_alert(to_email: str, txn_id: str, amount: float, probability: float, risk: str):
    """Send fraud alert email in a background thread."""
    try:
        subject = f"⚠ Fraud Alert — CardGuard | {txn_id}"
        body = f"""
CardGuard Fraud Detection System
─────────────────────────────────
Transaction ID  : {txn_id}
Amount          : ₹{amount:.2f}
Fraud Probability: {probability*100:.1f}%
Risk Level      : {risk}
Detected At     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ACTION REQUIRED: Please verify this transaction immediately.
If this was not you, contact your bank to block the card.

— CardGuard Automated Alert
        """.strip()

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        print(f"[✓] Email alert sent to {to_email}")
    except Exception as e:
        print(f"[!] Email failed: {e}")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors frontend logic)
# ─────────────────────────────────────────────
def build_features(data: dict) -> np.ndarray:
    """
    Reconstruct the 35-dimensional feature vector used during training.
    Expects keys: features (list[float]), amount (float), time (float)
    """
    raw_features = data.get("features", [0] * 35)

    # Ensure length 35 (pad / truncate)
    if len(raw_features) < 35:
        raw_features = raw_features + [0.0] * (35 - len(raw_features))
    raw_features = raw_features[:35]

    return np.array(raw_features, dtype=np.float32).reshape(1, -1)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the CardGuard frontend."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health-check endpoint used by the frontend status indicator."""
    status = {
        "status"       : "ok",
        "model_loaded" : model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp"    : datetime.now().isoformat()
    }
    return jsonify(status), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Request JSON:
      {
        "features": [float × 35],   // V1–V28 + engineered features
        "amount"  : float,
        "email"   : str  (optional),
        "phone"   : str  (optional),
        "id"      : str  (optional)
      }

    Response JSON:
      {
        "fraud"      : bool,
        "probability": float,
        "risk"       : "LOW" | "MEDIUM" | "HIGH",
        "alert_sent" : {"email": bool, "sms": bool}
      }
    """
    data = request.get_json(force=True)

    # ── Build feature array ──
    X = build_features(data)

    # ── Scale if scaler available ──
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"[!] Scaler transform failed: {e}")

    # ── Predict ──
    if model is not None:
        try:
            probability = float(model.predict_proba(X)[0][1])
        except Exception as e:
            return jsonify({"error": f"Model prediction failed: {e}"}), 500
    else:
        # Model not loaded — return error
        return jsonify({"error": "Model not loaded. Run model/train_model.py first."}), 503

    is_fraud = probability > 0.5
    risk     = "HIGH" if probability > 0.7 else ("MEDIUM" if probability > 0.4 else "LOW")

    amount   = float(data.get("amount", 0))
    email    = data.get("email", "").strip()
    phone    = data.get("phone", "").strip()
    txn_id   = data.get("id", "TXN-UNKNOWN")

    alert_sent = {"email": False, "sms": False}

    # ── Send email alert in background if fraud ──
    if is_fraud and email:
        t = threading.Thread(
            target=send_email_alert,
            args=(email, txn_id, amount, probability, risk),
            daemon=True
        )
        t.start()
        alert_sent["email"] = True

    # ── SMS via Twilio (optional — add credentials below) ──
    # if is_fraud and phone:
    #     send_sms_alert(phone, txn_id, amount, probability, risk)
    #     alert_sent["sms"] = True

    return jsonify({
        "fraud"      : bool(is_fraud),
        "probability": round(probability, 4),
        "risk"       : risk,
        "alert_sent" : alert_sent
    })


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  CardGuard — Fraud Detection API")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
