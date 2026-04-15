# CardGuard — Credit Card Fraud Detection

**MCA Project | Aditya Degree & PG College**
**Guide:** Sai Vardhan Sir | **Roll No.:** 49

---

## Project Structure

```
cardguard_fraud_detection/
│
├── app.py                  ← Flask API server (run this)
├── requirements.txt        ← Python dependencies
├── README.md
│
├── templates/
│   └── index.html          ← CardGuard frontend (served by Flask)
│
└── model/
    ├── train_model.py      ← Train XGBoost + save model/scaler
    ├── xgboost_model.pkl   ← (generated after training)
    ├── scaler.pkl          ← (generated after training)
    └── creditcard.csv      ← (download from Kaggle — see below)
```

---

## Setup Instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset

Download `creditcard.csv` from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it inside the `model/` folder:
```
model/creditcard.csv
```

### Step 3 — Train the Model

```bash
python model/train_model.py
```

This will:
- Engineer features (log_amount, hour, is_high_value, etc.)
- Apply SMOTE for class imbalance
- Run 5-Fold Stratified Cross Validation
- Train XGBoost
- Save `model/xgboost_model.pkl` and `model/scaler.pkl`

Expected AUC: **0.991**

### Step 4 — Run the App

```bash
python app.py
```

Open browser: http://localhost:5000

---

## API Endpoints

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | Serve frontend UI         |
| GET    | `/health`  | API health check          |
| POST   | `/predict` | Run fraud prediction      |

### POST /predict — Request

```json
{
  "features": [-1.36, -0.07, 2.53, 1.38, ...35 values total...],
  "amount": 149.62,
  "email": "user@example.com",
  "phone": "+91XXXXXXXXXX",
  "id": "TXN-123456"
}
```

### POST /predict — Response

```json
{
  "fraud": false,
  "probability": 0.0023,
  "risk": "LOW",
  "alert_sent": { "email": false, "sms": false }
}
```

---

## Email Alerts (Optional)

To enable email alerts for fraud:

1. Use a Gmail account
2. Enable **App Passwords** in Google Account settings
3. Set environment variables before running app.py:

```bash
set ALERT_EMAIL=your_gmail@gmail.com
set ALERT_PASSWORD=your_app_password
python app.py
```

---

## Model Details

| Model    | AUC   | Technique           |
|----------|-------|---------------------|
| XGBoost  | 0.991 | SMOTE + Stratified K-Fold |

**Features Used:** V1–V28 (PCA), log_amount, hour, is_high_value,
is_micro, above_median, Amount_scaled, Time_scaled

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript (Vanilla)
- **Backend:** Python, Flask, Flask-CORS
- **ML:** XGBoost, scikit-learn, imbalanced-learn
- **Dataset:** ULB Kaggle Credit Card Fraud (284,807 transactions)
