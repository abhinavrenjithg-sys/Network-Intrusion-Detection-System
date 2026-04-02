import os
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Real-Time Fraud Detection Engine", version="1.0.0")

# Lazy loading in real production, but we'll load globally here for simplicity
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_xgboost_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "fraud_scaler.pkl")

model = None
scaler = None

class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.on_event("startup")
def load_assets():
    global model, scaler
    print("Initializing Fraud Prediction Engine...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"CRITICAL ERROR: Models not found at {MODEL_PATH}")
        return
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("XGBoost Model & Scaler successfully injected to RAM.")

@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    """Predicts instantly whether the provided transaction payload is fraudulent."""
    if model is None:
        return {"error": "Model offline."}
        
    # Scale Time and Amount exactly as requested
    scaled_amount = scaler.transform([[transaction.Amount]])[0][0]
    scaled_time = scaler.transform([[transaction.Time]])[0][0]
    
    features = [
        scaled_time, transaction.V1, transaction.V2, transaction.V3, transaction.V4, 
        transaction.V5, transaction.V6, transaction.V7, transaction.V8, transaction.V9, 
        transaction.V10, transaction.V11, transaction.V12, transaction.V13, transaction.V14, 
        transaction.V15, transaction.V16, transaction.V17, transaction.V18, transaction.V19, 
        transaction.V20, transaction.V21, transaction.V22, transaction.V23, transaction.V24, 
        transaction.V25, transaction.V26, transaction.V27, transaction.V28, scaled_amount
    ]
    
    # Needs 2D array for Predict
    features_np = np.array(features).reshape(1, -1)
    
    prediction = int(model.predict(features_np)[0])
    prob = float(model.predict_proba(features_np)[0][1])
    
    # 0 = Legitimate, 1 = Fraud
    return {
        "status": "success",
        "is_fraud": bool(prediction == 1),
        "fraud_confidence": prob,
        "message": "Transaction blocked." if prediction == 1 else "Transaction authorized."
    }

if __name__ == "__main__":
    uvicorn.run("fraud_api:app", host="0.0.0.0", port=8001, reload=True)
