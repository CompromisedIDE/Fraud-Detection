from fastapi import FastAPI, HTTPException
import logging
import os
import csv
import datetime
from pydantic import BaseModel
from pydantic import field_validator
import pandas as pd
import joblib
import numpy as np
from src.features import add_engineered_features, scale_data
from src.config import MODEL_PATH, SCALER_PATH

logger = logging.getLogger(__name__)

app=FastAPI(
    title='Fraud Detection API',
    description='API for detecting fraudulent transactions using a trained XGBoost model.',
    version="1.0"
)
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
logger.info("Loading the model and scaler...")
try:
    model=joblib.load(MODEL_PATH)
    scaler=joblib.load(SCALER_PATH)
    logger.info("Model and Scaler loaded successfully")
except FileNotFoundError:
    logger.error("Model or Scaler file not found. Ensure you ran the training script first.")
class Transaction(BaseModel):
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
    @field_validator('Amount')
    @classmethod
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount cannot be negative')
        return v

    @field_validator('Time')
    @classmethod
    def time_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Time cannot be negative')
        return v
def log_prediction(transaction: dict, is_fraud: bool, probability: float):
    log_path = "prediction_log.csv"
    file_exists = os.path.exists(log_path)
    
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "amount", "is_fraud", "fraud_probability_percent"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transaction.get("Amount"),
            is_fraud,
            round(probability * 100, 2)
        ])
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        input_data=pd.DataFrame([transaction.model_dump()])
        processed_data=add_engineered_features(input_data)
        scaled_data=scaler.transform(processed_data)
        prediction=model.predict(scaled_data)[0]
        probability=model.predict_proba(scaled_data)[0][1]
        is_fraud=bool(prediction==1)
        fraud_probability=float(probability)
        log_prediction(transaction.model_dump(), is_fraud, fraud_probability)
        return {
            "is_fraud": is_fraud,
            "fraud_probability_percent":round(fraud_probability*100, 2),
            "status":"FRAUD ALERT" if is_fraud else "Transaction is Legitimate"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    





