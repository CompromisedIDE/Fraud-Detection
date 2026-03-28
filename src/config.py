import os
DATA_PATH="creditcard.csv"
MODEL_PATH="models/fraud_model_xgboost.pkl"
SCALER_PATH="models/scaler.pkl"
SEED=42
TEST_SIZE=0.2
MODEL_PARAMS={
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "scale_pos_weight": 1,
    "n_jobs": -1,
    "random_state": SEED
}