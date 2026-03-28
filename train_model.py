import pandas as pd
import joblib
import logging
import os
import datetime
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from src.config import DATA_PATH, MODEL_PATH, SCALER_PATH, SEED, TEST_SIZE, MODEL_PARAMS
from src.features import add_engineered_features, scale_data


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting fraud detection pipeline...")
try:
    df=pd.read_csv(DATA_PATH)
    logger.info(f"Data loaded successfully! Total transactions: {len(df)}")
except FileNotFoundError:
    logger.error(f"Error: Data file '{DATA_PATH}' not found. Please check the path and try again.")
    exit(1)
logger.info("Adding engineered features...")
df=add_engineered_features(df)
X=df.drop('Class', axis=1)
y=df['Class']
logger.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)
logger.info("Scaling features...")
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
logger.info("Applying SMOTE to handle class imbalance...")
smote=SMOTE(random_state=SEED)
X_train_rebalanced, y_train_rebalanced=smote.fit_resample(X_train_scaled,y_train)
logger.info(f"Original fraud cases in training set: {sum(y_train==1)}")
logger.info(f"Balanced fraud cases in training set: {sum(y_train_rebalanced==1)}")
logger.info("Training the XGBoost model.. ")
model=XGBClassifier(**MODEL_PARAMS)
model.fit(X_train_rebalanced, y_train_rebalanced)
logger.info("Evaluating model performance on test set...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
print(f"  True Negatives  (Legit correctly identified): {tn}")
print(f"  False Positives (Legit flagged as fraud):     {fp}")
print(f"  False Negatives (Fraud missed):               {fn}")
print(f"  True Positives  (Fraud correctly caught):     {tp}")
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(report)
print("="*50)
print(f"  AUC-ROC Score:  {auc:.4f}")
print(f"  Precision:      {precision:.4f}")
print(f"  Recall:         {recall:.4f}")
print(f"  F1 Score:       {f1:.4f}")
print("="*50)

logger.info(f"AUC-ROC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_model_path = f"models/fraud_model_xgboost_{timestamp}.pkl"
versioned_scaler_path = f"models/scaler_{timestamp}.pkl"

joblib.dump(model, versioned_model_path)
joblib.dump(scaler, versioned_scaler_path)
logger.info(f"Versioned model saved to {versioned_model_path}")
logger.info(f"Versioned scaler saved to {versioned_scaler_path}")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
logger.info(f"Default model updated at {MODEL_PATH}")
logger.info(f"Default scaler updated at {SCALER_PATH}")


registry_path = "models/model_registry.json"
registry = []

if os.path.exists(registry_path):
    with open(registry_path, "r") as f:
        registry = json.load(f)

registry.append({
    "timestamp": timestamp,
    "model_path": versioned_model_path,
    "scaler_path": versioned_scaler_path,
    "metrics": {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn)
    }
})

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

logger.info(f"Model registry updated at {registry_path}")
logger.info("Fraud detection pipeline completed successfully!")