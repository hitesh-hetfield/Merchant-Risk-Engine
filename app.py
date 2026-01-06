from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import boto3
from pathlib import Path

def load_model_from_s3(bucket_name, model_key, local_path):
    s3 = boto3.client(
        "s3",
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name = os.getenv("AWS_REGION")
    )

    s3.download_file(bucket_name, model_key, local_path)

# MODEL_PATH = "artifacts/churn_model.pkl"
# FEATURE_PATH = "artifacts/feature_names.pkl"

# churn_model = joblib.load(MODEL_PATH)
# feature_names = joblib.load(FEATURE_PATH)

app = FastAPI(title="Merchant Risk Engine")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

BUKCET_NAME = "merchant-risk-engine-models"

MODEL_KEY = "models/churn_model.pkl"
FEATURE_KEY = "models/feature_names.pkl"

MODEL_PATH = ARTIFACTS_DIR / "churn_model.pkl"
FEATURE_PATH = ARTIFACTS_DIR / "feature_names.pkl"

@app.on_event("startup")
def load_artifacts():

    load_model_from_s3(BUKCET_NAME, MODEL_KEY, MODEL_PATH)
    load_model_from_s3(BUKCET_NAME, FEATURE_KEY, FEATURE_PATH)

    global churn_model, feature_names
    churn_model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

@app.get("/")
def health_check():
    return{"status": "API is running"}

class MerchantInput(BaseModel):
    TxFreq: int
    TotalTx90: float
    AvgTx90: float
    Recency: int

LOW_RISK = 0.30
HIGH_RISK = 0.60

def assign_risk_bucket(churn_prob: float) -> str:
    if churn_prob < LOW_RISK:
        return "LOW"
    elif churn_prob < HIGH_RISK:
        return "MEDIUM"
    else:
        return "HIGH"

def recommended_action(risk_bucket: str) -> str:
    if risk_bucket == "LOW":
        return "NAR"
    elif risk_bucket == "MEDIUM":
        return "Monitor Usage"
    else:
        return "Trigger Retention Offer"

@app.post("/predict-churn")
def predict_churn(data: MerchantInput):
    # Convert inpur into Dataframe
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[feature_names] # enforce correct order

    # Predict Probability
    churn_prob = churn_model.predict_proba(input_df)[0][1]

    # Business Logic
    churn_flag = int(churn_prob >= 0.5)
    risk_bucket = assign_risk_bucket(churn_prob)
    action = recommended_action(risk_bucket)

    return {
        "churn_probability": round(churn_prob, 3),
        "churn_flag": churn_flag,
        "risk_bucket": risk_bucket,
        "recommended_action": action
    }

