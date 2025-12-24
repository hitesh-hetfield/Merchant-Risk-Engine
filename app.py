from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

MODEL_PATH = "artifacts/churn_model.pkl"
FEATURE_PATH = "artifacts/feature_names.pkl"

churn_model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

app = FastAPI(title="Merchant Churn Prediction API")

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

