# Merchant Risk Engine
### Churn Risk & Retention Decision System for Fintech Platforms

---

## Overview

Fintech platforms such as Razorpay and Paytm depend heavily on active merchants for transaction volume and revenue. When merchants silently reduce usage or stop transacting altogether, platforms often detect churn **after the damage is already done**.

**Merchant Risk Engine** is an end-to-end machine learning system designed to proactively assess merchant churn risk based on recent transaction behavior and expose actionable insights via a production-ready API.

The system bridges the gap between ML predictions and real business decisions by translating risk scores into **clear retention actions**.

---

## Problem Statement

Merchant churn is costly and reactive detection limits the effectiveness of retention strategies.

### Key challenges:
- Churn is often identified only after activity drops to zero
- Blanket retention offers increase costs and reduce ROI
- Business teams need prioritized, explainable risk signals

### Objectives of this system:
- Predict churn risk **before** merchants disengage
- Segment merchants into actionable risk tiers
- Enable cost-aware, targeted retention strategies

---

## Solution Overview 

Merchant Risk Engine implements a **decision-oriented ML pipeline**:

1. A classification model estimates churn probability
2. Probabilities are mapped into business-defined risk buckets
3. A FastAPI service exposes predictions via a REST API
4. Downstream systems consume predictions to trigger actions

The focus is not just prediction accuracy, but **decision enablement**.

--- 

## System Architecture


### Training and Modeling
- Feature engineering on historical transaction data
- Logistic Regression model trained and evaluated
- Model selection driven by recall to minimize false negatives
- Model and feature schema serialized for reuse

### Inference and Serving 
- Model loaded once at API startup
- API accepts real-time merchant inputs
- Outputs churn probability, risk category, and recommended action
- Stateless and scalable design


---

## Features Used


The model evaluates merchant behavior using the following signals:
- **Transaction Frequency (TxFreq)** – number of transactions
- **Total Transaction Value (90 days)** - `TotalTx90`
- **Average Transaction Value (90 days)** - `AvgTx90`
- **Recency** – days since last transaction 

These features capture both engagement intensity and behavioral decay.

---

## Risk Bucketing Logic

Raw probabilities are converted into **business-friendly risk tiers**:

| Churn Probability | Risk Bucket | Recommended Action |
|------------------|-------------|--------------------|
| < 0.30 | LOW | No Action Required |
| 0.30 – 0.60 | MEDIUM | Monitor usage |
| > 0.60 | HIGH | Trigger retention offer |

This abstraction allows business teams to tune retention aggressiveness **without retraining the model**.

--- 

## API Usage

### Endpoint

POST /predict-churn

### Sample Request
```json
{
  "TxFreq": 3,
  "TotalTx90": 1200,
  "AvgTx90": 400,
  "Recency": 12
}

### Sample Response
{
  "churn_probability": 0.403,
  "churn_flag": 0,
  "risk_bucket": "MEDIUM",
  "recommended_action": "Monitor usage"
}

--- 

## Tech Stack
- Python
- Scikit-learn
- Pandas
- FastAPI
- Pydantic
- Joblib

--- 

## How to Run Locally
1. Install Dependencies 
$ pip install -r requirements.txt

2. Start the API
$ uvicorn app:app --reload

3. Open Swagger UI
$ http://127.0.0.1:8000/docs

--- 

## Future Extensions
- Threshold tuning based on retention cost analysis
- UI dashboard for operations teams
- Dockerized deployment for cloud environments
- Expansion to multi-risk scoring (fraud, inactivity, revenue risk)