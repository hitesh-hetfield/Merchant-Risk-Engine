# Merchant Churn Prediction Using Transaction Behavior

This project explores how historical transaction behavior can be used to **predict merchant churn** on digital payment platforms.

The focus is on building a clean, leakage-free data pipeline and meaningful behavioral features before applying machine learning models.

---

## Problem Statement

Payment platforms rely on sustained merchant activity to generate revenue. However, many merchants stop transacting without explicit signals, leading to revenue loss and inefficient acquisition spend.

The objective of this project is to identify **early warning signals of merchant churn** by analyzing transaction behavior in the period leading up to inactivity.

Merchant churn is framed as a **binary classification problem**, where the goal is to predict whether a merchant is likely to stop transacting in the near future.

---

## Churn Definition

A merchant is considered **churned** if they show **no transaction activity in the last 30 days** of the available data.

This definition mirrors real-world retention monitoring practices used by payment and SaaS platforms.

---

## Data

- Public dataset: **UCI Online Retail Dataset**
- Dataset is used as a **proxy** for payment platform transaction logs
- `CustomerID` is treated as a merchant identifier
- `InvoiceDate` represents transaction timestamps

The dataset is not included in the repository due to size and licensing constraints.

---

## Methodology

### Time Windows
- **Churn window:** Last 30 days (used only for labeling)
- **Observation window:** 90 days prior to the churn window (used for feature engineering)

This separation prevents data leakage and ensures realistic prediction setup.

### Feature Engineering (Current Scope)
Features are computed at the merchant level using the 90-day observation window:

- **Transaction Frequency:** Number of unique invoices
- **Total Transaction Value:** Aggregate value of transactions
- **Average Transaction Value:** Mean invoice value
- **Recency:** Days since last transaction before churn cutoff

---
