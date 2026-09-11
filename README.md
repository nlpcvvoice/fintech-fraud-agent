# Fintech Fraud Detection Agent

AI-assisted risk and fraud-detection pipeline for card-transaction data. Trains a
calibrated XGBoost fraud model from the public ULB credit-card fraud dataset,
measures its calibrated probability quality, and exposes the data-fetch layer as a
reusable tool for an agentic workflow.

## What it does

1. Pulls the ULB credit-card fraud transaction set from BigQuery into a pandas
   DataFrame (risk features, `amount`, `class` label).
2. Splits the data with stratification so the rare-event (fraud) ratio is preserved
   across train, calibration, and test partitions.
3. Trains an XGBoost classifier with automatic class-imbalance weighting
   (`scale_pos_weight`) derived from the observed fraud rate.
4. Applies an isotonic probability-calibration layer on an isolated calibration split
   to produce well-calibrated fraud probabilities (avoiding leakage from training).
5. Scores raw vs. calibrated Brier score plus a full classification report
   (precision / recall / F1 for the Legit and Fraud classes).
6. Persists the calibrated model wrapper as a reusable asset.

The data-fetch function (`fetch_ulb_fraud_data`) is written as a tool so an agent can
call it as a step, then hand the DataFrame to the modeling pipeline.

## Directory layout

    data/
      GetData_fromULB.py / GetData_fromULB.ipynb   Load the ULB set into BigQuery
    src/
      agent_tools.py      BigQuery data-fetch tool
      main.py             End-to-end pipeline: split -> XGBoost -> isotonic calibration -> metrics
      train_xgb.ipynb     Interactive training + evaluation notebook
      models/
        fraud_xgb_model.json            Base XGBoost booster
        calibrated_fraud_model.pkl      Calibrated model wrapper (reusable asset)

## Requirements

    google-cloud-bigquery
    xgboost
    scikit-learn
    pandas
    numpy

## Run

1. Stage the dataset in BigQuery (see `data/GetData_fromULB.py`), pointing the
   query at the `fraud_detection.ulb_credit_card_fraud` table.
2. Authorize BigQuery credentials in your environment.
3. Train and evaluate:

       python src/main.py

The run prints the raw and calibrated Brier scores and the production classification
profile, then writes `src/models/calibrated_fraud_model.pkl`.

## Notes

- Stratified splits are mandatory for fraud data; the calibration split is isolated
  to keep the isotonic fit independent of both training and test.
- The calibrated wrapper is the artifact intended for downstream use in a risk
  decisioning flow.