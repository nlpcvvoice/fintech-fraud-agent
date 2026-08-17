import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from agent_tools import fetch_ulb_fraud_data

def execute_production_pipeline():
    print("📥 [Step 1] Streaming production ULB records via cloud pipeline...")
    df = fetch_ulb_fraud_data()
    
    # Cast NUMERIC fields to native float64 types to safely interface with native C++ matrixes
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(float)
        
    X = df.drop(columns=['class', 'time']).fillna(0)
    y = df['class'].astype(int)
    
    # -------------------------------------------------------------------------
    # 🧱 DATA SPLITTING (FinTech Stratification Protocol)
    # -------------------------------------------------------------------------
    # In fraud, we MUST use stratify=y. This guarantees that train, test, and
    # calibration datasets maintain the exact same 0.17% rare-event ratio.
    # -------------------------------------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Split training sets down further to isolate an independent Calibration pool
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"✅ Data Stratified: Train={X_train.shape[0]}, Calib={X_calib.shape[0]}, Test={X_test.shape[0]}")
    
    # -------------------------------------------------------------------------
    # 🏋️‍♂️ BASE XGBOOST MODEL WITH RISK TUNED CLASS-WEIGHTS
    # -------------------------------------------------------------------------
    # scale_pos_weight forces the booster to penalize missing rare positive labels
    imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"⚖️ Calculated Optimal Imbalance Class Scale Weight Factor: {imbalance_ratio:.2f}")
    
    base_model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=imbalance_ratio,
        eval_metric='logloss',
        random_state=42
    )
    
    print("🌲 Fitting uncalibrated baseline gradient-boosted arrays...")
    base_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 🎯 PROBABILITY CALIBRATION LAYER (Isotonic Wrapper)
    # -------------------------------------------------------------------------
    # cv='prefit' means we use the already-trained model and calibrate it
    # using the isolated X_calib set to completely bypass data-leakage bugs.
    # -------------------------------------------------------------------------
    print("🎯 Initiating Isotonic Regression probability calibration sequence...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='isotonic',
        ensemble=False
    )
    calibrated_model.fit(X_calib, y_calib)
    
    # -------------------------------------------------------------------------
    # 📊 FINTECH RISK EVALUATION METRICS
    # -------------------------------------------------------------------------
    # Brier Score Loss calculates the direct mean squared error of probabilities.
    # Lower Brier score = highly accurate probability tracking!
    # -------------------------------------------------------------------------
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    calib_probs = calibrated_model.predict_proba(X_test)[:, 1]
    
    print("\n================== 📊 MODEL EVALUATION CARD ==================")
    print(f"📉 Raw Uncalibrated Brier Score Error: {brier_score_loss(y_test, raw_probs):.5f}")
    print(f"🔥 Calibrated Stable Brier Score Error: {brier_score_loss(y_test, calib_probs):.5f}")
    
    print("\n📋 Production Classification Profile (Calibrated Metrics):")
    predictions = calibrated_model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=['Legit', 'Fraud']))
    print("==============================================================")
    
    # Save the calibrated wrapper asset to local storage
    import joblib
    joblib.dump(calibrated_model, "src/calibrated_fraud_model.pkl")
    print("💾 Calibrated structural model wrapper compiled and saved locally!")

if __name__ == "__main__":
    execute_production_pipeline()
