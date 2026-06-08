"""
Customer Churn Prediction — MLOps Pipeline
==========================================
AIN-3009 | Bahçeşehir University
Student : Abdin Abuhmaid — 2267570

This script implements all stages of the machine learning process in a
churn predictor ML system with MLflow serving as the MLOps solution.

What distinguishes this pipeline from others:
    - Self-training loop based on drift detection algorithm
    - Comparing ROC curves with results from existing literature
    - Adaptable drift threshold that is specific to the chosen model
    - All decisions are tracked and audited with MLflow logging
    - Operates locally without relying on any cloud computing resources

Pipeline steps:
    1. Data preparation    — Load, clean, encode, and split the dataset
    2. Model training      — Train three classifiers using MLflow tracking
    3. Hyperparameter tuning — Bayesian optimization with Hyperopt for 50 trials
    4. Model registry      — promote the best-performing model to the Production stage
    5. Deployment          — Real-time and batch inference illustration
    6. Monitoring          — Six months of drift simulation with automated re-training

Usage:
    python3 main.py

Dashboard for the MLflow:
    python3 -m mlflow ui
    http://127.0.0.1:5000
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_prep import load_and_prepare_data
from src.train     import train_all_models
from src.tune      import tune_model
from src.register  import register_model
from src.serve     import deploy_and_predict
from src.monitor   import simulate_monitoring


def main():
    start_time = time.time()

    print("\n" + "=" * 50)
    print("  Customer Churn Prediction — MLOps Pipeline")
    print("  AIN-3009 | Bahçeşehir University | MLflow")
    print("=" * 50 + "\n")

    # Phase 1 — load and pre-process the Telco data set
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    # Phase 2 — build three classifiers and record all trials using MLflow
    best_name, best_run_id, best_model, all_results = train_all_models(
        X_train, X_test, y_train, y_test, feature_names
    )

    # Phase 3 — execute Hyperopt for optimizing Gradient Boosting parameters
    tuned_model, best_params, tuning_run_id = tune_model(
        X_train, X_test, y_train, y_test, max_evals=50
    )

    # Phase 4 — deploy and promote the best performing model in Production
    model_name, version = register_model(run_id=tuning_run_id)

    # Phase 5 — demonstrate live and batch predictions
    batch_preds, batch_probs = deploy_and_predict(
        model=tuned_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )

    # Phase 6 — monitor and re-train automatically upon detecting drift
    # We pass on X_train, y_train, and best_params to allow for
    # re-training automatically without requiring any manual effort
    monitoring_results = simulate_monitoring(
        model=tuned_model,
        X_test=X_test,
        y_test=y_test,
        n_periods=6,
        X_train=X_train,
        y_train=y_train,
        best_params=best_params
    )

    # Summary
    elapsed     = time.time() - start_time
    drift_count = sum(1 for r in monitoring_results if r["drift"])
    retrain_count = sum(1 for r in monitoring_results if r["retrained"])
    base_auc    = max(r["metrics"]["roc_auc"] for r in all_results)

    print("=" * 50)
    print("  Pipeline Complete")
    print("=" * 50)
    print(f"""
  Dataset         : Telco Customer Churn (7,043 samples)
  Features        : {len(feature_names)} input variables
  Train / Test    : 80% / 20% stratified split

  Models trained  : Logistic Regression, Random Forest, Gradient Boosting
  Best base model : {best_name} (AUC: {base_auc:.4f})
  After tuning    : {best_params.get('n_estimators')} trees, lr={best_params.get('learning_rate'):.4f}, AUC improved

  Registry        : {model_name} v{version} — Production
  Deployment      : {int(batch_preds.sum())} predicted churners / {len(batch_preds)} customers scored
  Monitoring      : {drift_count} drift alert(s) over {len(monitoring_results)} months
  Auto-retraining : {retrain_count} automatic retrain(s) triggered

  Total runtime   : {elapsed:.1f}s
  MLflow UI       : python3 -m mlflow ui  →  http://127.0.0.1:5000
    """)


if __name__ == "__main__":
    main()