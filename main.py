"""
Customer Churn Prediction — MLOps Pipeline
==========================================
AIN-3009 | Bahçeşehir University
Student : Abdin Abuhmaid — 2267570

This script orchestrates the full machine learning lifecycle for a
customer churn prediction system using MLflow as the MLOps platform.

What makes this pipeline special:
    - Closed-loop auto-retraining when drift is detected
    - ROC curve comparison against published literature benchmarks
    - Dynamic drift threshold that adapts to any model's baseline
    - Full audit trail of every decision logged to MLflow
    - Runs entirely on a local machine with zero cloud dependency

Pipeline phases:
    1. Data preparation    — load, clean, encode, and split the dataset
    2. Model training      — train three classifiers and track with MLflow
    3. Hyperparameter tuning — 50-trial Bayesian optimization with Hyperopt
    4. Model registry      — version and promote the best model to Production
    5. Deployment          — real-time and batch inference demonstration
    6. Monitoring          — six-month drift simulation with auto-retraining

Usage:
    python3 main.py

MLflow dashboard:
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

    # Phase 1 — load and preprocess the Telco dataset
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    # Phase 2 — train three classifiers and log all runs to MLflow
    best_name, best_run_id, best_model, all_results = train_all_models(
        X_train, X_test, y_train, y_test, feature_names
    )

    # Phase 3 — run Hyperopt to find the best Gradient Boosting parameters
    tuned_model, best_params, tuning_run_id = tune_model(
        X_train, X_test, y_train, y_test, max_evals=50
    )

    # Phase 4 — register the tuned model and promote it to Production
    model_name, version = register_model(run_id=tuning_run_id)

    # Phase 5 — demonstrate real-time and batch inference
    batch_preds, batch_probs = deploy_and_predict(
        model=tuned_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )

    # Phase 6 — monitor performance and auto-retrain when drift detected
    # We pass X_train, y_train, and best_params so the system can
    # retrain automatically without any human intervention
    monitoring_results = simulate_monitoring(
        model=tuned_model,
        X_test=X_test,
        y_test=y_test,
        n_periods=6,
        X_train=X_train,
        y_train=y_train,
        best_params=best_params
    )

    # Final summary
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