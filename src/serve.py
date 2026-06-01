import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

warnings.filterwarnings("ignore")

MODEL_NAME          = "ChurnPredictionModel"
CHURN_STAGE         = "Production"
HIGH_RISK_THRESHOLD = 0.70  # Customers whose probability exceeds the churn threshold become high-risk customers

# Customers for illustration purposes only (two customers selected at random)
# Customer A – Customer has high potential to churn since he/she has:
#   high churn probability due to short tenure, high monthly fees, month-to-month contract, and no security services
# Customer B – Customer has low potential to churn since he/she has:
#   long duration, moderate charges, two-year plan, security services provided.
SAMPLE_CUSTOMERS = pd.DataFrame([
    {
        "tenure": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0,
        "Contract": 0, "InternetService": 1, "PaymentMethod": 0,
        "SeniorCitizen": 0, "PaperlessBilling": 1,
        "TechSupport": 1, "OnlineSecurity": 1
    },
    {
        "tenure": 48, "MonthlyCharges": 45.0, "TotalCharges": 2160.0,
        "Contract": 2, "InternetService": 0, "PaymentMethod": 2,
        "SeniorCitizen": 0, "PaperlessBilling": 0,
        "TechSupport": 0, "OnlineSecurity": 0
    }
])

SAMPLE_LABELS = ["High-risk profile", "Low-risk profile"]


def deploy_and_predict(model, X_test, y_test, feature_names):
    """
    Demonstrates two deployment modes using the registered Production model:

    1. Real-time inference — scores individual customers on demand.
       This mimics what a live API endpoint would do when a customer
       record is submitted for scoring.

    2. Batch inference — scores the entire test set in one pass.
       This mirrors an overnight batch job that scores all customers
       and flags those at risk for a retention campaign the next morning.

    Deployment metrics are logged to MLflow for traceability.

    Args:
        model:         trained model object used as fallback if registry fails
        X_test:        held-out test features for batch scoring
        y_test:        held-out test labels for metric calculation
        feature_names: list of input feature column names

    Returns:
        batch_predictions, batch_probabilities
    """

    print("-" * 50)
    print("Phase 5: Model Deployment & Serving")
    print("-" * 50)

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Churn_Prediction")

    # Import latest Production version of the model from the MLflow registry
    print(f"Loading '{MODEL_NAME}' from registry (stage: {CHURN_STAGE})")
    try:
        production_model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}/{CHURN_STAGE}"
        )
        print("Model loaded successfully from registry\n")
    except Exception as e:
        # If that fails, import the model saved in memory
        print(f"Registry load failed: {e}")
        print("Falling back to in-memory model\n")
        production_model = model

    # --- Real-time prediction demo ---
    print("Real-time inference demo (2 sample customers):")
    print(f"  {'Profile':<22} {'Prediction':<15} {'Churn Prob':<14} {'Recommended Action'}")
    print(f"  {'-' * 70}")

    rt_preds = production_model.predict(SAMPLE_CUSTOMERS)
    rt_probs = production_model.predict_proba(SAMPLE_CUSTOMERS)[:, 1]

    for label, pred, prob in zip(SAMPLE_LABELS, rt_preds, rt_probs):
        outcome = "Likely to churn" if pred == 1 else "Likely to stay"
        action  = "Offer retention incentive" if pred == 1 else "No action required"
        print(f"  {label:<22} {outcome:<15} {prob * 100:.1f}%{'':<9} {action}")

    # --- Batch prediction ---
    # Predict for all test customer records in one go and calculate business KPIs
    print(f"\nBatch scoring: {len(X_test)} customers")
    batch_predictions   = production_model.predict(X_test)
    batch_probabilities = production_model.predict_proba(X_test)[:, 1]

    total      = len(batch_predictions)
    churners   = int(batch_predictions.sum())
    churn_rate = churners / total
    # High risk customers are those with churn probability greater than the threshold value
    high_risk  = int((batch_probabilities > HIGH_RISK_THRESHOLD).sum())

    print(f"  Predicted to churn       : {churners} / {total} ({churn_rate:.1%})")
    print(f"  High risk (prob > {HIGH_RISK_THRESHOLD:.0%})   : {high_risk} customers")

    # Track deployment run metrics into MLflow for continuity monitoring
    with mlflow.start_run(run_name="Deployment_Demo"):
        mlflow.set_tag("run_type", "deployment")

        batch_auc = roc_auc_score(y_test, batch_probabilities)
        batch_acc = accuracy_score(y_test, batch_predictions)

        mlflow.log_metrics({
            "deploy_auc":         round(batch_auc, 4),
            "deploy_accuracy":    round(batch_acc, 4),
            "total_scored":       total,
            "predicted_churners": churners,
            "churn_rate":         round(float(churn_rate), 4),
            "high_risk_count":    high_risk,
        })
        mlflow.log_param("model_uri",  f"models:/{MODEL_NAME}/{CHURN_STAGE}")
        mlflow.log_param("batch_size", total)

        deploy_run_id = mlflow.active_run().info.run_id

    print(f"\nDeployment metrics logged — run ID: {deploy_run_id}")

    # Steps to deploy the model as a REST service using MLflow Server
    print(f"""
To serve this model as a REST API, run in a new terminal:

    mlflow models serve \\
        -m "models:/{MODEL_NAME}/{CHURN_STAGE}" \\
        --port 5001 --no-conda

Example prediction request:

    curl -X POST http://127.0.0.1:5001/invocations \\
        -H "Content-Type: application/json" \\
        -d '{{"dataframe_split": {{
            "columns": {feature_names},
            "data": [[2, 85.5, 171.0, 0, 1, 0, 0, 1, 1, 1]]
        }}}}'
    """)

    print()
    return batch_predictions, batch_probabilities


if __name__ == "__main__":
    print("Run this module via main.py")
