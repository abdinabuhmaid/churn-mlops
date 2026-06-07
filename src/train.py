import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "Churn_Prediction"


def compute_metrics(model, X_test, y_test):
    """Returns a dictionary of classification metrics for the given model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Saves a confusion matrix plot and returns the file path."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_feature_importance(model, feature_names, model_name):
    """Saves a feature importance bar chart and returns the file path."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices], rotation=45, ha="right"
    )
    ax.set_title(f"Feature Importances — {model_name}")
    ax.set_ylabel("Importance")
    plt.tight_layout()

    path = f"feature_importance_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_roc_curves(results, y_test, X_test):
    """
    Plots ROC curves for all trained models on one chart.

    The ROC curve shows the tradeoff between catching real churners
    (true positive rate) and falsely flagging loyal customers
    (false positive rate). A curve that hugs the top-left corner
    is better — that is what AUC measures.

    We also plot the literature benchmark of AUC 0.845 (Sung, 2025)
    on the same Telco dataset so the reader can see how our tuned
    Gradient Boosting compares to published research.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1E2761", "#4A90D9", "#10B981"]

    for i, result in enumerate(results):
        model   = result["model"]
        name    = result["name"]
        auc     = result["metrics"]["roc_auc"]
        y_prob  = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{name} (AUC = {auc:.4f})")

    # Literature benchmark line — Sung (2025) on same Telco dataset
    ax.axhline(y=0.845, color="orange", linestyle="--", linewidth=1.5,
               label="Literature baseline AUC = 0.845 (Sung, 2025)")

    # Random guess diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC = 0.50)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models vs Literature Benchmark", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = "roc_curves_comparison.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains three classifiers and logs each run to MLflow.
    Returns the name, run ID, and object of the best-performing model.
    """

    print("-" * 50)
    print("Phase 2: Model Training & Experiment Tracking")
    print("-" * 50)

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = [
        {
            "name":   "Logistic Regression",
            "model":  LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
        },
        {
            "name":   "Random Forest",
            "model":  RandomForestClassifier(
                          n_estimators=100, max_depth=10,
                          min_samples_split=5, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
        },
        {
            "name":   "Gradient Boosting",
            "model":  GradientBoostingClassifier(
                          n_estimators=100, learning_rate=0.1,
                          max_depth=4, random_state=42),
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4}
        }
    ]

    results = []

    for entry in models:
        name   = entry["name"]
        model  = entry["model"]
        params = entry["params"]

        print(f"Training: {name}")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = compute_metrics(model, X_test, y_test)

            mlflow.log_params({**params, "model_type": name})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train.iloc[:3]
            )

            # Log confusion matrix
            cm_path = plot_confusion_matrix(model, X_test, y_test, name)
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)

            # Log feature importances for tree-based models
            if hasattr(model, "feature_importances_"):
                fi_path = plot_feature_importance(model, feature_names, name)
                mlflow.log_artifact(fi_path)
                os.remove(fi_path)

            run_id = mlflow.active_run().info.run_id

        print(f"  AUC-ROC  : {metrics['roc_auc']}")
        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  F1 Score : {metrics['f1_score']}\n")

        results.append({
            "name": name, "run_id": run_id,
            "metrics": metrics, "model": model
        })

    # Print comparison table
    print("Model Comparison:")
    comparison = pd.DataFrame([
        {"Model": r["name"], **r["metrics"]} for r in results
    ]).sort_values("roc_auc", ascending=False)
    print(comparison.to_string(index=False))

    # Plot and log ROC curves for all models in one chart
    with mlflow.start_run(run_name="ROC_Curve_Comparison"):
        mlflow.set_tag("run_type", "roc_analysis")
        roc_path = plot_roc_curves(results, y_test, X_test)
        mlflow.log_artifact(roc_path)
        os.remove(roc_path)
        print("\nROC curve comparison logged to MLflow")

    best = max(results, key=lambda x: x["metrics"]["roc_auc"])
    print(f"\nBest model: {best['name']} (AUC-ROC: {best['metrics']['roc_auc']})\n")

    return best["name"], best["run_id"], best["model"], results


if __name__ == "__main__":
    from data_prep import load_and_prepare_data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    train_all_models(X_train, X_test, y_train, y_test, feature_names)