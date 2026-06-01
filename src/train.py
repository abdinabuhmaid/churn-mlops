import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend to plot without requiring a display
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# All executions in this notebook are under one MLflow experiment
EXPERIMENT_NAME = "Churn_Prediction"


def compute_metrics(model, X_test, y_test):
    """
    Runs the model on the test set and returns a dictionary of
    standard classification metrics.

    AUC-ROC is the primary metric because the dataset is imbalanced
    (roughly 26% churn vs 74% no-churn), so accuracy alone is misleading.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of churning

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Generates a confusion matrix plot and saves it as a PNG.
    The file is logged to MLflow as an artifact then removed locally.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"])

    # Label every cell with its corresponding count
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
    """
    Plots feature importances for tree-based models (Random Forest,
    Gradient Boosting). Saves and returns the file path.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # sort descending

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


def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains three classifiers and logs each as a separate MLflow run.

    Models trained:
        - Logistic Regression  (linear baseline)
        - Random Forest        (bagging ensemble)
        - Gradient Boosting    (boosting ensemble, typically best performer)

    Each run logs: parameters, all five metrics, confusion matrix,
    and feature importances where applicable.

    Returns:
        best_name, best_run_id, best_model, all_results
    """

    print("-" * 50)
    print("Phase 2: Model Training & Experiment Tracking")
    print("-" * 50)

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Create the three models with their configurations
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

        # Each model is assigned a unique MLflow run
        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)
            metrics = compute_metrics(model, X_test, y_test)

            # Document hyperparameters and results of evaluating the model
            mlflow.log_params({**params, "model_type": name})
            mlflow.log_metrics(metrics)

            # Store the trained model as an artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train.iloc[:3]
            )

            # Store a plot of the confusion matrix
            cm_path = plot_confusion_matrix(model, X_test, y_test, name)
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)

            # Store feature importance scores (only for tree-based models)
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

    # Display a comparison of all three models side by side
    print("Model Comparison:")
    comparison = pd.DataFrame([
        {"Model": r["name"], **r["metrics"]} for r in results
    ]).sort_values("roc_auc", ascending=False)
    print(comparison.to_string(index=False))

    # Choose the best model with the highest AUC-ROC score
    best = max(results, key=lambda x: x["metrics"]["roc_auc"])
    print(f"\nBest model: {best['name']} (AUC-ROC: {best['metrics']['roc_auc']})\n")

    return best["name"], best["run_id"], best["model"], results


if __name__ == "__main__":
    from data_prep import load_and_prepare_data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    train_all_models(X_train, X_test, y_train, y_test, feature_names)
