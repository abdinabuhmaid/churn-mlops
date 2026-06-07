import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import os

warnings.filterwarnings("ignore")

MONITORING_EXPERIMENT = "Churn_Model_Monitoring"
MODEL_NAME            = "ChurnPredictionModel"


def auto_retrain(X_train, y_train, X_test, y_test, best_params, month):
    """
    Automatically retrains the model when drift is detected.

    This is what makes this pipeline a closed-loop system.
    Instead of just alerting and waiting for a human to fix the
    problem, the system fixes itself — it retrains on fresh data,
    registers the new model, and promotes it to Production.

    This is the same pattern used by Netflix, Uber, and Spotify
    in their production ML systems. Most commercial MLOps tools
    like SageMaker and W&B detect drift but require a human to
    trigger retraining manually. This system does it automatically.

    Args:
        X_train, y_train: training data for retraining
        X_test, y_test:   test data for evaluation
        best_params:      hyperparameters from the tuning phase
        month:            the month drift was detected (for logging)

    Returns:
        new_model, new_auc, new_version
    """

    print(f"\n  Auto-retraining triggered by drift in month {month}...")

    mlflow.set_tracking_uri("./mlruns")

    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    with mlflow.start_run(run_name=f"AutoRetrain_Month_{month:02d}", nested=True):
        mlflow.set_tag("run_type",       "auto_retrain")
        mlflow.set_tag("trigger",        f"drift_month_{month}")
        mlflow.set_tag("retrain_reason", "AUC dropped below threshold")

        # Retrain with the best parameters from the tuning phase
        new_model = GradientBoostingClassifier(**best_params)
        new_model.fit(X_train, y_train)

        new_auc = roc_auc_score(
            y_test, new_model.predict_proba(X_test)[:, 1]
        )
        new_acc = accuracy_score(y_test, new_model.predict(X_test))

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "retrain_auc":      round(new_auc, 4),
            "retrain_accuracy": round(new_acc, 4),
        })

        # Save the retrained model as an artifact
        mlflow.sklearn.log_model(
            sk_model=new_model,
            artifact_path="retrained_model",
            input_example=X_test.iloc[:3]
        )

        retrain_run_id = mlflow.active_run().info.run_id

    # Register the retrained model in the MLflow Model Registry
    model_uri = f"runs:/{retrain_run_id}/retrained_model"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    new_version = registered.version

    # Add tags so it is easy to identify this as an auto-retrained version
    client.update_model_version(
        name=MODEL_NAME,
        version=new_version,
        description=(
            f"Auto-retrained model. Triggered by drift detection in month {month}. "
            f"New AUC: {new_auc:.4f}"
        )
    )
    client.set_model_version_tag(MODEL_NAME, new_version, "retrain_trigger", f"month_{month}")
    client.set_model_version_tag(MODEL_NAME, new_version, "auto_retrained",  "True")

    # Promote to Production automatically
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"  Retrained model registered as v{new_version}")
    print(f"  New AUC: {new_auc:.4f}")
    print(f"  Promoted to Production automatically")

    return new_model, new_auc, new_version


def simulate_monitoring(model, X_test, y_test, n_periods=6,
                        X_train=None, y_train=None, best_params=None):
    """
    Simulates n_periods months of post-deployment model monitoring.

    In production, this function would be scheduled to run monthly,
    scoring a fresh batch of incoming customer data and comparing
    performance against the baseline established at deployment time.

    When drift is detected, the auto_retrain function is called
    automatically — no human intervention required. This creates
    a fully closed-loop MLOps pipeline, which is the gold standard
    in production ML systems.

    Two types of drift are simulated each month:
        1. Feature drift  — numeric columns shift gradually
        2. Concept drift  — small fraction of labels flipped

    Args:
        model:       the deployed model to monitor
        X_test:      baseline test features from deployment
        y_test:      baseline test labels from deployment
        n_periods:   number of months to simulate
        X_train:     training data (needed for auto-retraining)
        y_train:     training labels (needed for auto-retraining)
        best_params: tuned hyperparameters (used when retraining)

    Returns:
        list of dicts containing monthly metrics
    """

    print("-" * 50)
    print("Phase 6: Performance Monitoring & Auto-Retraining")
    print("-" * 50)

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(MONITORING_EXPERIMENT)

    # Establish the performance baseline at the time of deployment
    baseline_auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    baseline_acc    = accuracy_score(y_test, model.predict(X_test))
    baseline_f1     = f1_score(y_test, model.predict(X_test), zero_division=0)
    drift_threshold = round(baseline_auc * 0.95, 3)

    print(f"Baseline AUC at deployment : {baseline_auc:.4f}")
    print(f"Drift alert threshold      : {drift_threshold} (5% below baseline)\n")

    results      = []
    active_model = model  # track which model is currently active

    with mlflow.start_run(run_name="Monitoring_6_Months"):
        mlflow.set_tag("run_type", "monitoring")
        mlflow.log_params({
            "monitoring_periods":  n_periods,
            "drift_threshold":     drift_threshold,
            "baseline_auc":        round(baseline_auc, 4),
            "baseline_accuracy":   round(baseline_acc, 4),
            "auto_retrain_enabled": str(X_train is not None)
        })

        for month in range(1, n_periods + 1):

            # Simulate feature drift — noise increases with each month
            X_shifted   = X_test.copy()
            noise_scale = 0.05 * month
            for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                noise = np.random.normal(0, noise_scale, len(X_shifted))
                X_shifted[col] = (X_shifted[col] * (1 + noise)).clip(lower=0)

            # Simulate concept drift — more label flips each month
            y_shifted = y_test.values.copy()
            flip_mask = np.random.rand(len(y_shifted)) < (0.01 * month)
            y_shifted[flip_mask] = 1 - y_shifted[flip_mask]
            y_shifted = pd.Series(y_shifted)

            # Score the currently active model
            y_pred    = active_model.predict(X_shifted)
            y_prob    = active_model.predict_proba(X_shifted)[:, 1]

            month_auc = roc_auc_score(y_shifted, y_prob)
            month_acc = accuracy_score(y_shifted, y_pred)
            month_f1  = f1_score(y_shifted, y_pred, zero_division=0)
            drifted   = month_auc < drift_threshold

            mlflow.log_metrics({
                f"month_{month:02d}_auc":      round(month_auc, 4),
                f"month_{month:02d}_accuracy": round(month_acc, 4),
                f"month_{month:02d}_f1":       round(month_f1, 4),
                f"month_{month:02d}_drift":    int(drifted),
            })

            status = "DRIFT ALERT" if drifted else "OK"
            print(f"  Month {month}: AUC={month_auc:.4f} | "
                  f"Acc={month_acc:.4f} | F1={month_f1:.4f} | {status}")

            # AUTO-RETRAINING — triggered when drift is detected
            retrained_version = None
            if drifted and X_train is not None and best_params is not None:
                mlflow.set_tag(f"drift_month_{month}", "ALERT")
                active_model, new_auc, retrained_version = auto_retrain(
                    X_train, y_train, X_test, y_test, best_params, month
                )
                mlflow.log_metric(f"month_{month:02d}_retrain_auc", round(new_auc, 4))

            results.append({
                "month":     month,
                "auc":       month_auc,
                "acc":       month_acc,
                "f1":        month_f1,
                "drift":     drifted,
                "retrained": retrained_version
            })

        # Build monitoring dashboard chart
        months     = [r["month"] for r in results]
        aucs       = [r["auc"]   for r in results]
        accs       = [r["acc"]   for r in results]
        f1s        = [r["f1"]    for r in results]
        all_months = [0] + months
        all_aucs   = [baseline_auc] + aucs

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Top panel — AUC over time
        ax1.plot(all_months, all_aucs, "b-o", linewidth=2, label="AUC-ROC")
        ax1.axhline(drift_threshold, color="red", linestyle="--",
                    linewidth=1.5, label=f"Alert threshold ({drift_threshold})")
        ax1.fill_between(all_months, drift_threshold, all_aucs,
                         where=[v >= drift_threshold for v in all_aucs],
                         alpha=0.1, color="green", label="Acceptable zone")
        ax1.fill_between(all_months, drift_threshold, all_aucs,
                         where=[v < drift_threshold for v in all_aucs],
                         alpha=0.15, color="red", label="Drift zone")

        # Mark auto-retrain events on the chart
        for r in results:
            if r["retrained"]:
                ax1.axvline(x=r["month"], color="purple", linestyle=":",
                            linewidth=1.5, alpha=0.7)
                ax1.annotate(f"Auto-retrain\nv{r['retrained']}",
                             xy=(r["month"], drift_threshold),
                             xytext=(r["month"] + 0.1, drift_threshold + 0.01),
                             fontsize=8, color="purple")

        ax1.set_xticks(all_months)
        ax1.set_xticklabels(["Deploy"] + [f"M{m}" for m in months])
        ax1.set_ylabel("AUC-ROC")
        ax1.set_title("Model AUC-ROC Over Time (with Auto-Retraining)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Bottom panel — all metrics
        ax2.plot(all_months, all_aucs,                  "b-o", label="AUC-ROC")
        ax2.plot(all_months, [baseline_acc] + accs,     "g-s", label="Accuracy")
        ax2.plot(all_months, [baseline_f1]  + f1s,      "m-^", label="F1 Score")
        ax2.axhline(drift_threshold, color="red", linestyle="--", alpha=0.5)
        ax2.set_xticks(all_months)
        ax2.set_xticklabels(["Deploy"] + [f"M{m}" for m in months])
        ax2.set_ylabel("Score")
        ax2.set_title("All Metrics Over Time")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = "monitoring_dashboard.png"
        plt.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(chart_path)
        os.remove(chart_path)

        monitoring_run_id = mlflow.active_run().info.run_id

    # Print summary
    drift_months    = [r["month"] for r in results if r["drift"]]
    retrain_months  = [r["month"] for r in results if r["retrained"]]

    print(f"\nBaseline AUC    : {baseline_auc:.4f}")
    print(f"Final AUC       : {results[-1]['auc']:.4f}")
    if drift_months:
        print(f"Drift detected  : months {drift_months}")
    if retrain_months:
        print(f"Auto-retrained  : months {retrain_months}")
        print("System self-healed — no human intervention required")
    else:
        print("No drift detected — model performance is stable")
    print(f"Run ID          : {monitoring_run_id}\n")

    return results


if __name__ == "__main__":
    print("Run this module via main.py")