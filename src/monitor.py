import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
import os

warnings.filterwarnings("ignore")

MONITORING_EXPERIMENT = "Churn_Model_Monitoring"


def simulate_monitoring(model, X_test, y_test, n_periods=6):
    """
    Simulates n_periods months of post-deployment model monitoring.

    In production, this function would be scheduled to run monthly,
    scoring a fresh batch of incoming customer data and comparing
    performance against the baseline established at deployment time.

    Here we simulate drift by introducing two types of degradation:
        1. Feature drift  — numeric columns shift slightly each month,
                            mimicking gradual changes in customer behavior
        2. Concept drift  — a small fraction of labels are flipped each month,
                            representing a change in what actually causes churn

    The drift alert threshold is set at 5% below the deployment baseline AUC.
    If performance drops below this threshold, the model is flagged for retraining.

    All monthly metrics are logged to a dedicated MLflow experiment and a
    dashboard chart is saved as an artifact for visual inspection.

    Args:
        model:     the deployed model to monitor
        X_test:    baseline test features from the deployment period
        y_test:    baseline test labels from the deployment period
        n_periods: number of months to simulate (default 6)

    Returns:
        list of dicts containing monthly metrics
    """

    print("-" * 50)
    print("Phase 6: Performance Monitoring")
    print("-" * 50)

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(MONITORING_EXPERIMENT)

    # Set up the baseline performance at the moment of deployment
    baseline_auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    baseline_acc    = accuracy_score(y_test, model.predict(X_test))
    baseline_f1     = f1_score(y_test, model.predict(X_test), zero_division=0)
    drift_threshold = round(baseline_auc * 0.95, 3)  # alert if AUC falls by 5% below the baseline

    print(f"Baseline AUC at deployment : {baseline_auc:.4f}")
    print(f"Drift alert threshold      : {drift_threshold} (5% below baseline)\n")

    results = []

    with mlflow.start_run(run_name="Monitoring_6_Months"):
        mlflow.set_tag("run_type", "monitoring")
        mlflow.log_params({
            "monitoring_periods": n_periods,
            "drift_threshold":    drift_threshold,
            "baseline_auc":       round(baseline_auc, 4),
            "baseline_accuracy":  round(baseline_acc, 4)
        })

        for month in range(1, n_periods + 1):

            # Create synthetic drift for features by increasing noise each month
            X_shifted  = X_test.copy()
            noise_scale = 0.05 * month
            for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                noise = np.random.normal(0, noise_scale, len(X_shifted))
                X_shifted[col] = (X_shifted[col] * (1 + noise)).clip(lower=0)

            # Create synthetic drift for concepts by switching some labels each month
            y_shifted = y_test.values.copy()
            flip_mask = np.random.rand(len(y_shifted)) < (0.01 * month)
            y_shifted[flip_mask] = 1 - y_shifted[flip_mask]
            y_shifted = pd.Series(y_shifted)

            # Measure model performance on the data shifted for this month
            y_pred = model.predict(X_shifted)
            y_prob = model.predict_proba(X_shifted)[:, 1]

            month_auc = roc_auc_score(y_shifted, y_prob)
            month_acc = accuracy_score(y_shifted, y_pred)
            month_f1  = f1_score(y_shifted, y_pred, zero_division=0)
            drifted   = month_auc < drift_threshold

            # Log all metrics for each month into MLflow
            mlflow.log_metrics({
                f"month_{month:02d}_auc":      round(month_auc, 4),
                f"month_{month:02d}_accuracy": round(month_acc, 4),
                f"month_{month:02d}_f1":       round(month_f1, 4),
                f"month_{month:02d}_drift":    int(drifted),
            })

            # Tag the experiment run if any drift occurred during this month
            if drifted:
                mlflow.set_tag(f"drift_month_{month}", "ALERT")

            status = "DRIFT ALERT" if drifted else "OK"
            print(f"  Month {month}: AUC={month_auc:.4f} | "
                  f"Acc={month_acc:.4f} | F1={month_f1:.4f} | {status}")

            results.append({
                "month": month,
                "auc":   month_auc,
                "acc":   month_acc,
                "f1":    month_f1,
                "drift": drifted
            })

        # Create a two-chart monitoring dashboard and log it as an artifact
        months     = [r["month"] for r in results]
        aucs       = [r["auc"]   for r in results]
        accs       = [r["acc"]   for r in results]
        f1s        = [r["f1"]    for r in results]
        all_months = [0] + months
        all_aucs   = [baseline_auc] + aucs

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Top chart: AUC over time with the threshold zone
        ax1.plot(all_months, all_aucs, "b-o", linewidth=2, label="AUC-ROC")
        ax1.axhline(drift_threshold, color="red", linestyle="--",
                    linewidth=1.5, label=f"Alert threshold ({drift_threshold})")
        ax1.fill_between(all_months, drift_threshold, all_aucs,
                         where=[v >= drift_threshold for v in all_aucs],
                         alpha=0.1, color="green", label="Acceptable zone")
        ax1.fill_between(all_months, drift_threshold, all_aucs,
                         where=[v < drift_threshold for v in all_aucs],
                         alpha=0.15, color="red", label="Drift zone")
        ax1.set_xticks(all_months)
        ax1.set_xticklabels(["Deploy"] + [f"M{m}" for m in months])
        ax1.set_ylabel("AUC-ROC")
        ax1.set_title("Model AUC-ROC Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom chart: Comparison of all three metrics on one plot
        ax2.plot(all_months, all_aucs,                   "b-o", label="AUC-ROC")
        ax2.plot(all_months, [baseline_acc] + accs,      "g-s", label="Accuracy")
        ax2.plot(all_months, [baseline_f1]  + f1s,       "m-^", label="F1 Score")
        ax2.axhline(drift_threshold, color="red", linestyle="--", alpha=0.5)
        ax2.set_xticks(all_months)
        ax2.set_xticklabels(["Deploy"] + [f"M{m}" for m in months])
        ax2.set_ylabel("Score")
        ax2.set_title("All Metrics Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = "monitoring_dashboard.png"
        plt.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(chart_path)
        os.remove(chart_path)  # Delete the file copy locally after logging

        monitoring_run_id = mlflow.active_run().info.run_id

    # Output a brief analysis and re-training advice if applicable
    drift_months = [r["month"] for r in results if r["drift"]]
    print(f"\nBaseline AUC : {baseline_auc:.4f}")
    print(f"Final AUC    : {results[-1]['auc']:.4f}")
    if drift_months:
        print(f"Drift alerts : months {drift_months}")
        print("Recommendation: schedule model retraining")
    else:
        print("Drift alerts : none — model performance is stable")
    print(f"Run ID       : {monitoring_run_id}\n")

    return results


if __name__ == "__main__":
    print("Run this module via main.py")
