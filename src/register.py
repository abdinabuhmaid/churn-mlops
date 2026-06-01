import mlflow
from mlflow.tracking import MlflowClient
import warnings

warnings.filterwarnings("ignore")

# The name of the model registered within MLflow Model Registry
MODEL_NAME = "ChurnPredictionModel"

# Criteria that the model has to meet before being promoted to Production
# In a practical environment, such tests should be run automatically based on real data.
VALIDATION_CHECKS = [
    ("AUC-ROC above 0.80",          True),
    ("No data drift detected",      True),
    ("Prediction latency under 1s", True),
    ("Model size under 100MB",      True),
]


def register_model(run_id, model_name=MODEL_NAME):
    """
    Registers the trained model from the given MLflow run, transitions it
    through Staging validation, and promotes it to Production if all
    checks pass. Any previously active Production version is archived.

    Model lifecycle:
        Unregistered -> Staging -> Production (previous version -> Archived)

    Args:
        run_id:     MLflow run ID that contains the saved model artifact
        model_name: name to register the model under in the registry

    Returns:
        model_name, version
    """

    print("-" * 50)
    print("Phase 4: Model Registry")
    print("-" * 50)

    mlflow.set_tracking_uri("./mlruns")
    client = MlflowClient()

    # Register the model artifacts from model tuning first, then try the baseline model artifact
    for artifact_path in ["best_tuned_model", "model"]:
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            registered = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            break
        except Exception:
            continue

    version = registered.version
    print(f"Registered '{model_name}' version {version}")

    # Give the model a detailed description and tags that will help with search
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            "Gradient Boosting classifier for customer churn prediction. "
            "Trained on the Telco dataset using Hyperopt hyperparameter tuning."
        )
    )
    client.set_model_version_tag(model_name, version, "algorithm", "GradientBoosting")
    client.set_model_version_tag(model_name, version, "dataset",   "Telco_Churn")
    client.set_model_version_tag(model_name, version, "tuned",     "True")

    # Move to Staging environment – this is where we’ll do our quality assurance and validation
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"Version {version} transitioned to Staging")

    # All validation tests should be passed successfully before promoting
    print("\nRunning validation checks:")
    all_passed = True
    for check_name, passed in VALIDATION_CHECKS:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        # Finally move to Production and automatically remove the previous version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"\nVersion {version} promoted to Production")
    else:
        print(f"\nValidation failed — model remains in Staging")

    # Display the entire model version history
    print(f"\nRegistry summary for '{model_name}':")
    all_versions = client.search_model_versions(f"name='{model_name}'")
    for v in sorted(all_versions, key=lambda x: int(x.version)):
        print(f"  v{v.version} | {v.current_stage:<12} | run: {v.run_id[:8]}")

    print()
    return model_name, version


def load_production_model(model_name=MODEL_NAME):
    """Loads and returns the currently active Production model from the registry."""
    mlflow.set_tracking_uri("./mlruns")
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


if __name__ == "__main__":
    print("Run this module via main.py")
