import mlflow
import os
from dotenv import load_dotenv

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

# Configure MLflow tracking
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", None))
mlflow.set_experiment(
    os.getenv("MLFLOW_EXPERIMENT_NAME", None)
)

@custom
def register_random_forest_model(*args, **kwargs):
    """
    Finds the run with best test score and registers it.
    """
    experiment = mlflow.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME", None))
    best_run = mlflow.search_runs(
        [experiment.experiment_id], order_by=["metrics.test_f1_weighted DESC"]
    )
    best_run_id = best_run.loc[0, 'run_id']

    model_details = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/models", name="BestRandomForestModel"
    )
    print(f"Model registered successfully: {model_details.name} with version {model_details.version}")
    
    # Set to production stage
    mlflow.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model {model_details.name} version {model_details.version} transitioned to Production stage.")
    
    return {
        "model_name": model_details.name,
        "model_version": model_details.version
    }
    

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert "model_name" in output, "Output should contain 'model_name'"
    assert "model_version" in output, "Output should contain 'model_version'"
