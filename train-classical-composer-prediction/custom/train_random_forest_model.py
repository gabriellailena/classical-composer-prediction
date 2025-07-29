from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import mlflow
import mlflow.sklearn
from scipy.stats import randint
from dotenv import load_dotenv
import os
import pandas as pd

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

# Configure MLflow tracking
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", None))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", None))


def evaluate_random_forest_model(
    model: RandomForestClassifier, test_df: pd.DataFrame, target_column: str
) -> None:
    """
    Evaluates the best model against the test data and logs the test score into MLFlow.

    Args:
        model: Trained Random Forest Classifier model
        test_df: Test DataFrame containing features and target variable
        target_column: Name of the target column in the DataFrame

    Returns:
        None (test score is logged to MLFlow)
    """
    # Ensure the target column exists
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Split the data into features and target
    exclude_columns = ["split", "file_id", target_column]
    X_test = test_df.drop(columns=exclude_columns)
    y_test = test_df[target_column]

    y_preds = model.predict(X_test)

    test_score = f1_score(y_test, y_preds, average="weighted")

    mlflow.log_metric("test_f1_weighted", test_score)
    print(f"Weighted F1-score on test dataset: {test_score}")

    return


@custom
def train_random_forest_model(*args, **kwargs) -> dict:
    """
    Train a Random Forest Classifier with hyperparameter optimization using RandomizedSearchCV.

    Args:
        train_df: DataFrame containing features and target variable
        target_column: Name of the target column in the DataFrame

    Returns:
        Dictionary containing the best model and results
    """
    train_df = args[0][0]
    test_df = args[0][1]
    target_column = "composer"

    # Ensure the target column exists
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Split the data into features and target
    exclude_columns = ["split", "file_id", target_column]
    X_train = train_df.drop(columns=exclude_columns)
    y_train = train_df[target_column]

    with mlflow.start_run(run_name="rf_train"):
        # Log basic info
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("optimization", "RandomizedSearchCV")
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])
        mlflow.log_param("n_classes", len(np.unique(y_train)))

        # Define hyperparameter search space
        param_distributions = {
            "n_estimators": randint(50, 500),
            "max_depth": [None] + list(range(10, 50, 5)),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7],
            "bootstrap": [True, False],
            "class_weight": ["balanced", "balanced_subsample", None],
        }

        # Log search space
        mlflow.log_param("param_distributions", str(param_distributions))

        # Initialize RandomForestClassifier
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Setup cross-validation
        n_splits = 10
        n_iters = 200
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Setup RandomizedSearchCV
        target_score = "f1_weighted"
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iters,
            cv=cv,
            scoring=target_score,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        # Log search parameters
        mlflow.log_param("n_iter", n_iters)
        mlflow.log_param("cv_folds", n_splits)
        mlflow.log_param("scoring", target_score)

        print("Starting hyperparameter search...")
        print(
            f"Testing {n_iters} parameter combinations with {n_splits}-fold CV = {n_iters * n_splits} total fits"
        )

        # Perform the search
        random_search.fit(X_train, y_train)

        # Get the best model
        best_model = random_search.best_estimator_

        # Log best parameters
        print("Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"{param}: {value}")
            mlflow.log_param(f"best_{param}", value)

        # Log best CV score
        mlflow.log_metric(f"best_cv_{target_score}", random_search.best_score_)

        # Log the best model
        mlflow.sklearn.log_model(
            best_model,
            name="models",
            signature=mlflow.models.infer_signature(X_train, y_train),
        )

        print(f"Best CV score ({target_score}): {random_search.best_score_:.4f}")

        # Evaluate the model on test set
        evaluate_random_forest_model(best_model, test_df, target_column)

    # Return optimized results and test dataframe for further evaluation
    return {
        "best_model": best_model,
        "best_params": random_search.best_params_,
        "best_cv_score": random_search.best_score_,
        "test_data": test_df,
    }


@test
def test_output(output, *args) -> None:
    """
    Test the output of the training function.
    """
    assert output is not None, "Training should return results"
    assert "best_model" in output, "Output should contain best model"
    assert "best_params" in output, "Output should contain best parameters"
    assert "best_cv_score" in output, "Output should contain best CV score"
    print("Test passed: Hyperparameter optimization completed successfully")
