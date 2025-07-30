import os
import pandas as pd
import numpy as np
import requests

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

from datetime import datetime
from dotenv import load_dotenv

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

load_dotenv()

@custom
def predict(reference_data: dict, current_data: dict, **kwargs) -> dict:
    """
    Fetches model from the registry and predict composer on the provided reference and current data.
    """
    # Get reference data features
    reference_data_df = reference_data.get("test")
    reference_data_df["file_id"] = reference_data_df["file_id"].astype(str)
    reference_file_ids = reference_data_df["file_id"].to_list()
    reference_features_df = reference_data_df.drop(columns=["split", "file_id", "composer"])
    X_reference = reference_features_df.to_dict(orient='records')

    current_data_df = current_data.get("prod")
    current_data_df["file_id"] = current_data_df["file_id"].astype(str)
    current_file_ids = current_data_df["file_id"].to_list()
    current_features_df = current_data_df.drop(columns=["file_id"])
    X_current = current_features_df.to_dict(orient='records')

    # Run prediction
    url = "http://api:8000/predict"
    ref_response = requests.post(url, json={"features": X_reference, "file_ids": reference_file_ids})
    ref_preds = pd.DataFrame(ref_response.json().get("results"))
    curr_response = requests.post(url, json={"features": X_current, "file_ids": current_file_ids})
    curr_preds = pd.DataFrame(curr_response.json().get("results"))

    # Add back the file_id column
    reference_features_df["file_id"] = reference_file_ids
    current_features_df["file_id"] = current_file_ids

    # Add predictions to the DataFrames
    reference_features_df = pd.merge(reference_features_df, ref_preds, on="file_id")
    current_features_df = pd.merge(current_features_df, curr_preds, on="file_id")

    # Run Evidently Report for data drift and prediction drift
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    numerical_columns = [
        col for col in current_features_df.columns
        if col not in ["composer", "file_id"]
    ]
    column_mapping = ColumnMapping()
    column_mapping.prediction = "composer"
    column_mapping.numerical_features = numerical_columns

    report.run(
        reference_data=reference_features_df,
        current_data=current_features_df,
        column_mapping=column_mapping
    )

    report_dir = "/data/reports/"
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, f"evidently_report_{datetime.now().date()}.html")
    print(report_path)
    report.save_html(report_path)

    print(f"Report saved to {report_path}")

    return {
        "reference_predictions": ref_preds,
        "current_predictions": curr_preds,
        "report_path": report_path
    }



@test
def test_output(output, *args) -> None:
    """
    Test the output of the training function.
    """
    assert output is not None, "Prediction should return results"
    print("Test passed: Prediction completed successfully")
