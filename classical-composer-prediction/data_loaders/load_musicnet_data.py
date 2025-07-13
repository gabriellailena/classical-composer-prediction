import os
import pandas as pd
import requests

if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_musicnet_data(*args, **kwargs) -> pd.DataFrame:
    """
    Reads the MusicNet metadata and constructs file paths to the corresponding audio files.

    Returns:
        DataFrame containing information on train/test split
    """
    data_dir = os.path.join(os.getcwd(), "data", "raw")

    # Read metadata
    metadata_df = pd.read_csv(os.path.join(data_dir, "musicnet_metadata.csv"))
    if metadata_df.empty:
        raise ValueError("Metadata file is empty or not found.")

    # Mark specific ids as train or test
    # NOTE: Casting to integer as the metadata ids are integer type
    train_ids = [
        int(file.replace(".wav", ""))
        for file in os.listdir(os.path.join(data_dir, "musicnet", "train_data"))
    ]
    test_ids = [
        int(file.replace(".wav", ""))
        for file in os.listdir(os.path.join(data_dir, "musicnet", "test_data"))
    ]

    splits = []
    for data_id in metadata_df["id"]:
        if data_id in train_ids:
            splits.append("train")
        elif data_id in test_ids:
            splits.append("test")
        else:
            print(f"Warning: {data_id} not found in train or test data.")
            splits.append(None)

    metadata_df["split"] = splits

    print("MusicNet metadata loaded successfully.")
    print(
        f"Number of training samples: {len(metadata_df[metadata_df['split'] == 'train'])}"
    )
    print(
        f"Number of testing samples: {len(metadata_df[metadata_df['split'] == 'test'])}"
    )
    return metadata_df


@test
def test_output(output, *args) -> None:
    """
    Test the output of the MusicNet data loader.
    """
    assert isinstance(output, pd.DataFrame), "Output should be a pandas DataFrame"
    assert output is not None, "The output is undefined"
    assert "split" in output.columns, 'Output DataFrame should contain a "split" column'
    assert output["split"].notnull().all(), (
        'All entries in the "split" column should be defined'
    )
