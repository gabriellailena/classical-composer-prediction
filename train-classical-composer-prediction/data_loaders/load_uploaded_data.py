import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_uploaded_data(*args, **kwargs):
    """
    Loads the uploaded audio files to be used to monitor input and prediction drift.
    """
    data_dir = os.path.join(os.getcwd(), "data", "uploads")

    # Create a metadata file with the file_id and file extension
    metadata = []
    for file in os.listdir(data_dir):
        metadata.append(
            {
                "id": file.split('.')[0].lower(),
                "extension": file.split('.')[-1].lower(),
            }
        )

    return pd.DataFrame(metadata)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
