import pytest
from unittest.mock import patch, MagicMock

with patch("model.ComposerPredictionModel.load_model", return_value=MagicMock()) as mock_load_model:
    import app

@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        yield client
