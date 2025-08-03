import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from model import ComposerPredictionModel


@patch("mlflow.sklearn.load_model")
def test_load_model_success(mock_load_model):
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    model = ComposerPredictionModel("test_model", "1")
    loaded_model = model.load_model()

    assert loaded_model == mock_model
    assert model.model == mock_model


@patch("mlflow.sklearn.load_model", side_effect=Exception("mock failure"))
def test_load_model_failure(mock_load_model):
    model = ComposerPredictionModel("test_model", "1")
    result = model.load_model()
    assert result is None


def test_predict_success():
    model = ComposerPredictionModel("test_model", "1")
    mock_model = MagicMock()
    mock_model.predict.return_value = ["Mozart"]
    model.model = mock_model

    df = pd.DataFrame([{"feature1": 0.5, "feature2": 1.2}])
    result = model.predict(df)

    assert result == ["Mozart"]
    mock_model.predict.assert_called_once()


def test_predict_with_no_model_loaded():
    model = ComposerPredictionModel("test_model", "1")
    df = pd.DataFrame([{"feature1": 0.5}])

    with pytest.raises(AttributeError, match="object has no attribute 'model'"):
        model.predict(df)


def test_predict_invalid_input():
    model = ComposerPredictionModel("test_model", "1")
    model.model = MagicMock()

    with pytest.raises(ValueError, match="Features must be a pandas DataFrame"):
        model.predict([{"feature1": 0.5}])


@patch("model.librosa.load")
@patch("model.librosa.feature")
@patch("model.librosa.effects.hpss")
@patch("model.librosa.beat.beat_track")
@patch("model.librosa.frames_to_time")
@patch("model.librosa.onset.onset_detect")
def test_extract_audio_features(
    mock_onset,
    mock_frames_to_time,
    mock_beat_track,
    mock_hpss,
    mock_feature,
    mock_load
):
    # Setup mocks
    mock_load.return_value = (np.ones(22050), 22050)  # 1 second of fake audio

    mock_feature.mfcc.return_value = np.array([[1.0]*10]*13)
    mock_feature.spectral_centroid.return_value = np.array([[0.5]*10])
    mock_feature.spectral_rolloff.return_value = np.array([[0.5]*10])
    mock_feature.spectral_bandwidth.return_value = np.array([[0.5]*10])
    mock_feature.zero_crossing_rate.return_value = np.array([[0.1]*10])
    mock_hpss.return_value = (np.array([0.5]*10), np.array([0.1]*10))
    mock_beat_track.return_value = ([120.0], [1, 2, 3])
    mock_frames_to_time.return_value = [0.0, 0.5, 1.0]
    mock_feature.chroma_stft.return_value = np.array([[0.2]*10]*12)
    mock_feature.tonnetz.return_value = np.array([[0.1]*10]*6)
    mock_feature.rms.return_value = np.array([[0.5]*10])
    mock_onset.return_value = [1, 2, 3]
    mock_feature.spectral_contrast.return_value = np.array([[0.3]*10]*7)

    model = ComposerPredictionModel("test_model", "1")
    features = model.extract_audio_features("fake_path.wav")

    assert isinstance(features, dict)
    assert "mfcc_0_mean" in features
    assert "tempo" in features
    assert "onset_rate" in features
    assert "spectral_contrast_0_std" in features
