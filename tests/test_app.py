import io
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from flask import Response

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"<html" in response.data.lower() or b"<!doctype" in response.data.lower()

@patch('app.model')
def test_model_info(mock_model, client):
    mock_model.n_estimators = 10
    mock_model.feature_importances_ = [0.1, 0.2]
    mock_model.classes_ = np.array(['Mozart', 'Beethoven'])

    response = client.get('/model-info')
    assert response.status_code == 200
    assert 'model_name' in response.json
    assert response.json['classes'] == ['Mozart', 'Beethoven']

@patch('app.model_cls.extract_audio_features')
@patch('app.model_cls.predict')
@patch('app.log_prediction_response')
@patch('app.save_uploaded_file')
def test_predict_valid_audio(mock_save, mock_log, mock_predict, mock_extract, client):
    mock_extract.return_value = {"tempo": 120, "chroma": [0.1]*12}
    mock_predict.return_value = np.array(["Mozart"])
    mock_save.return_value = ("fake_path", "prod-1234", "wav")
    mock_log.return_value = None

    data = {
        'audio_file': (io.BytesIO(b"fake audio content"), 'test.wav')
    }

    response = client.post('/predict', content_type='multipart/form-data', data=data)
    assert response.status_code == 200
    assert response.json['composer'] == "Mozart"

def test_predict_empty_file(client):
    data = {
        'audio_file': (io.BytesIO(b''), 'empty.wav')
    }

    response = client.post('/predict', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    assert 'Empty file' in response.json['error']

def test_predict_invalid_file_type(client):
    data = {
        'audio_file': (io.BytesIO(b"some content"), 'track.txt')
    }

    response = client.post('/predict', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    assert 'Invalid file format' in response.json['error']

def test_predict_oversized_file(client):
    data = {
        'audio_file': (io.BytesIO(b"a" * (65 * 1024 * 1024)), 'large.wav')
    }

    response = client.post('/predict', content_type='multipart/form-data', data=data)
    assert response.status_code == 413
    assert 'exceeds the maximum limit' in response.json['error']

@patch('app.model_cls.predict')
def test_predict_json_batch(mock_predict, client):
    mock_predict.return_value = ["Beethoven", "Mozart"]
    payload = {
        "features": [
            {"tempo": 120, "chroma": [0.1]*12},
            {"tempo": 130, "chroma": [0.2]*12},
        ],
        "file_ids": ["prod-1234", "prod-5678"]
    }

    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json['results'][0]['composer'] == "Beethoven"
    assert response.json['results'][1]['file_id'] == "prod-5678"

def test_predict_invalid_json_format(client):
    response = client.post('/predict', json={"features": "not-a-list"})
    assert response.status_code == 400
    assert 'Features must be a list' in response.json['error']

@patch("app.os.listdir")
@patch("app.send_from_directory")
def test_get_evidently_report(mock_send_from_directory, mock_listdir, client):
    mock_listdir.return_value = ["report1.html", "report2.html"]
    mock_response = Response("Mock report content", status=200)
    mock_send_from_directory.return_value = mock_response

    response = client.get("/monitoring/report")

    assert response.status_code == 200
    assert b"Mock report content" in response.data
    mock_listdir.assert_called_once_with("/data/reports")
    mock_send_from_directory.assert_called_once_with("/data/reports", "report2.html")
