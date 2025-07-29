import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import logging
import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure MLFlow tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configuration for file uploads
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60MB max file size
ALLOWED_EXTENSIONS = {"wav", "mp3"}
UPLOAD_PATH = "/data/uploads/"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# Global variable to store the model
model = None
model_name = "BestRandomForestModel"
model_version = "latest"


class ComposerPredictionModel:
    """
    Class to handle predictions and data preparation using a loaded model.
    """

    def __init__(self, model):
        self.model = model

    def extract_audio_features(self, audio_path):
        """
        Extract features from the given audio file.
        """
        sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", 22050))
        duration = int(os.getenv("AUDIO_DURATION", 30))
        offset = float(os.getenv("AUDIO_OFFSET", 0.0))
        n_mfcc = int(os.getenv("N_MFCC", 13))

        try:
            # Load audio file
            y, sr = librosa.load(
                audio_path, sr=sample_rate, duration=duration, offset=offset
            )
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            raise ValueError(f"Could not load audio file: {str(e)}")

        features = {}

        # --- SPECTRAL FEATURES ---
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(mfccs.shape[0]):
            features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
            features[f"mfcc_{i}_std"] = np.std(mfccs[i])
            features[f"mfcc_{i}_max"] = np.max(mfccs[i])
            features[f"mfcc_{i}_min"] = np.min(mfccs[i])

        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = np.mean(spectral_centroid)
        features["spectral_centroid_std"] = np.std(spectral_centroid)

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
        features["spectral_rolloff_std"] = np.std(spectral_rolloff)

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
        features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)

        # --- HARMONIC AND PERCUSSIVE FEATURES ---
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Harmonic-to-percussive ratio
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        features["harmonic_percussive_ratio"] = harmonic_energy / (
            percussive_energy + 1e-8
        )

        # --- TEMPO AND RHYTHM FEATURES ---
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = tempo[0]  # The returned tempo is always a 1-d array

        # Beat consistency (standard deviation of beat intervals)
        if len(beats) > 1:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times)
            features["beat_consistency"] = np.std(beat_intervals)
        else:
            features["beat_consistency"] = 0

        # --- CHROMA FEATURES (pitch class profiles) ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(chroma.shape[0]):
            features[f"chroma_{i}_mean"] = np.mean(chroma[i])
            features[f"chroma_{i}_std"] = np.std(chroma[i])

        # --- TONNETZ FEATURES (harmonic network) ---
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for i in range(tonnetz.shape[0]):
            features[f"tonnetz_{i}_mean"] = np.mean(tonnetz[i])
            features[f"tonnetz_{i}_std"] = np.std(tonnetz[i])

        # --- ENERGY AND DYNAMICS ---
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)
        features["rms_max"] = np.max(rms)

        # Dynamic range
        features["dynamic_range"] = np.max(rms) - np.min(rms)

        # --- ONSET FEATURES ---
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        features["onset_rate"] = len(onset_frames) / (len(y) / sr)  # onsets per second

        # --- SPECTRAL CONTRAST ---
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f"spectral_contrast_{i}_mean"] = np.mean(spectral_contrast[i])
            features[f"spectral_contrast_{i}_std"] = np.std(spectral_contrast[i])

        return features

    def predict(self, features):
        return self.model.predict(features)


def load_model():
    """
    Load the trained model from MLFlow Model Registry marked as Production.
    """
    global model, model_name, model_version

    print("Loading model from MLFlow Model Registry...")
    logger.info("About to load model...")

    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
        logger.info(
            f"Model loaded successfully from MLFlow Model Registry: {model_name}, version: {model_version}"
        )
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        print(f"Error loading model: {str(e)}")
        return False


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint
    """

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    pred_model = ComposerPredictionModel(model)
    if "audio_file" in request.files:
        file = request.files["audio_file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file.filename.split(".")[-1].lower() not in ALLOWED_EXTENSIONS:
            return jsonify(
                {
                    "error": f"Invalid file format. Supported formats: {list(ALLOWED_EXTENSIONS)}"
                }
            ), 400
        
        # Check file size
        file_bytes = file.read()
        if len(file_bytes) > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify({"error": "File size exceeds the maximum limit"}), 413

        # Read and inspect file content before saving
        logger.info(f"Uploaded file size: {len(file_bytes)} bytes")

        if len(file_bytes) == 0:
            logger.error("Uploaded file is empty!")
            return jsonify({"error": "Empty file uploaded"}), 400

        # Reset stream and save
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        file.stream.seek(0)
        file.save(file_path)

        # Extract features from the uploaded audio file
        try:
            logger.info(f"Extracting features from file: {file_path}")
            features = pred_model.extract_audio_features(file_path)
            predictions = pred_model.predict(pd.DataFrame([features]))

            response = {
                "predictions": predictions.tolist(),
                "timestamp": datetime.now().isoformat(),
            }

            return jsonify(response)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"error": "Failed to parse request"}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """
    Get model information.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    info = {
        "model_name": model_name,
        "model_version": model_version,
        "model_type": type(model).__name__,
        "timestamp": datetime.now().isoformat(),
    }

    if hasattr(model, "n_estimators"):
        info["n_estimators"] = model.n_estimators
    if hasattr(model, "feature_importances_"):
        info["n_features"] = len(model.feature_importances_)
    if hasattr(model, "classes_"):
        info["classes"] = model.classes_.tolist()

    logger.info("Returning model info")
    return jsonify(info)


if __name__ == "__main__":
    # Load model on startup
    logger.info("Starting application...")
    if not load_model():
        logger.error("Failed to load model. Exiting...")
        exit(1)

    # Start Flask app
    logger.info(f"Starting Flask app, is_model_loaded={model is not None}...")
    port = int(os.getenv("FLASK_PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # For production deployment (when not running directly)
    logger.info("Loading model for production deployment...")
    is_model_loaded = load_model()
    if not is_model_loaded:
        logger.error("Model could not be loaded for production deployment.")
    else:
        logger.info("Model loaded successfully for production deployment.")
