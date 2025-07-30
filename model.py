"""
Defines the ComposerPredictionModel class for handling audio feature extraction and predictions using a pre-trained model.
The trained model is loaded from the MLFlow Model Registry.
"""

import mlflow
import os
import pandas as pd
import numpy as np
import librosa
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure MLFlow tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ComposerPredictionModel:
    """
    Class to handle predictions and data preparation using a loaded model.
    """

    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version

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

    def load_model(self):
        """
        Load the trained model from MLFlow Model Registry marked as Production.
        """
        print("Loading model from MLFlow Model Registry...")
        logger.info("About to load model...")

        try:
            self.model = mlflow.sklearn.load_model(
                f"models:/{self.model_name}/{self.model_version}"
            )
            logger.info(
                f"Model loaded successfully from MLFlow Model Registry: {self.model_name}, version: {self.model_version}"
            )
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")

        return None

    def predict(self, features: pd.DataFrame):
        """
        Make predictions using the loaded model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        logger.info("Making predictions...")
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame.")

        return self.model.predict(features)
