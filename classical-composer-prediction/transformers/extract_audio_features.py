import pandas as pd
import numpy as np
import librosa
import mlflow
from tqdm import tqdm
from typing import Dict, Optional
import os
from dotenv import load_dotenv

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


# Configure MLflow tracking
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", None))
mlflow.set_experiment(
    os.getenv("MLFLOW_EXPERIMENT_NAME", None)
)


def extract_features(
    audio_path: str,
    sample_rate: int,
    duration: Optional[int],
    offset: float,
    n_mfcc: int,
) -> Dict[str, float]:
    """
    Extract audio features from a single .wav file.

    Args:
        audio_path: Path to the audio file
        sample_rate: Sample rate for audio loading (in Hertz, i.e., samples per second)
        duration: Duration of audio to load (in seconds). If None, the full audio is loaded.
        offset: Offset (in seconds) to start loading audio
        n_mfcc: Number of MFCCs to return

    Returns:
        Dictionary of extracted features
    """
    try:
        # Load audio file
        y, sr = librosa.load(
            audio_path, sr=sample_rate, duration=duration, offset=offset
        )

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
        features["tempo"] = tempo

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

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return {}


def extract_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from multiple audio files.

    Args:
        df: DataFrame with metadata

    Returns:
        DataFrame with extracted features
    """
    feature_list = []

    params = {"sample_rate": 22050, "duration": 30, "offset": 0, "n_mfcc": 13}
    with mlflow.start_run():
        mlflow.log_param("sample_rate", params["sample_rate"])
        mlflow.log_param("duration", params["duration"])
        mlflow.log_param("offset", params["offset"])
        mlflow.log_param("n_mfcc", params["n_mfcc"])

        print(f"Extracting features from {len(df)} audio files...")

        for idx, row in tqdm(df.iterrows()):
            dataset_type = row["split"]
            if dataset_type == "train":
                audio_path = os.path.join(
                    "data", "raw", "musicnet", "train_data", f"{row['id']}.wav"
                )
            elif dataset_type == "test":
                audio_path = os.path.join(
                    "data", "raw", "musicnet", "test_data", f"{row['id']}.wav"
                )
            else:
                print(f"Skipping file {row['id']} with unknown split: {dataset_type}")
                continue
            features = extract_features(
                audio_path, sample_rate=22050, duration=30, offset=0, n_mfcc=13
            )

            if features:
                # Add metadata
                features["split"] = dataset_type
                features["file_id"] = row.get("id", None)
                features["composer"] = row.get("composer", "Unknown")

                feature_list.append(features)

        feature_df = pd.DataFrame(feature_list)
        print(f"Feature extraction complete. Shape: {feature_df.shape}")

        return feature_df


@transformer
def extract_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract audio features from a DataFrame containing audio metadata.

    Args:
        df: DataFrame with metadata

    Returns:
        DataFrame with extracted features
    """
    features = extract_features_batch(df)
    train_df = features.loc[features["split"] == "train"]
    test_df = features.loc[features["split"] == "test"]

    return train_df, test_df


@test
def test_output(output: pd.DataFrame, *args) -> None:
    """
    Test the output of the feature extraction.

    Args:
        output: DataFrame with extracted features
    """
    assert isinstance(output, pd.DataFrame), "Output should be a pandas DataFrame"
    assert "file_id" in output.columns, "Output DataFrame must contain file_id column"
    assert "composer" in output.columns, "Output DataFrame must contain composer column"
    print("Test passed: Output DataFrame is valid.")
