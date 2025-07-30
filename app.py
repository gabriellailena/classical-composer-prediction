import os
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv

from model import ComposerPredictionModel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configuration for file uploads
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60MB max file size
ALLOWED_EXTENSIONS = {"wav", "mp3"}
UPLOAD_PATH = "/data/uploads/"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# Load model on startup
model_name = os.getenv("MODEL_NAME", "composer_prediction_model")
model_version = os.getenv("MODEL_VERSION", "1")

model_cls = ComposerPredictionModel(model_name, model_version)
model = model_cls.load_model()


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
            features = model_cls.extract_audio_features(file_path)
            predictions = model_cls.predict(pd.DataFrame([features]))

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
    if model is None:
        logger.error("Failed to load model. Exiting...")
        exit(1)

    # Start Flask app
    logger.info(f"Starting Flask app, is_model_loaded={model is not None}...")
    port = int(os.getenv("FLASK_PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # For production deployment (when not running directly)
    logger.info("Loading model for production deployment...")
    if model is None:
        logger.error("Model could not be loaded for production deployment.")
    else:
        logger.info("Model loaded successfully for production deployment.")
