import os
import hashlib
import json
from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv

from model import ComposerPredictionModel
from werkzeug.exceptions import RequestEntityTooLarge

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
PREDICTION_LOG_PATH = "/data/predictions/"

# Load model on startup
model_name = os.getenv("MODEL_NAME", "composer_prediction_model")
model_version = os.getenv("MODEL_VERSION", "1")

model_cls = ComposerPredictionModel(model_name, model_version)
model = model_cls.load_model()


def string_to_code(s):
    """
    Convert a string to a 4-digit code with 'prod' prefix using SHA256 hash. This is used to ensure uploaded files have unique identifiers.
    The code is zero-padded to always be 4 digits.
    """
    hash_object = hashlib.sha256(s.encode())
    hash_int = int(hash_object.hexdigest(), 16)

    # Take modulo 10000 to get a 4-digit number (0–9999)
    code = hash_int % 10000

    return f"prod-{code:04d}"


def log_prediction_response(file_id, response):
    """
    Log the prediction response to a file.
    """
    os.makedirs(PREDICTION_LOG_PATH, exist_ok=True)
    log_file_path = os.path.join(PREDICTION_LOG_PATH, f"{file_id}.json")
    with open(log_file_path, "w") as log_file:
        json.dump(response, log_file)

    logger.info(f"Logged prediction response to {log_file_path}")


def save_uploaded_file(file):
    """
    Save the uploaded file to the upload directory and return its path.
    """
    file_ext = file.filename.split(".")[-1].lower()
    file_id = string_to_code(file.filename.split(".")[0])  # Generate unique ID

    os.makedirs(UPLOAD_PATH, exist_ok=True)
    file_path = os.path.join(UPLOAD_PATH, file_id + "." + file_ext)
    file.stream.seek(0)
    file.save(file_path)

    logger.info(f"Saved uploaded file to {file_path}")
    return file_path, file_id, file_ext

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File size exceeds the maximum limit"}), 413

@app.route("/", methods=["GET", "POST"])
def index():
    logger.info("Rendering index page...")
    return render_template("index.html")


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

    # Handle single file uploads
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
        logger.info(f"Uploaded file size: {len(file_bytes)} bytes")
        if len(file_bytes) > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify({"error": "File size exceeds the maximum limit"}), 413

        if len(file_bytes) == 0:
            logger.error("Uploaded file is empty!")
            return jsonify({"error": "Empty file uploaded"}), 400

        # Save the uploaded file
        file_path, file_id, file_ext = save_uploaded_file(file)

        # Extract features from the uploaded audio file
        try:
            logger.info(f"Extracting features from file: {file_path}")
            features = model_cls.extract_audio_features(file_path)
            predictions = model_cls.predict(pd.DataFrame([features]))

            response = {
                "status": "success",
                "file_id": file_id,
                "file_extension": file_ext,
                "file_name": file.filename,
                "composer": predictions.tolist()[0],
                "timestamp": datetime.now().isoformat(),
            }

            # Log the prediction
            log_prediction_response(file_id, response)

            return jsonify(response)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Handle batch predictions
    elif "features" in request.json:
        features = request.json.get("features", [])
        file_ids = request.json.get("file_ids", [])

        if not isinstance(features, list):
            return jsonify({"error": "Features must be a list"}), 400

        predictions = model_cls.predict(pd.DataFrame(features))

        # Pair each file_id with a prediction
        results = []
        for fid, pred in zip(file_ids, predictions):
            results.append({"file_id": fid, "composer": pred})

        return jsonify({"status": "success", "results": results})

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


@app.route("/monitoring/report", methods=["GET"])
def get_evidently_report():
    report_dir = "/data/reports"
    report_filename = sorted(os.listdir(report_dir))[-1]  # latest report
    return send_from_directory(report_dir, report_filename)


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
