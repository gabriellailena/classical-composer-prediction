import os
import pickle
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
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(mlflow_tracking_uri)    

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Global variable to store the model
model = None
model_name = 'BestRandomForestModel'
model_version = 'latest'

def load_model():
    """
    Load the trained model from MLFlow Model Registry marked as Production.
    """
    global model, model_name, model_version
    
    print("Loading model from MLFlow Model Registry...")
    logger.info("About to load model...")

    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
        logger.info(f"Model loaded successfully from MLFlow Model Registry: {model_name}, version: {model_version}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        print(f"Error loading model: {str(e)}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format. Expected a dictionary'}), 400

        features = data.get('features', None)
        if features is None:
            return jsonify({'error': 'Missing "features" key in request data'}), 400

        predictions = model.predict(features)
        
        response = {
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get model information.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_name': model_name,
        'model_version': model_version,
        'model_type': type(model).__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    if hasattr(model, 'feature_importances_'):
        info['n_features'] = len(model.feature_importances_)
    if hasattr(model, 'classes_'):
        info['classes'] = model.classes_.tolist()

    logger.info("Returning model info")
    return jsonify(info)


if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting application...")
    if not load_model():
        logger.error("Failed to load model. Exiting...")
        exit(1)
    
    # Start Flask app
    logger.info(f"Starting Flask app, is_model_loaded={model is not None}...")
    port = int(os.getenv('FLASK_PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # For production deployment (when not running directly)
    logger.info("Loading model for production deployment...")
    is_model_loaded = load_model()
    if not is_model_loaded:
        logger.error("Model could not be loaded for production deployment.")
    else:
        logger.info("Model loaded successfully for production deployment.")