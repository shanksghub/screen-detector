"""
app.py - Flask server for Screen/Real detection
Deploys on Render: https://render.com
"""

import os
import sys
import base64
import time
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image

# Import your feature extractor
from feature_extraction_fast_16feat import extract_features_fast

app = Flask(__name__)

# Load model ONCE when server starts
print("=" * 60)
print("LOADING MODEL...")
print("=" * 60)

try:
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        # Try alternative paths
        alt_paths = ["model_16feat.joblib", "model_9feat.joblib"]
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                break
        else:
            raise FileNotFoundError("No model file found!")
    
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle.get("threshold", 0.329)
    feature_names = bundle.get("feature_names", [])
    
    print(f"✅ Model loaded: {model_path}")
    print(f"✅ Features: {len(feature_names)}")
    print(f"✅ Threshold: {threshold:.3f}")
    print("=" * 60)

except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


def predict_image(img):
    """Predict if image is REAL or SCREEN"""
    try:
        # Extract features
        feats = extract_features_fast(img).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        score = float(model.predict_proba(feats_scaled)[0][1])
        
        # Decision
        decision = "SCREEN" if score >= threshold else "REAL"
        confidence = score if decision == "SCREEN" else 1 - score
        
        return {
            "score": round(score, 4),
            "decision": decision,
            "threshold": round(threshold, 3),
            "confidence": round(confidence * 100, 1),
            "features": feats.shape[1]
        }
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    """Serve the webpage"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receive image from camera, predict, return result"""
    try:
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        # Predict
        result = predict_image(img)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "features": len(feature_names)
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)